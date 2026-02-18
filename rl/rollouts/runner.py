"""Rollout collection for PPO training."""
import torch
import numpy as np
import chess
from typing import List, Tuple

from rl.env import ChessEnv, board_to_tensor, legal_action_mask
from rl.opponents import OpponentBase
from .buffers import RolloutBuffer, Transition


class RolloutCollector:
    """Collects trajectories from environment interactions."""

    def __init__(
        self,
        policy,  # agent policy
        opponent_pool: List[OpponentBase],
        device: str = "cpu",
        num_envs: int = 1,
        use_shaped_reward: bool = False,
        shaped_reward_coef: float = 0.01,
        stockfish_depth: int = 3,
    ):
        """
        Initialize rollout collector.
        
        Args:
            policy: agent policy for sampling actions
            opponent_pool: list of opponent instances
            device: torch device
            num_envs: number of parallel environments (currently fixed at 1)
            use_shaped_reward: whether to use Stockfish-based position rewards
            shaped_reward_coef: coefficient for position rewards (e.g., 0.01)
            stockfish_depth: Stockfish search depth for position evaluation
        """
        self.policy = policy
        self.opponent_pool = opponent_pool
        self.device = device
        self.num_envs = num_envs
        self.use_shaped_reward = use_shaped_reward
        self.shaped_reward_coef = shaped_reward_coef
        self.stockfish_depth = stockfish_depth
        
        # For now, simple single-env collection
        self.env = ChessEnv()

    def collect_episode(self) -> Tuple[float, int, RolloutBuffer]:
        """
        Collect one full episode.
        
        Returns:
            total_return: cumulative reward from agent perspective
            episode_length: number of steps
            buffer: RolloutBuffer with transitions
        """
        buffer = RolloutBuffer(capacity=512)
        
        # Pick opponent
        opponent = np.random.choice(self.opponent_pool)
        opponent.reset()
        
        obs, info = self.env.reset()
        agent_perspective = 0  # White plays first
        
        total_return = 0.0
        episode_length = 0
        done = False
        
        while not done:
            mask = legal_action_mask(self.env.board)
            
            # Agent's turn (white)
            if self.env.board.turn == chess.WHITE:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    self.policy.eval()
                    output = self.policy.forward(obs_tensor)
                    if isinstance(output, tuple):
                        logits, value = output
                        if isinstance(value, torch.Tensor) and value.dim() == 1:
                            value = value.unsqueeze(-1)
                    else:
                        logits = output
                        value = torch.tensor([[0.0]], device=self.device)

                # Masked softmax for policy
                masked_logits = logits.clone()
                masked_logits[0, mask == 0] = -float('inf')
                probs = torch.softmax(masked_logits, dim=-1)

                # Sample action
                action = torch.multinomial(probs.squeeze(0), num_samples=1).item()
                logprob = torch.log(probs[0, action] + 1e-10).item()
                value_scalar = value.item() if isinstance(value, torch.Tensor) else 0.0

                # Step environment first so we can store real reward, next_obs, done
                obs_next, reward, terminated, truncated, info_next = self.env.step(
                    action,
                    use_shaped_reward=self.use_shaped_reward,
                    shaped_reward_coef=self.shaped_reward_coef,
                    stockfish_depth=self.stockfish_depth,
                )
                done = terminated or truncated
                total_return += reward

                # Store complete transition with real reward/next_obs/done
                transition = Transition(
                    obs=obs,
                    action=action,
                    reward=reward,
                    next_obs=obs_next,
                    done=done,
                    logprob=logprob,
                    value=value_scalar,
                    mask=mask,
                )
                buffer.add(transition)
            else:
                # Opponent's turn (black)
                action = opponent.get_action(self.env.board)
                obs_next, reward, terminated, truncated, info_next = self.env.step(action)
                done = terminated or truncated
                total_return -= reward  # Opponent's reward is negative for agent
            
            obs = obs_next
            episode_length += 1
        
        buffer.end_episode(total_return, episode_length)
        return total_return, episode_length, buffer

    def collect_rollouts(self, num_episodes: int = 32) -> RolloutBuffer:
        """
        Collect multiple episodes.
        
        Args:
            num_episodes: number of episodes to collect
            
        Returns:
            combined buffer with all transitions
        """
        combined_buffer = RolloutBuffer(capacity=num_episodes * 512)
        
        print(f"\nCollecting {num_episodes} episodes...")
        for ep in range(num_episodes):
            ret, length, buffer = self.collect_episode()
            
            # Add all transitions from this episode
            for transition in buffer.transitions:
                combined_buffer.add(transition)
            
            combined_buffer.end_episode(ret, length)
            
            # Progress update every 10 episodes
            if (ep + 1) % 10 == 0 or ep == 0:
                print(f"  Episode {ep + 1}/{num_episodes} complete (len={length}, ret={ret:.2f})")
        
        print(f"✓ Collected {num_episodes} episodes, {len(combined_buffer.transitions)} transitions\n")
        return combined_buffer
