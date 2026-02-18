"""Parallel rollout collection using multiprocessing."""
import torch
import numpy as np
import chess
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import os
import multiprocessing as mp
import warnings

# Set spawn method for CUDA compatibility (must be called before creating processes)
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, which is fine
    pass

from rl.env import ChessEnv, legal_action_mask
from rl.opponents import OpponentBase
from .buffers import RolloutBuffer, Transition


def collect_single_episode_worker(args):
    """
    Collect one episode in subprocess.

    This function is pickled and sent to worker process.

    Args:
        args: tuple of (policy_state_dict, opponent_config, device, seed,
                       use_shaped_reward, shaped_reward_coef, stockfish_depth,
                       use_accuracy_reward)

    Returns:
        tuple: (transitions_list, total_return, episode_length)
    """
    (policy_state, opponent_config, device, seed,
     use_shaped_reward, shaped_reward_coef, stockfish_depth,
     use_accuracy_reward) = args
    
    # Suppress Stockfish output
    import subprocess
    import sys
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    try:
        # Re-initialize in subprocess with proper imports
        import torch
        import numpy as np
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        from rl.env import ChessEnv, legal_action_mask
        from rl.models import CheckpointLoader
        from rl.opponents import RandomOpponent, StockfishOpponent
        
        # Load policy on CPU in worker
        policy = CheckpointLoader.load_chessformer(device='cpu')
        policy.load_state_dict(policy_state)
        policy.eval()
        
        # Create opponent
        if opponent_config['type'] == 'random':
            opponent = RandomOpponent()
        elif opponent_config['type'] == 'stockfish':
            opponent = StockfishOpponent(depth=opponent_config.get('depth', 5))
        else:
            opponent = RandomOpponent()
        
        opponent.reset()
        
        # Collect episode
        env = ChessEnv()
        obs, info = env.reset()
        
        transitions = []
        total_return = 0.0
        episode_length = 0
        done = False
        
        while not done:
            mask = legal_action_mask(env.board)
            
            if env.board.turn == chess.WHITE:
                # Agent's turn
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    output = policy.forward(obs_tensor)
                    if isinstance(output, tuple):
                        logits, value = output
                        if isinstance(value, torch.Tensor) and value.dim() == 1:
                            value = value.unsqueeze(-1)
                    else:
                        logits = output
                        value = torch.tensor([[0.0]])
                
                masked_logits = logits.clone()
                masked_logits[0, mask == 0] = -float('inf')
                probs = torch.softmax(masked_logits, dim=-1)
                
                action = torch.multinomial(probs.squeeze(0), num_samples=1).item()
                logprob = torch.log(probs[0, action] + 1e-10).item()
                
                # Store transition
                transitions.append({
                    'obs': obs.copy(),
                    'action': action,
                    'logprob': logprob,
                    'value': value.item() if isinstance(value, torch.Tensor) else 0.0,
                    'mask': mask.copy(),
                })
                
                # Step with configured reward mode
                obs_next, reward, terminated, truncated, info = env.step(
                    action,
                    use_shaped_reward=use_shaped_reward,
                    shaped_reward_coef=shaped_reward_coef,
                    stockfish_depth=stockfish_depth,
                    use_accuracy_reward=use_accuracy_reward,
                )
                done = terminated or truncated
                total_return += reward
            else:
                # Opponent's turn
                action = opponent.get_action(env.board)
                obs_next, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_return -= reward
            
            obs = obs_next
            episode_length += 1
        
        return transitions, total_return, episode_length
    
    except Exception as e:
        # Return empty episode on error
        print(f"Error in worker: {e}")
        return [], 0.0, 0


class ParallelRolloutCollector:
    """Collects trajectories in parallel processes."""
    
    def __init__(
        self,
        policy,
        opponent_pool: List[OpponentBase],
        device: str = 'cpu',
        num_workers: int = 4,
        use_shaped_reward: bool = False,
        shaped_reward_coef: float = 0.01,
        stockfish_depth: int = 3,
        use_accuracy_reward: bool = False,
    ):
        """
        Initialize parallel rollout collector.

        Args:
            policy: agent policy network
            opponent_pool: list of opponent instances
            device: torch device (main process)
            num_workers: number of worker processes
            use_shaped_reward: whether to use Stockfish eval-delta rewards
            shaped_reward_coef: coefficient for position rewards
            stockfish_depth: Stockfish depth for evaluation
            use_accuracy_reward: use per-move accuracy reward instead of terminal
        """
        self.policy = policy
        self.opponent_pool = opponent_pool
        self.device = device
        self.num_workers = num_workers
        self.use_shaped_reward = use_shaped_reward
        self.shaped_reward_coef = shaped_reward_coef
        self.stockfish_depth = stockfish_depth
        self.use_accuracy_reward = use_accuracy_reward
    
    def collect_rollouts(self, num_episodes: int = 32) -> RolloutBuffer:
        """Collect episodes in parallel."""
        print(f"\n[Parallel Mode] Collecting {num_episodes} episodes ({self.num_workers} workers)...")
        
        buffer = RolloutBuffer(capacity=num_episodes * 512)
        
        # Move policy to CPU for pickling (CUDA tensors can't be pickled efficiently)
        policy_cpu = self.policy.__class__.__new__(self.policy.__class__)
        policy_state = {}
        for k, v in self.policy.state_dict().items():
            if isinstance(v, torch.Tensor):
                policy_state[k] = v.cpu()
            else:
                policy_state[k] = v
        
        # Prepare worker arguments
        args_list = []
        for i in range(num_episodes):
            opponent = np.random.choice(self.opponent_pool)
            opponent_config = {
                'type': 'random' if 'random' in type(opponent).__name__.lower() else 'stockfish',
            }
            if hasattr(opponent, 'depth'):
                opponent_config['depth'] = opponent.depth
            
            args_list.append((
                policy_state,
                opponent_config,
                'cpu',  # Workers use CPU
                np.random.randint(0, 1000000),  # Unique seed per worker
                self.use_shaped_reward,
                self.shaped_reward_coef,
                self.stockfish_depth,
                self.use_accuracy_reward,
            ))
        
        # Collect in parallel using spawn method
        completed = 0
        start_time = time.time()
        
        try:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {
                    executor.submit(collect_single_episode_worker, args): i 
                    for i, args in enumerate(args_list)
                }
                
                for future in as_completed(futures):
                    try:
                        transitions, ret, length = future.result()
                        
                        # Add to buffer
                        for trans in transitions:
                            buffer.add(Transition(
                                obs=trans['obs'],
                                action=trans['action'],
                                reward=0.0,  # Will be computed from returns
                                next_obs=trans['obs'],  # Placeholder
                                done=False,
                                logprob=trans['logprob'],
                                value=trans['value'],
                                mask=trans['mask'],
                            ))
                        
                        buffer.end_episode(ret, length)
                        
                    except Exception as e:
                        print(f"  ✗ Episode failed: {e}")
                    
                    completed += 1
                    if completed % max(1, num_episodes // 10) == 0 or completed == 1:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        eta = (num_episodes - completed) / rate if rate > 0 else 0
                        print(f"  Episode {completed:3d}/{num_episodes} ({rate:.1f} eps/s, ETA {eta:.0f}s)")
        
        except Exception as e:
            print(f"✗ Parallel collection failed: {e}")
            import traceback
            traceback.print_exc()
            # Return partial buffer
        
        elapsed = time.time() - start_time
        print(f"✓ Collected {num_episodes} episodes in {elapsed:.1f}s, {len(buffer.transitions)} transitions\n")
        return buffer
