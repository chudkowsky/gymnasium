"""PPO trainer for chess RL."""
import torch
import torch.optim as optim
from pathlib import Path
import json
from typing import Dict, Any
import numpy as np

from rl.rollouts import RolloutCollector, ParallelRolloutCollector
from rl.opponents import OpponentBase
from .losses import compute_policy_loss, compute_value_loss, compute_entropy_loss


class PPOTrainer:
    """Proximal Policy Optimization trainer for chess."""

    def __init__(
        self,
        policy,
        opponent_pool,
        config: Dict[str, Any],
        device: str = "cpu",
        collector_config: Dict[str, Any] = None,
    ):
        """
        Initialize PPO trainer.
        
        Args:
            policy: actor-critic policy
            opponent_pool: list of OpponentBase opponents
            config: training configuration dict
            device: torch device
            collector_config: rollout collection configuration
        """
        self.policy = policy
        self.opponent_pool = opponent_pool
        self.device = device
        self.config = config
        
        # Training hyperparams
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.n_epochs = config.get('n_epochs', 3)
        self.batch_size = config.get('batch_size', 32)
        self.clip_ratio = config.get('clip_ratio', 0.2)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_coef = config.get('value_coef', 0.5)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        
        # Collector configuration
        if collector_config is None:
            collector_config = {}
        
        # Extract accuracy reward config (takes precedence over shaped reward)
        accuracy_reward_config = collector_config.get('accuracy_reward', {})
        use_accuracy_reward = accuracy_reward_config.get('enabled', False)
        accuracy_stockfish_depth = accuracy_reward_config.get('stockfish_depth', 5)

        # Extract shaped reward config
        shaped_reward_config = collector_config.get('shaped_reward', {})
        use_shaped_reward = shaped_reward_config.get('enabled', False) and not use_accuracy_reward
        shaped_reward_coef = shaped_reward_config.get('position_reward_coef', 0.01)
        shaped_stockfish_depth = shaped_reward_config.get('stockfish_depth', 3)

        # Unified Stockfish depth: accuracy reward uses its own depth
        stockfish_depth = accuracy_stockfish_depth if use_accuracy_reward else shaped_stockfish_depth

        # Extract parallel config
        parallel_config = collector_config.get('parallel', {})
        use_parallel = parallel_config.get('enabled', False)
        num_workers = parallel_config.get('num_workers', 4)

        # Create appropriate collector
        if use_parallel:
            self.collector = ParallelRolloutCollector(
                policy,
                opponent_pool,
                device='cpu',  # Workers use CPU
                num_workers=num_workers,
                use_shaped_reward=use_shaped_reward,
                shaped_reward_coef=shaped_reward_coef,
                stockfish_depth=stockfish_depth,
                use_accuracy_reward=use_accuracy_reward,
            )
        else:
            self.collector = RolloutCollector(
                policy,
                opponent_pool,
                device=device,
                num_envs=1,
                use_shaped_reward=use_shaped_reward,
                shaped_reward_coef=shaped_reward_coef,
                stockfish_depth=stockfish_depth,
                use_accuracy_reward=use_accuracy_reward,
            )
        
        # Logging
        self.num_updates = 0
        self.num_steps = 0

    def train_epoch(self, buffer, num_minibatches: int = 4):
        """
        One PPO update epoch.

        Args:
            buffer: RolloutBuffer with transitions
            num_minibatches: number of minibatches to split data into

        Returns:
            metrics: dict of training metrics
        """
        # Use eval mode to disable dropout during training — critical for PPO consistency.
        # The transformer has 0.5 dropout which would make logprobs inconsistent between
        # rollout (eval/no_grad) and training (formerly train/grad), causing huge KL.
        # eval() only disables dropout/batchnorm — it does NOT block gradient computation.
        self.policy.eval()

        data = buffer.get_all()
        if data is None:
            return {}
        
        # Compute advantages with next value estimate
        obs_next = data['next_obs'][-1].unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.policy.forward(obs_next)
            if isinstance(output, tuple):
                _, value_next = output
            else:
                value_next = torch.tensor([[0.0]], device=self.device)
            if isinstance(value_next, torch.Tensor) and value_next.dim() == 1:
                value_next = value_next.unsqueeze(-1)
        
        advantages, returns = buffer.compute_gae(value_next, gamma=self.gamma, lam=self.gae_lambda)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Move to device
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)
        
        # Collect metrics
        metrics = {
            'mean_advantage': advantages.mean().item(),
            'std_advantage': advantages.std().item(),
        }
        
        # Mini-batch updates
        n_samples = len(advantages)
        indices = np.arange(n_samples)
        
        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                obs_batch = data['obs'][batch_indices]
                actions_batch = data['actions'][batch_indices]
                old_logprobs_batch = data['logprobs'][batch_indices]
                masks_batch = data['masks'][batch_indices]
                advantages_batch = advantages[batch_indices]
                returns_batch = returns[batch_indices]
                
                # Forward pass (single call)
                output = self.policy.forward(obs_batch)
                if isinstance(output, tuple):
                    logits, values = output
                    if isinstance(values, torch.Tensor) and values.dim() == 1:
                        values = values.unsqueeze(-1)
                else:
                    logits = output
                    values = torch.zeros(len(obs_batch), 1, device=self.device)
                
                # Compute losses
                policy_loss, approx_kl = compute_policy_loss(
                    logits, actions_batch, advantages_batch,
                    old_logprobs_batch, masks_batch,
                    clip_ratio=self.clip_ratio
                )
                
                value_loss = compute_value_loss(
                    values.squeeze(-1), returns_batch
                )
                
                entropy_loss = compute_entropy_loss(logits, masks_batch)
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track metrics
                metrics['policy_loss'] = policy_loss.item()
                metrics['value_loss'] = value_loss.item()
                metrics['entropy_loss'] = entropy_loss.item()
                metrics['total_loss'] = loss.item()
                metrics['approx_kl'] = approx_kl.item()
        
        self.num_updates += 1
        return metrics

    def train_step(self, num_episodes: int = 32) -> Dict[str, Any]:
        """
        One training step: collect rollouts and update policy.
        
        Args:
            num_episodes: number of episodes to collect
            
        Returns:
            metrics: dict with training statistics
        """
        # Collect rollouts
        buffer = self.collector.collect_rollouts(num_episodes)
        self.num_steps += buffer.stats().get('n_transitions', 0)
        
        # Get buffer stats
        buf_stats = buffer.stats()
        
        # PPO update
        print(f"Running PPO update (n_epochs={self.n_epochs})...")
        update_metrics = self.train_epoch(buffer)
        
        # Combine metrics
        metrics = {
            'step': self.num_steps,
            'update': self.num_updates,
            **buf_stats,
            **update_metrics,
        }
        
        return metrics

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.state_dict(), path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Checkpoint loaded: {path}")
