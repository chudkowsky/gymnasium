"""Rollout buffers for trajectory collection."""
import numpy as np
import torch
from typing import NamedTuple


class Transition(NamedTuple):
    """Single step transition."""
    obs: np.ndarray  # (15, 8, 8)
    action: int
    reward: float
    next_obs: np.ndarray  # (15, 8, 8)
    done: bool
    logprob: float
    value: float
    mask: np.ndarray  # (4100,) legal action mask


class RolloutBuffer:
    """Stores trajectories for PPO updates."""

    def __init__(self, capacity: int = 2048):
        """
        Initialize rollout buffer.
        
        Args:
            capacity: max transitions to store before update
        """
        self.capacity = capacity
        self.transitions = []
        self.episode_returns = []
        self.episode_lengths = []

    def add(self, transition: Transition):
        """Add a transition."""
        self.transitions.append(transition)

    def start_episode(self):
        """Mark start of new episode."""
        pass

    def end_episode(self, ret: float, length: int):
        """Mark end of episode with return and length."""
        self.episode_returns.append(ret)
        self.episode_lengths.append(length)

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return len(self.transitions) >= self.capacity

    def get_all(self):
        """Get all transitions as tensors."""
        if not self.transitions:
            return None

        obses = torch.stack([torch.tensor(t.obs, dtype=torch.float32) for t in self.transitions])
        actions = torch.tensor([t.action for t in self.transitions], dtype=torch.long)
        rewards = torch.tensor([t.reward for t in self.transitions], dtype=torch.float32)
        next_obses = torch.stack([torch.tensor(t.next_obs, dtype=torch.float32) for t in self.transitions])
        dones = torch.tensor([t.done for t in self.transitions], dtype=torch.float32)
        logprobs = torch.tensor([t.logprob for t in self.transitions], dtype=torch.float32)
        values = torch.tensor([t.value for t in self.transitions], dtype=torch.float32)
        masks = torch.stack([torch.tensor(t.mask, dtype=torch.float32) for t in self.transitions])

        return {
            'obs': obses,
            'actions': actions,
            'rewards': rewards,
            'next_obs': next_obses,
            'dones': dones,
            'logprobs': logprobs,
            'values': values,
            'masks': masks,
        }

    def compute_gae(self, values_next: torch.Tensor, gamma: float = 0.99, lam: float = 0.95):
        """
        Compute generalized advantage estimation (GAE).
        
        Args:
            values_next: value at terminal state
            gamma: discount factor
            lam: GAE lambda
            
        Returns:
            advantages: (N,) advantages
            returns: (N,) discounted returns
        """
        data = self.get_all()
        rewards = data['rewards']
        values = data['values']
        dones = data['dones']

        advantages = []
        gae = 0.0
        next_value = values_next

        # Compute backwards
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = values_next
                done = dones[t]
            else:
                next_value_t = values[t + 1]
                done = dones[t]

            delta = rewards[t] + gamma * next_value_t * (1 - done) - values[t]
            gae = delta + gamma * lam * (1 - done) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + values
        return advantages, returns

    def clear(self):
        """Clear buffer."""
        self.transitions.clear()

    def stats(self):
        """Get buffer statistics."""
        if not self.episode_returns:
            return {}
        return {
            'n_transitions': len(self.transitions),
            'n_episodes': len(self.episode_returns),
            'mean_return': np.mean(self.episode_returns),
            'mean_length': np.mean(self.episode_lengths),
        }
