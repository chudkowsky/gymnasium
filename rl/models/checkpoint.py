"""
Model checkpoint loader and policy wrapper.

This module handles loading pretrained transformer checkpoints
and provides inference utilities.

Supports both:
- SimpleTransformerPolicy (placeholder for testing)
- ChessTransformer (your real pretrained model via adapter)
"""

import torch
import torch.nn as nn
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any


class SimpleTransformerPolicy(nn.Module):
    """
    Simple transformer-based policy for chess.
    
    This is a placeholder architecture. Replace with your actual model.
    
    Input: obs tensor (B, 15, 8, 8)
    Output: logits (B, 4100) and optional value head (B,)
    """
    
    def __init__(self, n_actions: int = 4100, hidden_dim: int = 256, device: str = 'cpu'):
        super().__init__()
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.device_str = device
        
        # Simple CNN feature extractor
        self.conv1 = nn.Conv2d(15, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Global average pooling + FC
        # Output: 128 features after pooling
        self.fc_hidden = nn.Linear(128, hidden_dim)
        
        # Policy head (logits)
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        
        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, obs: torch.Tensor) -> tuple:
        """
        Forward pass.
        
        Args:
            obs: (B, 15, 8, 8) observation tensor
        
        Returns:
            (logits, value) where:
            - logits: (B, n_actions)
            - value: (B,)
        """
        # Feature extraction
        x = torch.relu(self.conv1(obs))
        x = torch.relu(self.conv2(x))
        
        # Global average pooling
        x = torch.mean(x, dim=(2, 3))  # (B, 128)
        
        # Hidden layer
        x = torch.relu(self.fc_hidden(x))
        
        # Heads
        logits = self.policy_head(x)  # (B, n_actions)
        value = self.value_head(x).squeeze(-1)  # (B,)
        
        return logits, value


class CheckpointLoader:
    """Load and manage model checkpoints."""
    
    @staticmethod
    def load_chessformer(
        checkpoint_path: str = 'data/checkpoints/pretrained.pth',
        device: str = 'cpu'
    ):
        """
        Load real ChessTransformer model with adapter.
        
        Args:
            checkpoint_path: Path to .pth checkpoint
            device: torch device
        
        Returns:
            ChessformerPolicyWrapper ready for inference
        """
        from .chessformer_adapter import ChessformerAdapter, ChessformerPolicyWrapper
        from chessformer import ChessTransformer
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load model
        model = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Create adapter
        adapter = ChessformerAdapter(model, device=device)
        
        # Return policy wrapper
        return ChessformerPolicyWrapper(adapter, device=device)
    
    @staticmethod
    def load_checkpoint(
        checkpoint_path: str,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to .pth file
            device: torch device
        
        Returns:
            Dict with 'model', 'config', 'metadata'
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        return {
            'state_dict': checkpoint.get('state_dict', checkpoint),
            'config': checkpoint.get('config', {}),
            'metadata': checkpoint.get('metadata', {}),
        }
    
    @staticmethod
    def save_checkpoint(
        model: nn.Module,
        path: str,
        config: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ):
        """Save a model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'state_dict': model.state_dict(),
            'config': config or {},
            'metadata': metadata or {},
        }
        
        torch.save(checkpoint, path)


class PolicyWrapper:
    """Wrapper around a policy model for easy inference."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
    ):
        """
        Initialize the policy wrapper.
        
        Args:
            model: nn.Module (policy + value heads)
            device: torch device
        """
        self.model = model.to(device)
        self.device = torch.device(device)
        self.model.eval()
    
    def forward(self, obs: torch.Tensor) -> tuple:
        """
        Forward pass with automatic device handling.
        
        Args:
            obs: (B, 15, 8, 8) tensor or (15, 8, 8) single obs
        
        Returns:
            (logits, value)
        """
        # Handle single observation
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        
        obs = obs.to(self.device)
        
        with torch.no_grad():
            logits, value = self.model(obs)
        
        return logits, value
    
    def predict_action(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        deterministic: bool = False,
        return_value: bool = False,
    ):
        """
        Predict an action from observation with action masking.
        
        Args:
            obs: (B, 15, 8, 8) or (15, 8, 8) tensor
            mask: (B, 4100) or (4100,) binary mask of legal actions
            deterministic: if True, use argmax; else sample
            return_value: if True, also return value
        
        Returns:
            action (int or torch.Tensor), or (action, value) if return_value=True
        """
        # Ensure batch dimension
        single_obs = obs.dim() == 3
        if single_obs:
            obs = obs.unsqueeze(0)
            mask = mask.unsqueeze(0)
        
        obs = obs.to(self.device)
        mask = mask.to(self.device)
        
        with torch.no_grad():
            logits, value = self.model(obs)
        
        # Apply action mask (set illegal actions to -inf)
        mask_float = mask.float()
        
        # Check if mask has any legal actions
        mask_sum = mask_float.sum(dim=1)  # (B,)
        assert torch.all(mask_sum > 0), "Mask must have at least one legal action per batch item"
        
        # Add log mask to logits (handles masking)
        logits = logits + torch.log(mask_float + 1e-10)
        
        # Sample or argmax
        if deterministic:
            action = torch.argmax(logits, dim=1)
        else:
            # Sample from categorical distribution
            # Use numerically stable log-softmax + exp
            probs = torch.nn.functional.softmax(logits, dim=1)
            # Ensure probabilities are still valid (no NaNs if mask was used)
            probs = torch.where(mask_float > 0, probs, torch.zeros_like(probs))
            probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-10)
            action = torch.multinomial(probs, num_samples=1).squeeze(1)
        
        if single_obs:
            action = action.item()
        
        if return_value:
            return action, value
        return action
