"""
Adapter for ChessTransformer model to work with our RL framework.

Converts ChessTransformer's (64, 2) output to (4100,) logits compatible with our action space.
"""

import torch
import torch.nn as nn
import numpy as np
import chess
from typing import Optional, Tuple
import sys

# Action space constants (avoid circular imports)
N_ACTIONS = 4100
N_BASE_MOVES = 4096
PROMO_OFFSET = 4096


class ChessformerAdapter(nn.Module):
    """
    Wraps ChessTransformer and converts its output to our action space.
    
    ChessTransformer accepts:
    - Input: (B, 64) with piece indices [0, 12]
    - Output: (B, 64, 2) with [from_score, to_score] for each square
    
    This adapter converts to our format:
    - Input: (B, 15, 8, 8) - our 15-plane encoding, automatically converted to tokens
    - Output: (B, 4100) - logits over discrete actions
    """
    
    def __init__(self, chessformer_model: nn.Module, device: str = 'cpu', n_frozen_layers: int = 10):
        """
        Initialize adapter with pretrained ChessTransformer.

        Args:
            chessformer_model: Trained ChessTransformer instance
            device: torch device
            n_frozen_layers: how many of the 12 encoder layers to freeze (default: 10,
                             so only layers 10-11 and linear_output are fine-tuned)
        """
        super().__init__()
        self.transformer = chessformer_model.to(device)
        self.device = torch.device(device)

        # --- Layer freezing ---
        # Freeze everything first, then selectively unfreeze the last few encoder
        # layers + the output projection. This prevents RL's noisy gradients from
        # destroying the pretrained chess knowledge baked into earlier layers.
        for param in self.transformer.parameters():
            param.requires_grad = False

        # Unfreeze last (12 - n_frozen_layers) encoder layers
        n_total = 12
        for i in range(n_frozen_layers, n_total):
            for param in self.transformer.transformer_encoder.layers[i].parameters():
                param.requires_grad = True

        # Always fine-tune the output projection (it maps to our action space)
        for param in self.transformer.linear_output.parameters():
            param.requires_grad = True

        n_trainable = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        n_total_params = sum(p.numel() for p in self.transformer.parameters())
        print(f"  Transformer: {n_trainable:,} / {n_total_params:,} params trainable "
              f"(layers {n_frozen_layers}-11 + linear_output)")

        # --- Value head with rich hidden-state features ---
        # Hook the last encoder layer to capture its (B, 64, d_model) output,
        # which is far richer than the raw (B, 64, 2) scores used before.
        self._last_hidden: torch.Tensor | None = None
        self._hook_handle = None
        self._register_hidden_hook()
        d_model = 512  # known from ChessTransformer architecture
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        ).to(device)

    def _register_hidden_hook(self) -> None:
        """Register (or re-register) the forward hook that captures the last
        encoder layer's hidden state for the value head.  Must be called after
        deepcopy because PyTorch does not preserve hooks across deepcopy."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
        self._last_hidden = None
        self._hook_handle = self.transformer.transformer_encoder.layers[-1].register_forward_hook(
            lambda _m, _i, o: setattr(self, '_last_hidden', o)
        )
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through adapter.

        Args:
            obs: (B, 15, 8, 8) observation tensor [plane representation]
                or (B, 64) token tensor [direct token format]

        Returns:
            (logits, values) where:
            - logits: (B, 4100) action logits
            - values: (B,) value estimates from trainable value head
        """
        batch_size = obs.shape[0]

        # If input is 15-plane, convert to tokens
        if obs.dim() == 4 and obs.shape[1:] == (15, 8, 8):
            tokens = self._planes_to_tokens(obs)
        elif obs.dim() == 2 and obs.shape[1] == 64:
            tokens = obs.long()
        else:
            raise ValueError(f"Unexpected obs shape: {obs.shape}")

        tokens = tokens.to(self.device)

        # Forward through transformer — gradients flow for fine-tuning
        output = self.transformer(tokens)  # (B, 64, 2)

        # Convert (B, 64, 2) to (B, 4100) logits
        logits = self._scores_to_logits(output)  # (B, 4100)

        # Value head: use pooled last-encoder-layer hidden state (B, 64, 512)
        # captured by the forward hook — much richer than the raw (B, 64, 2) scores.
        hidden = self._last_hidden  # (B, 64, 512)
        features = hidden.mean(dim=1)  # (B, 512) — global average pool over squares
        values = self.value_head(features).squeeze(-1)  # (B,)

        return logits, values
    
    def _planes_to_tokens(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Convert 15-plane representation to 64-token representation.

        Planes 0-5:  White pieces (P=1, N=2, B=3, R=4, Q=5, K=6)
        Planes 6-11: Black pieces (p=7, n=8, b=9, r=10, q=11, k=12)
        Plane 12+:   Auxiliary (ignored here)

        Args:
            obs: (B, 15, 8, 8)

        Returns:
            (B, 64) token indices in [0, 12]
        """
        batch_size = obs.shape[0]

        # (B, 12, 8, 8) -> (B, 12, 64)
        piece_planes = obs[:, :12, :, :].reshape(batch_size, 12, 64)

        # Token values: plane 0 -> 1, plane 1 -> 2, ..., plane 11 -> 12
        token_ids = torch.arange(1, 13, device=obs.device, dtype=torch.long).view(1, 12, 1)

        # For each square, sum token_id where the plane is 1 (at most one piece per square)
        tokens = ((piece_planes > 0.5).long() * token_ids).sum(dim=1)  # (B, 64)

        return tokens
    
    def _scores_to_logits(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Convert (B, 64, 2) transformer output to (B, 4100) action logits.

        For each base move action i = from*64 + to:
            logit[i] = from_score[from] + to_score[to]

        Args:
            scores: (B, 64, 2) with [from_scores, to_scores]

        Returns:
            (B, 4100) logits over all actions
        """
        from_scores = scores[:, :, 0]  # (B, 64)
        to_scores = scores[:, :, 1]    # (B, 64)

        # Vectorized outer sum: base_logits[b, from, to] = from_scores[b, from] + to_scores[b, to]
        # (B, 64, 1) + (B, 1, 64) -> (B, 64, 64) -> (B, 4096)
        base_logits = from_scores.unsqueeze(2) + to_scores.unsqueeze(1)
        base_logits = base_logits.reshape(scores.shape[0], N_BASE_MOVES)  # (B, 4096)

        # Promotion logits: use max from_score as proxy for all 4 promotion types
        promo_logits = from_scores.max(dim=1, keepdim=True)[0].expand(-1, 4)  # (B, 4)

        return torch.cat([base_logits, promo_logits], dim=1)  # (B, 4100)


class ChessformerPolicyWrapper(nn.Module):
    """Wrapper for inference with ChessformerAdapter."""
    
    def __init__(self, adapter: ChessformerAdapter, device: str = 'cpu'):
        """
        Initialize policy wrapper.
        
        Args:
            adapter: ChessformerAdapter instance
            device: torch device
        """
        super().__init__()
        self.adapter = adapter.to(device)
        self.device = torch.device(device)
        self.adapter.eval()
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        return self.adapter(obs)
    
    def predict_action(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        deterministic: bool = False,
        return_value: bool = False,
    ):
        """
        Predict action with masking.
        
        Args:
            obs: (B, 15, 8, 8) or (15, 8, 8) tensor
            mask: (B, 4100) or (4100,) binary mask
            deterministic: if True, use argmax
            return_value: if True, also return value
        
        Returns:
            action (int or Tensor), or (action, value)
        """
        # Ensure batch dimension
        single_obs = obs.dim() == 3
        if single_obs:
            obs = obs.unsqueeze(0)
            mask = mask.unsqueeze(0)
        
        obs = obs.to(self.device)
        mask = mask.to(self.device)
        
        with torch.no_grad():
            logits, value = self.adapter(obs)
        
        # Apply mask
        mask_float = mask.float()
        mask_sum = mask_float.sum(dim=1)
        assert torch.all(mask_sum > 0), "Mask must have at least one legal action"
        
        logits = logits + torch.log(mask_float + 1e-10)
        
        # Sample or argmax
        if deterministic:
            action = torch.argmax(logits, dim=1)
        else:
            probs = torch.nn.functional.softmax(logits, dim=1)
            probs = torch.where(mask_float > 0, probs, torch.zeros_like(probs))
            probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-10)
            action = torch.multinomial(probs, num_samples=1).squeeze(1)
        
        if single_obs:
            action = action.item()
        
        if return_value:
            return action, value
        return action
