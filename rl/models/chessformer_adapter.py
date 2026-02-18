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
    
    def __init__(self, chessformer_model: nn.Module, device: str = 'cpu'):
        """
        Initialize adapter with pretrained ChessTransformer.
        
        Args:
            chessformer_model: Trained ChessTransformer instance
            device: torch device
        """
        super().__init__()
        self.transformer = chessformer_model.to(device)
        self.device = torch.device(device)
        self.transformer.eval()
        
        # Freeze transformer weights (inference only)
        for param in self.transformer.parameters():
            param.requires_grad = False
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through adapter.
        
        Args:
            obs: (B, 15, 8, 8) observation tensor [plane representation]
                or (B, 64) token tensor [direct token format]
        
        Returns:
            (logits, values) where:
            - logits: (B, 4100) action logits
            - values: (B,) dummy value head (all zeros) since transformer doesn't have it
        """
        batch_size = obs.shape[0]
        
        # If input is 15-plane, convert to tokens
        if obs.dim() == 4 and obs.shape[1:] == (15, 8, 8):
            # Convert 15-plane to tokens
            # This is a simplified conversion; real boards would need proper reconstruction
            tokens = self._planes_to_tokens(obs)
        elif obs.dim() == 2 and obs.shape[1] == 64:
            # Already tokens
            tokens = obs.long()
        else:
            raise ValueError(f"Unexpected obs shape: {obs.shape}")
        
        tokens = tokens.to(self.device)
        
        # Forward through transformer
        with torch.no_grad():
            output = self.transformer(tokens)  # (B, 64, 2)
        
        # Convert (B, 64, 2) to (B, 4100) logits
        logits = self._scores_to_logits(output)  # (B, 4100)
        
        # Dummy value head (transformer doesn't have one)
        values = torch.zeros(batch_size, device=self.device)
        
        return logits, values
    
    def _planes_to_tokens(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Convert 15-plane representation to 64-token representation.
        
        This is a heuristic conversion since we lose information in 15-plane format.
        For accuracy, consider modifying ChessEnv to output tokens directly.
        
        Args:
            obs: (B, 15, 8, 8)
        
        Returns:
            (B, 64) token indices
        """
        batch_size = obs.shape[0]
        tokens = torch.zeros(batch_size, 64, dtype=torch.long, device=obs.device)
        
        # Piece mapping: planes 0-11 contain piece placements
        piece_map = [
            'P', 'N', 'B', 'R', 'Q', 'K',  # White pieces (planes 0-5)
            'p', 'n', 'b', 'r', 'q', 'k',  # Black pieces (planes 6-11)
        ]
        piece_to_token = {
            'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
            'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12,
        }
        
        for b in range(batch_size):
            for sq in range(64):
                row = sq // 8
                col = sq % 8
                
                # Check each piece plane
                found = False
                for plane_idx, piece_char in enumerate(piece_map):
                    if obs[b, plane_idx, row, col] > 0.5:
                        tokens[b, sq] = piece_to_token[piece_char]
                        found = True
                        break
                
                if not found:
                    tokens[b, sq] = 0  # Empty
        
        return tokens
    
    def _scores_to_logits(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Convert (B, 64, 2) transformer output to (B, 4100) action logits.
        
        For each move action i = from*64 + to:
            logit[i] = score[from, 0] + score[to, 1]
        
        Args:
            scores: (B, 64, 2) with [from_scores, to_scores]
        
        Returns:
            (B, 4100) logits over all actions
        """
        batch_size = scores.shape[0]
        device = scores.device
        
        # Extract from and to scores
        from_scores = scores[:, :, 0]  # (B, 64)
        to_scores = scores[:, :, 1]    # (B, 64)
        
        # Compute logits for all base moves
        # logit[from*64 + to] = from_score[from] + to_score[to]
        logits = torch.zeros(batch_size, N_BASE_MOVES, device=device)
        
        for from_sq in range(64):
            for to_sq in range(64):
                if from_sq != to_sq:
                    action = from_sq * 64 + to_sq
                    logits[:, action] = from_scores[:, from_sq] + to_scores[:, to_sq]
        
        # For promotion moves: use best pawn promotion available
        # Simple heuristic: use max of (from_pawn, to_promotion_target)
        promo_logits = torch.max(from_scores, dim=1)[0]  # (B,)
        promo_logits = promo_logits.unsqueeze(1).expand(batch_size, 4)
        
        # Concatenate: (B, 4096 + 4)
        full_logits = torch.cat([logits, promo_logits], dim=1)  # (B, 4100)
        
        return full_logits


class ChessformerPolicyWrapper:
    """Wrapper for inference with ChessformerAdapter."""
    
    def __init__(self, adapter: ChessformerAdapter, device: str = 'cpu'):
        """
        Initialize policy wrapper.
        
        Args:
            adapter: ChessformerAdapter instance
            device: torch device
        """
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
