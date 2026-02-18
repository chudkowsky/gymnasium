"""Snapshot checkpoint opponent."""
import torch
import chess

from .base import OpponentBase
from rl.models import CheckpointLoader
from rl.env import legal_action_mask


class SnapshotOpponent(OpponentBase):
    """Opponent that plays using a checkpoint policy."""

    def __init__(self, checkpoint_path: str, device: str = "cpu", name: str = None):
        """
        Initialize snapshot opponent.
        
        Args:
            checkpoint_path: path to .pth checkpoint
            device: torch device
            name: optional name for logging
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.policy = CheckpointLoader.load_chessformer(device=device)
        self.policy.eval()
        self.name = name or f"snapshot({checkpoint_path.split('/')[-1]})"

    def get_action(self, board: chess.Board) -> int:
        """Choose action using policy with masked sampling."""
        # Prepare observation
        from rl.env import board_to_tensor
        obs = board_to_tensor(board)
        obs = obs.unsqueeze(0).to(self.device)  # (1, 15, 8, 8)
        
        # Get mask
        mask = legal_action_mask(board)
        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            logits = self.policy(obs)  # (1, 4100)
        
        # Greedy (deterministic) policy
        logits = logits.squeeze(0)
        masked_logits = logits.clone()
        masked_logits[mask == 0] = -float('inf')
        action = masked_logits.argmax(dim=-1).cpu().item()
        
        return int(action)

    def reset(self):
        """No state to reset."""
        pass

    def get_name(self) -> str:
        return self.name
