"""Random move opponent."""
import numpy as np
import chess

from .base import OpponentBase
from rl.env import legal_action_mask


class RandomOpponent(OpponentBase):
    """Opponent that plays random legal moves."""

    def __init__(self):
        self.name = "random"

    def get_action(self, board: chess.Board) -> int:
        """Choose a uniformly random legal action."""
        mask = legal_action_mask(board)
        legal_actions = np.where(mask)[0]
        return int(np.random.choice(legal_actions))

    def reset(self):
        """No state to reset."""
        pass

    def get_name(self) -> str:
        return self.name
