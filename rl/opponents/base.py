"""Base class for opponents in chess RL."""
from abc import ABC, abstractmethod
import chess


class OpponentBase(ABC):
    """Abstract base class for chess opponents."""

    @abstractmethod
    def get_action(self, board: chess.Board) -> int:
        """
        Choose an action given a board state.
        
        Args:
            board: python-chess Board object
            
        Returns:
            action: integer in [0, 4100)
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset opponent state (e.g., for stateful opponents)."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return opponent name for logging."""
        pass
