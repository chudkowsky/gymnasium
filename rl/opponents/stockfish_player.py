"""Stockfish UCI engine opponent."""
import subprocess
import chess
import sys
from pathlib import Path

from .base import OpponentBase
from rl.env import legal_action_mask, action_to_move


class StockfishOpponent(OpponentBase):
    """Opponent that plays using Stockfish UCI engine."""

    def __init__(
        self,
        stockfish_path: str = "stockfish",
        depth: int = 10,
        movetime_ms: int = None,
        name: str = None,
    ):
        """
        Initialize Stockfish opponent.
        
        Args:
            stockfish_path: path to stockfish binary
            depth: search depth (higher = stronger but slower)
            movetime_ms: time limit per move in milliseconds (overrides depth)
            name: optional name for logging
        """
        self.stockfish_path = stockfish_path
        self.depth = depth
        self.movetime_ms = movetime_ms
        self.name = name or f"stockfish(d={depth})"
        
        # Test that stockfish is available
        try:
            proc = subprocess.Popen(
                [stockfish_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            # Send quit command
            proc.stdin.write("quit\n")
            proc.stdin.flush()
            proc.wait(timeout=5)
        except Exception as e:
            raise RuntimeError(f"Failed to start Stockfish at {stockfish_path}: {e}")
        
        self.process = None

    def _start_engine(self):
        """Start Stockfish engine process."""
        if self.process is not None:
            return
        
        self.process = subprocess.Popen(
            [self.stockfish_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        
        # Send UCI command
        self.process.stdin.write("uci\n")
        self.process.stdin.flush()
        
        # Wait for uciok
        while True:
            line = self.process.stdout.readline()
            if "uciok" in line:
                break

    def _stop_engine(self):
        """Stop Stockfish engine process."""
        if self.process is None:
            return
        
        try:
            self.process.stdin.write("quit\n")
            self.process.stdin.flush()
            self.process.wait(timeout=5)
        except:
            self.process.kill()
        finally:
            self.process = None

    def get_action(self, board: chess.Board) -> int:
        """Choose action using Stockfish analysis."""
        self._start_engine()
        
        try:
            # Set position
            fen = board.fen()
            self.process.stdin.write(f"position fen {fen}\n")
            self.process.stdin.flush()
            
            # Go command
            if self.movetime_ms is not None:
                self.process.stdin.write(f"go movetime {self.movetime_ms}\n")
            else:
                self.process.stdin.write(f"go depth {self.depth}\n")
            self.process.stdin.flush()
            
            # Parse bestmove from output
            bestmove = None
            while True:
                line = self.process.stdout.readline()
                if "bestmove" in line:
                    parts = line.split()
                    bestmove = parts[1]  # e.g., "e2e4"
                    break
            
            # Convert UCI move to action
            move = chess.Move.from_uci(bestmove)
            
            # Import here to avoid circular imports
            from rl.env import move_to_action
            action = move_to_action(move, board)
            
            return action
        
        except Exception as e:
            print(f"Error getting Stockfish move: {e}", file=sys.stderr)
            # Fallback to random legal move
            mask = legal_action_mask(board)
            legal_actions = [i for i in range(len(mask)) if mask[i]]
            import numpy as np
            return int(np.random.choice(legal_actions))

    def reset(self):
        """Reset engine state (clear board history)."""
        self._stop_engine()

    def get_name(self) -> str:
        return self.name

    def __del__(self):
        """Cleanup on deletion."""
        self._stop_engine()
