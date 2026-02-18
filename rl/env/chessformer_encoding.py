"""
Encoding for ChessTransformer model - converts board to 64-token representation.

ChessTransformer uses a different encoding than the 15-plane format:
- Input: 64 squares with piece indices (0-12)
- Piece mapping: . (0), P (1-6), p (7-12) for white/black pieces
"""

import chess
import torch
import numpy as np
from typing import Optional


# Piece index mapping for ChessTransformer
PIECE_TO_TOKEN = {
    '.': 0,     # Empty square
    'P': 1,     # White Pawn
    'N': 2,     # White Knight
    'B': 3,     # White Bishop
    'R': 4,     # White Rook
    'Q': 5,     # White Queen
    'K': 6,     # White King
    'p': 7,     # Black Pawn
    'n': 8,     # Black Knight
    'b': 9,     # Black Bishop
    'r': 10,    # Black Rook
    'q': 11,    # Black Queen
    'k': 12,    # Black King
}

TOKEN_TO_PIECE = {v: k for k, v in PIECE_TO_TOKEN.items()}


def board_to_tokens(board: chess.Board, flip_perspective: bool = True) -> np.ndarray:
    """
    Convert chess board to 64-token representation for ChessTransformer.
    
    Args:
        board: chess.Board instance
        flip_perspective: If True, flip board to always show from current player's perspective
    
    Returns:
        np.ndarray of shape (64,) with dtype int64, values in [0, 12]
    """
    tokens = np.zeros(64, dtype=np.int64)
    
    # Get board string representation
    board_str = str(board).replace('\n', '')
    
    # Flip for black's perspective if needed
    if flip_perspective and not board.turn:
        # Black to move: flip the board
        lines = [board_str[i:i+8] for i in range(0, len(board_str), 8)][::-1]
        board_str = ''.join(lines).swapcase()
    
    # Convert to tokens
    for i, char in enumerate(board_str):
        tokens[i] = PIECE_TO_TOKEN.get(char, 0)
    
    return tokens


def board_to_tokens_tensor(
    board: chess.Board,
    flip_perspective: bool = True,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Convert board to token tensor for ChessTransformer.
    
    Returns:
        torch.Tensor of shape (1, 64) int64 (batch size 1)
    """
    if device is None:
        device = torch.device('cpu')
    
    tokens = board_to_tokens(board, flip_perspective=flip_perspective)
    return torch.from_numpy(tokens).unsqueeze(0).to(device)


def batch_board_to_tokens(
    boards: list,
    flip_perspective: bool = True,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Convert batch of boards to token tensors.
    
    Returns:
        torch.Tensor of shape (B, 64) int64
    """
    if device is None:
        device = torch.device('cpu')
    
    batch_size = len(boards)
    tokens_batch = np.zeros((batch_size, 64), dtype=np.int64)
    
    for i, board in enumerate(boards):
        tokens_batch[i] = board_to_tokens(board, flip_perspective=flip_perspective)
    
    return torch.from_numpy(tokens_batch).to(device)
