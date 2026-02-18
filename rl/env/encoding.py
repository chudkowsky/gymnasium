"""
Chess board encoding to tensor representation.

Standard representation:
- 12 planes for piece placement (6 piece types × 2 colors)
  Planes 0-5: white pieces (pawn, knight, bishop, rook, queen, king)
  Planes 6-11: black pieces
- 1 plane for side-to-move (1.0 if white, 0.0 if black)
- 1 plane for repetition count (typically 0 or small)
- 1 plane for halfmove clock (for 50-move rule, scaled)

Total: 15 planes of 8x8 = 120 squares

Output shape: (15, 8, 8) as float32
"""

import chess
import numpy as np
import torch
from typing import Union, Optional


# Piece type to plane index mapping (white 0-5, black 6-11)
PIECE_TO_PLANE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

N_PIECE_PLANES = 12  # 6 piece types * 2 colors
N_SIDE_PLANES = 1    # side-to-move
N_AUX_PLANES = 2     # repetition count, halfmove clock
N_OBS_PLANES = N_PIECE_PLANES + N_SIDE_PLANES + N_AUX_PLANES  # 15

OBS_SHAPE = (N_OBS_PLANES, 8, 8)


def board_to_tensor(board: chess.Board, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Convert a chess board to a tensor representation.
    
    Args:
        board: chess.Board instance
        device: torch device (e.g., 'cpu', 'cuda'). If None, uses CPU.
    
    Returns:
        torch.Tensor of shape (15, 8, 8) with dtype float32
    """
    if device is None:
        device = torch.device('cpu')
    
    # Initialize tensor
    obs = torch.zeros(N_OBS_PLANES, 8, 8, dtype=torch.float32, device=device)
    
    # Fill piece planes
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        
        # Get rank and file (0-7)
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        
        # Piece type plane (0-5)
        piece_type = piece.piece_type
        plane_offset = 0 if piece.color == chess.WHITE else 6
        plane = plane_offset + PIECE_TO_PLANE[piece_type]
        
        # Set the bit
        obs[plane, rank, file] = 1.0
    
    # Side-to-move plane (index 12)
    side_plane_idx = N_PIECE_PLANES
    if board.turn == chess.WHITE:
        obs[side_plane_idx, :, :] = 1.0
    # else: stays 0
    
    # Auxiliary planes (indices 13-14)
    # Repetition count (simple: 0 or 1)
    # In python-chess, we can check if threefold repetition is possible
    rep_plane_idx = N_PIECE_PLANES + N_SIDE_PLANES
    if board.can_claim_threefold_repetition():
        obs[rep_plane_idx, :, :] = 1.0
    else:
        obs[rep_plane_idx, :, :] = 0.0
    
    # Halfmove clock (for 50-move rule)
    halfmove_plane_idx = rep_plane_idx + 1
    obs[halfmove_plane_idx, :, :] = float(board.halfmove_clock) / 100.0  # scale to [0,1]
    
    return obs


def board_to_numpy(board: chess.Board) -> np.ndarray:
    """
    Convert a chess board to a numpy array representation.
    
    Returns:
        np.ndarray of shape (15, 8, 8) with dtype float32
    """
    tensor = board_to_tensor(board)
    return tensor.cpu().numpy()


def batch_board_to_tensor(boards: list, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Convert a batch of boards to a stacked tensor.
    
    Args:
        boards: list of chess.Board instances
        device: torch device
    
    Returns:
        torch.Tensor of shape (batch_size, 15, 8, 8) with dtype float32
    """
    if device is None:
        device = torch.device('cpu')
    
    batch_size = len(boards)
    batch = torch.zeros(batch_size, N_OBS_PLANES, 8, 8, dtype=torch.float32, device=device)
    
    for i, board in enumerate(boards):
        batch[i] = board_to_tensor(board, device=device)
    
    return batch


def get_obs_shape() -> tuple:
    """Return the observation tensor shape."""
    return OBS_SHAPE


def get_n_obs_planes() -> int:
    """Return number of observation planes."""
    return N_OBS_PLANES
