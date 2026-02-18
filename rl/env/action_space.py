"""
Action space definition and utilities for chess RL.

Action mapping:
- Base moves: action = from*64 + to, where from,to in [0,63]
  => 0..4095 (4096 base moves)
- Promotions: action = PROMO_OFFSET + promo_idx, where promo_idx in [0,3]
  for (q,r,b,n), only when move is a pawn promotion to rank 8/1
  => 4096..4099 (4 promotion types)

Total action space size: N = 4100
"""

import chess
import numpy as np
from typing import Optional, Tuple


# Action space constants
N_SQUARES = 64
N_BASE_MOVES = N_SQUARES * N_SQUARES  # 4096
PROMO_OFFSET = N_BASE_MOVES  # 4096
N_PROMOS = 4  # queen, rook, bishop, knight
N_ACTIONS = N_BASE_MOVES + N_PROMOS  # 4100

# Promotion piece mapping
PROMO_PIECES = {
    chess.QUEEN: 0,
    chess.ROOK: 1,
    chess.BISHOP: 2,
    chess.KNIGHT: 3,
}
PROMO_PIECES_REVERSE = {v: k for k, v in PROMO_PIECES.items()}


def square_to_idx(square: chess.Square) -> int:
    """Convert chess.Square to index [0, 63]."""
    return int(square)


def idx_to_square(idx: int) -> chess.Square:
    """Convert index [0, 63] to chess.Square."""
    return chess.Square(idx)


def move_to_action(move: chess.Move, board: Optional[chess.Board] = None) -> int:
    """
    Convert a chess.Move to an action id.
    
    Args:
        move: chess.Move object
        board: chess.Board (unused, for API consistency)
    
    Returns:
        Action id in range [0, N_ACTIONS)
    
    Raises:
        ValueError if move is not a valid promotion or base move
    """
    from_idx = square_to_idx(move.from_square)
    to_idx = square_to_idx(move.to_square)
    
    # Check if this is a promotion
    if move.promotion is not None:
        if move.promotion not in PROMO_PIECES:
            raise ValueError(f"Invalid promotion piece: {move.promotion}")
        promo_idx = PROMO_PIECES[move.promotion]
        return PROMO_OFFSET + promo_idx
    
    # Base move
    action = from_idx * N_SQUARES + to_idx
    if not (0 <= action < N_BASE_MOVES):
        raise ValueError(f"Invalid base move action {action}")
    
    return action


def action_to_move(action: int, board: chess.Board) -> chess.Move:
    """
    Convert an action id to a chess.Move on the given board.
    
    Args:
        action: Action id in range [0, N_ACTIONS)
        board: chess.Board (required to disambiguate and validate)
    
    Returns:
        chess.Move object
    
    Raises:
        ValueError if action is out of bounds or illegal on the board
    """
    if not (0 <= action < N_ACTIONS):
        raise ValueError(f"Action {action} out of bounds [0, {N_ACTIONS})")
    
    # Promotion move
    if action >= PROMO_OFFSET:
        promo_idx = action - PROMO_OFFSET
        if not (0 <= promo_idx < N_PROMOS):
            raise ValueError(f"Invalid promotion index {promo_idx}")
        
        # Find a legal pawn promotion move on the board
        # This is a placeholder; in practice we'd need to know which pawn
        # This shouldn't be called directly; use legal_action_mask first
        promotion_piece = PROMO_PIECES_REVERSE[promo_idx]
        
        # Scan board for pawn promotions
        for move in board.legal_moves:
            if move.promotion == promotion_piece:
                return move
        
        raise ValueError(f"No legal promotion with piece {promotion_piece}")
    
    # Base move
    from_idx = action // N_SQUARES
    to_idx = action % N_SQUARES
    
    from_square = idx_to_square(from_idx)
    to_square = idx_to_square(to_idx)
    
    try:
        move = chess.Move(from_square, to_square)
    except ValueError as e:
        raise ValueError(f"Invalid move squares: {from_square} -> {to_square}") from e
    
    # Verify legality
    if move not in board.legal_moves:
        raise ValueError(f"Move {move} is not legal on the given board")
    
    return move


def legal_action_mask(board: chess.Board) -> np.ndarray:
    """
    Compute a binary mask of legal actions for the given board.
    
    Returns:
        np.ndarray of shape (N_ACTIONS,) with dtype uint8, where mask[i] = 1
        if action i is legal, 0 otherwise.
    """
    mask = np.zeros(N_ACTIONS, dtype=np.uint8)
    
    for move in board.legal_moves:
        action = move_to_action(move)
        mask[action] = 1
    
    return mask


def sample_legal_action(board: chess.Board, deterministic: bool = False) -> int:
    """
    Sample a legal action uniformly at random.
    
    Args:
        board: chess.Board
        deterministic: if True, return argmax (single legal action if only one)
    
    Returns:
        Action id in range [0, N_ACTIONS)
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise ValueError("No legal moves available")
    
    if deterministic and len(legal_moves) == 1:
        return move_to_action(legal_moves[0])
    
    move = np.random.choice(legal_moves)
    return move_to_action(move)


def validate_action_space() -> bool:
    """
    Sanity check: verify action space constants are correct.
    """
    assert N_SQUARES == 64, "N_SQUARES must be 64"
    assert N_BASE_MOVES == 4096, "N_BASE_MOVES must be 4096"
    assert N_PROMOS == 4, "N_PROMOS must be 4"
    assert N_ACTIONS == 4100, "N_ACTIONS must be 4100"
    assert PROMO_OFFSET == 4096, "PROMO_OFFSET must be 4096"
    
    return True
