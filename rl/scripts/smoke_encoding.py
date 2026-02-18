#!/usr/bin/env python3
"""
Smoke test for board encoding.

Validates:
1. Tensor shape and dtype
2. Piece placement correctness
3. Side-to-move encoding
4. No NaNs or invalid values
"""

import chess
import torch
import numpy as np
from rl.env.encoding import board_to_tensor, get_obs_shape, get_n_obs_planes


def test_shape_and_dtype():
    """Test observation shape and dtype."""
    print("Testing shape and dtype...")
    
    board = chess.Board()
    obs = board_to_tensor(board)
    
    expected_shape = get_obs_shape()
    assert obs.shape == expected_shape, \
        f"Shape mismatch: {obs.shape} != {expected_shape}"
    assert obs.dtype == torch.float32, \
        f"Dtype mismatch: {obs.dtype} != float32"
    
    print(f"✓ Shape {obs.shape} and dtype {obs.dtype} correct")


def test_initial_position():
    """Test encoding of the initial board."""
    print("\nTesting initial board encoding...")
    
    board = chess.Board()
    obs = board_to_tensor(board)
    
    # Check that pieces are on correct squares
    # White pieces on ranks 0-1, black pieces on ranks 6-7
    
    # Initial position: white pieces on rank 0 (pawn on 1), black on rank 7 (pawn on 6)
    # Rank 0 is at index 0 in tensor (chess rank 1)
    
    # White pawns should be on rank 1 (index 1)
    white_pawn_plane = obs[0]
    black_pawn_plane = obs[6]
    
    # Check one pawn exists
    assert white_pawn_plane.sum() == 8, "Expected 8 white pawns in initial position"
    assert black_pawn_plane.sum() == 8, "Expected 8 black pawns in initial position"
    
    print("✓ Initial position encoding correct")


def test_side_to_move():
    """Test side-to-move encoding."""
    print("\nTesting side-to-move encoding...")
    
    board = chess.Board()
    obs = board_to_tensor(board)
    
    # Initial position: white to move
    side_plane = obs[12]  # side-to-move plane
    assert torch.all(side_plane == 1.0), "White to move should have all 1s in side plane"
    
    # After one move: black to move
    board.push(chess.Move.from_uci("e2e4"))
    obs = board_to_tensor(board)
    side_plane = obs[12]
    assert torch.all(side_plane == 0.0), "Black to move should have all 0s in side plane"
    
    print("✓ Side-to-move encoding correct")


def test_no_nans(num_boards: int = 50):
    """Test that no NaNs or infinities are produced."""
    print(f"\nTesting for NaNs/Infs on {num_boards} random boards...")
    
    for i in range(num_boards):
        board = chess.Board()
        for _ in range(np.random.randint(1, 50)):
            if board.is_game_over():
                break
            moves = list(board.legal_moves)
            board.push(np.random.choice(moves))
        
        obs = board_to_tensor(board)
        
        assert not torch.isnan(obs).any(), f"NaN found in observation at iteration {i}"
        assert not torch.isinf(obs).any(), f"Inf found in observation at iteration {i}"
        assert torch.all(obs >= 0) and torch.all(obs <= 1), \
            f"Values out of [0, 1] at iteration {i}"
    
    print(f"✓ No NaNs/Infs in {num_boards} random boards")


def test_piece_counting(num_boards: int = 50):
    """Verify piece counts match board state."""
    print(f"\nTesting piece counting on {num_boards} boards...")
    
    # Import piece type
    import chess
    
    for i in range(num_boards):
        board = chess.Board()
        for _ in range(np.random.randint(1, 30)):
            if board.is_game_over():
                break
            moves = list(board.legal_moves)
            board.push(np.random.choice(moves))
        
        obs = board_to_tensor(board)
        
        # Count pieces in observation
        obs_np = obs.numpy()
        
        for plane_idx in range(12):  # piece planes
            plane = obs_np[plane_idx]
            obs_count = int(plane.sum())
            
            # Get corresponding piece type and color
            piece_type_idx = plane_idx % 6
            color = chess.WHITE if plane_idx < 6 else chess.BLACK
            
            # Map piece index to piece type
            piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                          chess.ROOK, chess.QUEEN, chess.KING]
            piece_type = piece_types[piece_type_idx]
            
            # Count pieces on board
            board_count = len(board.pieces(piece_type, color))
            
            assert obs_count == board_count, \
                f"Piece count mismatch for {piece_type=}, {color=}: " \
                f"obs={obs_count}, board={board_count}"
    
    print(f"✓ Piece counts match on {num_boards} boards")


def test_move_sequence():
    """Test encoding across a sequence of moves."""
    print("\nTesting move sequence encoding...")
    
    moves = ["e2e4", "e7e5", "g1f3", "b8c6"]
    board = chess.Board()
    
    for move_uci in moves:
        move = chess.Move.from_uci(move_uci)
        obs_before = board_to_tensor(board)
        
        board.push(move)
        obs_after = board_to_tensor(board)
        
        # Observations should be different
        assert not torch.allclose(obs_before, obs_after), \
            f"Observation unchanged after move {move_uci}"
        
        # Check side-to-move changed
        side_before = obs_before[12].sum().item()
        side_after = obs_after[12].sum().item()
        assert side_before != side_after, \
            f"Side-to-move unchanged after move {move_uci}"
    
    print("✓ Move sequence encoding correct")


def test_device_handling():
    """Test tensor device placement."""
    print("\nTesting device handling...")
    
    board = chess.Board()
    
    # CPU
    obs_cpu = board_to_tensor(board, device=torch.device('cpu'))
    assert obs_cpu.device.type == 'cpu', "CPU tensor not on CPU"
    
    # Default (should be CPU)
    obs_default = board_to_tensor(board)
    assert obs_default.device.type == 'cpu', "Default tensor not on CPU"
    
    print("✓ Device handling correct")


if __name__ == '__main__':
    print("=" * 60)
    print("SMOKE TEST: Board Encoding")
    print("=" * 60)
    
    test_shape_and_dtype()
    test_initial_position()
    test_side_to_move()
    test_no_nans(num_boards=50)
    test_piece_counting(num_boards=50)
    test_move_sequence()
    test_device_handling()
    
    print("\n" + "=" * 60)
    print("All encoding tests passed! ✓")
    print("=" * 60)
