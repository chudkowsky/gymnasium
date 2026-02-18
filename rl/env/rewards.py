"""Reward shaping utilities for chess RL."""
import chess
import subprocess
from typing import Optional


def get_stockfish_eval(board: chess.Board, stockfish_path: str = "stockfish", depth: int = 5) -> float:
    """
    Get position evaluation from Stockfish.
    
    Args:
        board: chess.Board position
        stockfish_path: path to stockfish binary
        depth: search depth (5 is fast, 10 is accurate)
        
    Returns:
        evaluation in pawns (positive = white advantage)
        Clamped to [-10, 10] for stability
    """
    try:
        proc = subprocess.Popen(
            [stockfish_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        
        # Send position
        proc.stdin.write("uci\n")
        proc.stdin.flush()
        
        # Wait for uciok
        while True:
            line = proc.stdout.readline()
            if "uciok" in line:
                break
        
        # Set position
        proc.stdin.write(f"position fen {board.fen()}\n")
        proc.stdin.write(f"go depth {depth}\n")
        proc.stdin.flush()
        
        # Parse evaluation
        eval_cp = None
        mate_score = None
        
        while True:
            line = proc.stdout.readline()
            if "score cp" in line:
                parts = line.split()
                idx = parts.index("cp")
                eval_cp = int(parts[idx + 1])
            elif "score mate" in line:
                parts = line.split()
                idx = parts.index("mate")
                mate_moves = int(parts[idx + 1])
                mate_score = 10.0 if mate_moves > 0 else -10.0
            if "bestmove" in line:
                break
        
        proc.stdin.write("quit\n")
        proc.stdin.flush()
        proc.wait(timeout=1)
        
        # Return evaluation
        if mate_score is not None:
            return mate_score
        
        if eval_cp is not None:
            # Convert centipawns to pawns
            eval_pawns = eval_cp / 100.0
            # Clamp to reasonable range
            return max(-10.0, min(10.0, eval_pawns))
        
        return 0.0
        
    except Exception as e:
        # Fallback: return 0
        return 0.0


def compute_shaped_reward(
    board_before: chess.Board,
    board_after: chess.Board,
    terminal: bool,
    terminal_reward: float,
    use_stockfish: bool = False,
    stockfish_depth: int = 3,
) -> float:
    """
    Compute shaped reward for a move.
    
    Args:
        board_before: Board state before move
        board_after: Board state after move
        terminal: Whether game ended
        terminal_reward: Terminal reward (+1/-1/0)
        use_stockfish: Whether to use Stockfish eval (slower but more accurate)
        stockfish_depth: Stockfish search depth if enabled
        
    Returns:
        reward: shaped reward value
    """
    # Terminal states: use provided reward
    if terminal:
        return terminal_reward
    
    # Non-terminal: shaped reward
    if use_stockfish:
        # Stockfish evaluation (slower, more accurate)
        eval_before = get_stockfish_eval(board_before, depth=stockfish_depth)
        eval_after = get_stockfish_eval(board_after, depth=stockfish_depth)
        
        # Reward is improvement from white's perspective
        improvement = eval_after - eval_before
        
        # Scale to reasonable range [-1, 1]
        shaped_reward = max(-1.0, min(1.0, improvement / 2.0))
        
    else:
        # Simple material-based shaping (fast)
        material_before = compute_material(board_before)
        material_after = compute_material(board_after)
        
        improvement = material_after - material_before
        
        # Scale to reasonable range
        shaped_reward = improvement / 10.0  # ~1 pawn = 0.1 reward
    
    return shaped_reward


def compute_material(board: chess.Board) -> float:
    """
    Compute material balance (white perspective).
    
    Args:
        board: chess.Board position
        
    Returns:
        material value in pawns (P=1, N/B=3, R=5, Q=9)
    """
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }
    
    material = 0.0
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            value = piece_values[piece.piece_type]
            if piece.color == chess.WHITE:
                material += value
            else:
                material -= value
    
    return material
