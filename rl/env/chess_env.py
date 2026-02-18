"""
Chess environment for Gymnasium.

Implements gym.Env interface for chess RL training.
- Observation: 15-plane tensor (pieces + side-to-move + auxiliary)
- Action space: 4100 discrete actions (base moves + promotions)
- Reward: +1 win, 0 draw, -1 loss (white perspective)
"""

import gymnasium as gym
from gymnasium import spaces
import chess
import numpy as np
import torch
from typing import Optional, Tuple, Dict, Any

from .encoding import board_to_tensor, get_obs_shape, get_n_obs_planes
from .action_space import (
    legal_action_mask, move_to_action, action_to_move,
    N_ACTIONS, sample_legal_action
)


class ChessEnv(gym.Env):
    """
    Chess environment for RL agents.
    
    Observation:
    - obs: torch.Tensor of shape (15, 8, 8)
    - info['legal_mask']: binary mask of legal actions
    - info['fen']: FEN string of current position
    
    Action: int in [0, N_ACTIONS)
    
    Reward:
    - Per-step: 0
    - Terminal: +1 (white win), 0 (draw), -1 (black win)
    
    Info dict contains:
    - legal_mask: binary mask
    - fen: board FEN
    - move_uci: last move in UCI format
    - move_san: last move in SAN format
    - result: game result ('white_win', 'draw', 'black_win', or None if ongoing)
    - truncated_by_ply: whether truncated due to max_plies
    """
    
    metadata = {'render_modes': ['ansi']}
    
    def __init__(
        self,
        max_plies: int = 300,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the chess environment.
        
        Args:
            max_plies: Maximum number of half-moves before truncation
            device: torch device for observations ('cpu', 'cuda', etc.)
            seed: Random seed
        """
        super().__init__()
        
        self.max_plies = max_plies
        self.device = device if device is not None else torch.device('cpu')
        
        # Gymnasium spaces
        obs_shape = get_obs_shape()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=obs_shape,
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)
        
        # Game state
        self.board = None
        self.ply_count = 0
        self.move_history = []
        
        # Seed for reproducibility
        if seed is not None:
            self.seed(seed)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Additional options (unused)
        
        Returns:
            (obs, info) where:
            - obs: np.ndarray of shape (15, 8, 8)
            - info: dict with 'fen', 'legal_mask'
        """
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.board = chess.Board()
        self.ply_count = 0
        self.move_history = []
        self.eval_cache = {}  # Cache for Stockfish evaluations
        
        obs = board_to_tensor(self.board, device=self.device).cpu().numpy()
        mask = legal_action_mask(self.board)
        
        info = {
            'fen': self.board.fen(),
            'legal_mask': mask,
        }
        
        return obs.astype(np.float32), info
    
    def evaluate_position(self, board: chess.Board, depth: int = 3) -> float:
        """
        Evaluate board position using Stockfish.
        
        Args:
            board: Chess position to evaluate
            depth: Stockfish search depth (default 3 for speed)
        
        Returns:
            score: position evaluation normalized to [-1, 1]
                   Positive = white advantage, Negative = black advantage
        """
        # Check cache first
        fen = board.fen()
        cache_key = (fen, depth)
        if cache_key in self.eval_cache:
            return self.eval_cache[cache_key]
        
        try:
            import subprocess
            import time
            
            # Start stockfish process
            process = subprocess.Popen(
                ['stockfish'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # UCI handshake with timeout
            process.stdin.write("uci\n")
            process.stdin.flush()
            start_time = time.time()
            while time.time() - start_time < 2:
                line = process.stdout.readline()
                if "uciok" in line:
                    break
            
            # Set position and evaluate
            process.stdin.write(f"position fen {fen}\n")
            process.stdin.write(f"go depth {depth}\n")
            process.stdin.flush()
            
            # Parse evaluation
            score_cp = 0
            best_move_found = False
            timeout_start = time.time()
            
            while time.time() - timeout_start < 5:
                try:
                    line = process.stdout.readline(timeout=0.5)
                except:
                    line = ""
                
                if not line:
                    continue
                
                if "score cp" in line:
                    parts = line.split()
                    try:
                        cp_idx = parts.index("cp") + 1
                        score_cp = int(parts[cp_idx])
                    except (ValueError, IndexError):
                        pass
                elif "score mate" in line:
                    # Mate score
                    parts = line.split()
                    try:
                        mate_idx = parts.index("mate") + 1
                        mate_in = int(parts[mate_idx])
                        score_cp = 30000 if mate_in > 0 else -30000
                    except (ValueError, IndexError):
                        pass
                elif "bestmove" in line:
                    best_move_found = True
                    break
            
            try:
                process.stdin.write("quit\n")
                process.stdin.flush()
                process.wait(timeout=1)
            except:
                process.kill()
            
            # Normalize to [-1, 1] using tanh
            # Centipawn range: ±300 = substantial advantage, ±1000 = winning
            normalized = float(np.tanh(score_cp / 300.0))
            
            # Cache result
            self.eval_cache[cache_key] = normalized
            
            return normalized
        
        except Exception as e:
            # Fallback: no evaluation available
            return 0.0
    
    def step(
        self,
        action: int,
        use_shaped_reward: bool = False,
        shaped_reward_coef: float = 0.01,
        stockfish_depth: int = 3,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step of the environment.
        
        Args:
            action: Action id in [0, N_ACTIONS)
            use_shaped_reward: whether to use position-based shaped rewards
            shaped_reward_coef: coefficient for position reward (e.g., 0.01)
            stockfish_depth: depth for Stockfish evaluation (only if use_shaped_reward=True)
        
        Returns:
            (obs, reward, terminated, truncated, info)
        
        Raises:
            ValueError: if action is illegal
        """
        if self.board is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        
        # Convert action to move
        try:
            move = action_to_move(action, self.board)
        except ValueError as e:
            raise ValueError(f"Illegal action {action}: {e}") from e
        
        # Check for legality (should not happen if agent uses mask)
        if move not in self.board.legal_moves:
            raise ValueError(f"Action {action} maps to illegal move: {move}")
        
        # Get SAN before pushing
        move_san = self.board.san(move)
        
        # Compute position reward (before move)
        position_reward = 0.0
        if use_shaped_reward:
            eval_before = self.evaluate_position(self.board, depth=stockfish_depth)
        
        # Execute move
        self.board.push(move)
        self.ply_count += 1
        self.move_history.append(move)
        
        # Get new observation
        obs = board_to_tensor(self.board, device=self.device).cpu().numpy()
        mask = legal_action_mask(self.board)
        
        # Compute position reward (after move)
        if use_shaped_reward:
            eval_after = self.evaluate_position(self.board, depth=stockfish_depth)
            # Improvement in position = reward (from white's perspective)
            position_reward = (eval_after - eval_before) * shaped_reward_coef
        
        # Determine termination status
        terminated = self.board.is_game_over(claim_draw=True)
        truncated = self.ply_count >= self.max_plies
        
        # Compute terminal reward (strong signal)
        terminal_reward = 0.0
        result = None
        if terminated:
            outcome = self.board.outcome()
            if outcome is None:
                # Draw by stalemate or other rule
                terminal_reward = 0.0
                result = 'draw'
            elif outcome.winner == chess.WHITE:
                terminal_reward = 1.0
                result = 'white_win'
            elif outcome.winner == chess.BLACK:
                terminal_reward = -1.0
                result = 'black_win'
            else:
                # Draw
                terminal_reward = 0.0
                result = 'draw'
        
        # Combined reward
        reward = position_reward + terminal_reward
        
        # Build info dict
        info = {
            'fen': self.board.fen(),
            'legal_mask': mask,
            'move_uci': move.uci(),
            'move_san': move_san,
            'result': result,
            'ply': self.ply_count,
            'truncated_by_ply': truncated and not terminated,
            'position_reward': position_reward,
            'terminal_reward': terminal_reward,
        }
        
        return obs.astype(np.float32), float(reward), terminated, truncated, info
    
    def render(self, mode: str = 'ansi') -> Optional[str]:
        """Render the current board state."""
        if mode == 'ansi':
            return str(self.board)
        raise ValueError(f"Unknown render mode: {mode}")
    
    def close(self):
        """Clean up resources."""
        pass
    
    def get_board(self) -> chess.Board:
        """Return a copy of the current board."""
        if self.board is None:
            return chess.Board()
        return self.board.copy()
    
    def get_fen(self) -> str:
        """Return the FEN of the current position."""
        if self.board is None:
            return chess.Board().fen()
        return self.board.fen()
    
    def get_ply_count(self) -> int:
        """Return the current ply count."""
        return self.ply_count
