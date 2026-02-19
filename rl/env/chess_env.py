"""
Chess environment for Gymnasium.

Implements gym.Env interface for chess RL training.
- Observation: 15-plane tensor (pieces + side-to-move + auxiliary)
- Action space: 4100 discrete actions (base moves + promotions)
- Reward modes:
    terminal (default): +1 win, 0 draw, -1 loss (white perspective)
    shaped: terminal + Stockfish eval delta * coef
    accuracy: per-move accuracy in [0, 1] based on centipawn loss vs best move
"""

import math
import gymnasium as gym
from gymnasium import spaces
import chess
import chess.engine
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

        # Persistent Stockfish engine (lazy init, reused across all evaluations)
        self._engine = None
        self.eval_cache = {}

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
        self.eval_cache = {}  # Clear cache each episode
        self._prev_eval = None  # Cache last eval_after as next eval_before
        
        obs = board_to_tensor(self.board, device=self.device).cpu().numpy()
        mask = legal_action_mask(self.board)
        
        info = {
            'fen': self.board.fen(),
            'legal_mask': mask,
        }
        
        return obs.astype(np.float32), info
    
    def _get_engine(self):
        """Return the persistent Stockfish engine, starting it if needed."""
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci('stockfish')
            self._engine.configure({'Threads': 1})
        return self._engine

    def evaluate_position(self, board: chess.Board, depth: int = 3) -> float:
        """
        Evaluate board position using a persistent Stockfish process.

        Returns:
            score normalised to [-1, 1] from white's perspective.
        """
        fen = board.fen()
        cache_key = (fen, depth, 'norm')
        if cache_key in self.eval_cache:
            return self.eval_cache[cache_key]

        try:
            engine = self._get_engine()
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            score = info['score'].white()

            if score.is_mate():
                cp = 30000 if score.mate() > 0 else -30000
            else:
                cp = score.score(mate_score=30000)

            normalized = float(np.tanh(cp / 300.0))
            self.eval_cache[cache_key] = normalized
            return normalized

        except Exception:
            # Engine died — clear it so it restarts next call
            self._engine = None
            return 0.0

    def _evaluate_cp(self, board: chess.Board, depth: int = 5) -> float:
        """
        Evaluate board position, returning raw centipawns from white's perspective.
        Used for accuracy reward computation.
        """
        fen = board.fen()
        cache_key = (fen, depth, 'cp')
        if cache_key in self.eval_cache:
            return self.eval_cache[cache_key]

        try:
            engine = self._get_engine()
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            score = info['score'].white()

            if score.is_mate():
                cp = 30000.0 if score.mate() > 0 else -30000.0
            else:
                cp = float(score.score(mate_score=30000))

            self.eval_cache[cache_key] = cp
            return cp

        except Exception:
            self._engine = None
            return 0.0

    @staticmethod
    def _accuracy_from_cp(best_cp: float, played_cp: float) -> float:
        """
        Compute move accuracy in [0, 1] using the Chess.com accuracy formula.

        Converts centipawn scores to win percentages first (since win% is a
        nonlinear function of cp), then applies:
            accuracy = 103.1668 * exp(-0.04354 * win%_loss) - 3.1668

        Args:
            best_cp: Stockfish eval of the position before the move (centipawns,
                     white's perspective) — this equals the best-move value.
            played_cp: Stockfish eval after the agent's move (centipawns,
                       white's perspective).

        Examples (from an equal position, played_cp = best_cp - loss):
            0 cp loss   → ~1.00 (perfect)
            50 cp loss  → ~0.81
            200 cp loss → ~0.45
            600 cp loss → ~0.15
        """
        def win_pct(cp: float) -> float:
            return 100.0 / (1.0 + math.exp(-0.00368208 * cp))

        win_loss = max(0.0, win_pct(best_cp) - win_pct(played_cp))
        raw = 103.1668 * math.exp(-0.04354 * win_loss) - 3.1668
        return float(np.clip(raw / 100.0, 0.0, 1.0))
    
    def step(
        self,
        action: int,
        use_shaped_reward: bool = False,
        shaped_reward_coef: float = 0.01,
        stockfish_depth: int = 3,
        use_accuracy_reward: bool = False,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step of the environment.

        Reward modes (mutually exclusive; accuracy takes precedence):
            use_accuracy_reward=True:
                Per-move accuracy in [0, 1] based on centipawn loss vs Stockfish best
                move. No terminal reward. Requires two Stockfish calls per agent move.
            use_shaped_reward=True:
                terminal reward + Stockfish eval delta * shaped_reward_coef.
            default:
                terminal reward only (+1 win, 0 draw, -1 loss).

        Args:
            action: Action id in [0, N_ACTIONS)
            use_shaped_reward: enable Stockfish eval-delta reward
            shaped_reward_coef: coefficient for shaped position reward
            stockfish_depth: Stockfish search depth for evaluation
            use_accuracy_reward: enable per-move accuracy reward (overrides shaped)

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
        
        # Accuracy reward: evaluate position (= best-move eval) BEFORE the move.
        # We always call fresh — _prev_eval is not valid here because the opponent
        # has moved since the last agent step, changing the position.
        best_cp = None
        if use_accuracy_reward:
            best_cp = self._evaluate_cp(self.board, depth=stockfish_depth)

        # Shaped reward: reuse cached eval from previous agent step
        position_reward = 0.0
        if use_shaped_reward and not use_accuracy_reward:
            if self._prev_eval is None:
                self._prev_eval = self.evaluate_position(self.board, depth=stockfish_depth)
            eval_before = self._prev_eval

        # Execute move
        self.board.push(move)
        self.ply_count += 1
        self.move_history.append(move)

        # Get new observation
        obs = board_to_tensor(self.board, device=self.device).cpu().numpy()
        mask = legal_action_mask(self.board)

        # Determine termination status
        terminated = self.board.is_game_over(claim_draw=True)
        truncated = self.ply_count >= self.max_plies

        # Compute reward
        accuracy_reward = None
        terminal_reward = 0.0
        result = None

        if use_accuracy_reward:
            # Per-move accuracy in [0, 1] PLUS terminal outcome signal.
            # Without the terminal reward the agent never learns that losing is bad —
            # every move still gets a positive accuracy reward even in losing games.
            played_cp = self._evaluate_cp(self.board, depth=stockfish_depth)
            accuracy_reward = self._accuracy_from_cp(best_cp, played_cp)
            if terminated:
                outcome = self.board.outcome()
                if outcome is not None and outcome.winner == chess.WHITE:
                    terminal_reward = 1.0
                    result = 'white_win'
                elif outcome is not None and outcome.winner == chess.BLACK:
                    terminal_reward = -1.0
                    result = 'black_win'
                else:
                    result = 'draw'
            reward = accuracy_reward + terminal_reward
        else:
            # Terminal reward
            if terminated:
                outcome = self.board.outcome()
                if outcome is None:
                    result = 'draw'
                elif outcome.winner == chess.WHITE:
                    terminal_reward = 1.0
                    result = 'white_win'
                elif outcome.winner == chess.BLACK:
                    terminal_reward = -1.0
                    result = 'black_win'
                else:
                    result = 'draw'

            # Shaped reward (eval delta)
            if use_shaped_reward:
                eval_after = self.evaluate_position(self.board, depth=stockfish_depth)
                position_reward = (eval_after - eval_before) * shaped_reward_coef
                self._prev_eval = eval_after

            reward = position_reward + terminal_reward

        if terminated and result is None:
            # Determine result string for accuracy-reward mode too
            outcome = self.board.outcome()
            if outcome is None:
                result = 'draw'
            elif outcome.winner == chess.WHITE:
                result = 'white_win'
            elif outcome.winner == chess.BLACK:
                result = 'black_win'
            else:
                result = 'draw'

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
            'accuracy_reward': accuracy_reward,
        }
        
        return obs.astype(np.float32), float(reward), terminated, truncated, info
    
    def render(self, mode: str = 'ansi') -> Optional[str]:
        """Render the current board state."""
        if mode == 'ansi':
            return str(self.board)
        raise ValueError(f"Unknown render mode: {mode}")
    
    def close(self):
        """Clean up resources."""
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None
    
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
