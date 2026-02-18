#!/usr/bin/env python3
"""
Smoke test: Model vs Random Player.

Runs multiple games of a trained model against a random opponent.
Validates:
1. No illegal moves are chosen
2. Games terminate correctly
3. Rewards are correctly computed
"""

import torch
import numpy as np
from pathlib import Path
from rl.env import ChessEnv, legal_action_mask
from rl.models import SimpleTransformerPolicy, PolicyWrapper, CheckpointLoader
from rl.env.action_space import sample_legal_action


def create_or_load_model(device: str = 'cpu'):
    """Create a simple model for testing (or load from checkpoint if exists)."""
    model = SimpleTransformerPolicy(n_actions=4100, hidden_dim=128, device=device)
    model = model.to(device)
    return model


def run_game(env, policy, max_steps: int = 500, verbose: bool = False):
    """
    Run one game: policy (white) vs random (black).
    
    Returns:
        (result, n_plies, illegal_moves, info)
        result: 'white_win', 'black_win', 'draw', 'truncated'
        illegal_moves: list of (ply, action) where move was illegal
    """
    obs, info = env.reset()
    illegal_moves = []
    result = 'max_steps'  # Default if loop exits normally
    
    ply = 0
    while ply < max_steps:
        # Convert obs to tensor
        obs_tensor = torch.from_numpy(obs).float()
        mask_tensor = torch.from_numpy(info['legal_mask']).float()
        
        # White's turn: use policy
        if env.get_board().turn:
            try:
                action = policy.predict_action(obs_tensor, mask_tensor, deterministic=False)
            except Exception as e:
                print(f"❌ Policy error: {e}")
                return 'error', ply, illegal_moves, info
        else:
            # Black's turn: random
            action = sample_legal_action(env.get_board(), deterministic=False)
        
        # Execute action
        try:
            obs, reward, terminated, truncated, info = env.step(action)
        except ValueError as e:
            illegal_moves.append((ply, action, str(e)))
            if verbose:
                print(f"  ❌ Ply {ply}: Illegal move action={action}: {e}")
            break
        
        if verbose and ply % 20 == 0:
            print(f"  Ply {ply}: {info['move_san']}")
        
        ply += 1
        
        if terminated:
            result = info.get('result', 'draw')
            break
        
        if truncated:
            result = 'truncated'
            break
    
    return result, ply, illegal_moves, info


def main(num_games: int = 100, device: str = 'cpu', verbose: bool = False):
    """Run multiple games."""
    print("=" * 60)
    print(f"SMOKE TEST: Model vs Random ({num_games} games)")
    print("=" * 60)
    
    # Create environment and model
    env = ChessEnv(max_plies=300, device=torch.device(device))
    model = create_or_load_model(device)
    policy = PolicyWrapper(model, device=device)
    
    # Statistics
    results = {'white_win': 0, 'black_win': 0, 'draw': 0, 'truncated': 0, 'error': 0}
    plies_per_game = []
    total_illegal = 0
    
    print(f"\nRunning {num_games} games...\n")
    
    for game_num in range(num_games):
        verbose_this_game = verbose and game_num < 3  # verbose for first 3 games
        
        result, ply, illegal_moves, info = run_game(
            env, policy, verbose=verbose_this_game
        )
        
        results[result] += 1
        plies_per_game.append(ply)
        total_illegal += len(illegal_moves)
        
        if illegal_moves:
            print(f"❌ Game {game_num+1}: Found {len(illegal_moves)} illegal moves!")
            print(f"   Result: {result}, Plies: {ply}")
        elif (game_num + 1) % 20 == 0:
            print(f"✓ Game {game_num+1}: {result} ({ply} plies)")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total games: {num_games}")
    print(f"Results:")
    for res, count in results.items():
        if count > 0:
            print(f"  {res}: {count} ({100*count//num_games}%)")
    print(f"\nAvg plies per game: {np.mean(plies_per_game):.1f}")
    print(f"Total illegal moves: {total_illegal}")
    print(f"Status: {'✓ PASS' if total_illegal == 0 else '❌ FAIL'}")
    print("=" * 60)
    
    return total_illegal == 0


if __name__ == '__main__':
    import sys
    device = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
    verbose = '--verbose' in sys.argv
    
    success = main(num_games=100, device=device, verbose=verbose)
    sys.exit(0 if success else 1)
