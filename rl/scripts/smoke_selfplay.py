#!/usr/bin/env python3
"""
Smoke test: Model self-play.

Runs games where the model plays both sides to ensure stability.
"""

import torch
import numpy as np
from rl.env import ChessEnv
from rl.models import SimpleTransformerPolicy, PolicyWrapper


def run_selfplay_game(env, policy, max_steps: int = 500, deterministic: bool = False):
    """
    Run one self-play game (model vs itself).
    
    Returns:
        (result, n_plies, illegal_count)
    """
    obs, info = env.reset()
    illegal_count = 0
    result = 'max_steps'  # Default if loop exits normally
    
    ply = 0
    while ply < max_steps:
        obs_tensor = torch.from_numpy(obs).float()
        mask_tensor = torch.from_numpy(info['legal_mask']).float()
        
        # Both sides use same policy
        try:
            action = policy.predict_action(
                obs_tensor, mask_tensor,
                deterministic=deterministic
            )
        except Exception as e:
            print(f"❌ Policy error at ply {ply}: {e}")
            return 'error', ply, illegal_count
        
        try:
            obs, reward, terminated, truncated, info = env.step(action)
        except ValueError as e:
            illegal_count += 1
            break
        
        ply += 1
        
        if terminated:
            result = info.get('result', 'draw')
            break
        
        if truncated:
            result = 'truncated'
            break
    
    return result, ply, illegal_count


def main(num_games: int = 100, device: str = 'cpu', deterministic: bool = False):
    """Run self-play games."""
    print("=" * 60)
    print(f"SMOKE TEST: Model Self-Play ({num_games} games)")
    print(f"Deterministic: {deterministic}")
    print("=" * 60)
    
    env = ChessEnv(max_plies=300, device=torch.device(device))
    model = SimpleTransformerPolicy(n_actions=4100, hidden_dim=128, device=device)
    policy = PolicyWrapper(model, device=device)
    
    results = {'white_win': 0, 'black_win': 0, 'draw': 0, 'truncated': 0, 'max_steps': 0, 'error': 0}
    plies_per_game = []
    total_illegal = 0
    
    print(f"\nRunning {num_games} games...\n")
    
    for game_num in range(num_games):
        result, ply, illegal_count = run_selfplay_game(
            env, policy, deterministic=deterministic
        )
        
        results[result] += 1
        plies_per_game.append(ply)
        total_illegal += illegal_count
        
        if illegal_count > 0:
            print(f"❌ Game {game_num+1}: {illegal_count} illegal moves, {result}, {ply} plies")
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
    deterministic = '--deterministic' in sys.argv
    
    success = main(num_games=100, device=device, deterministic=deterministic)
    sys.exit(0 if success else 1)
