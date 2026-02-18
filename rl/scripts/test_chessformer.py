#!/usr/bin/env python3
"""
Test ChessTransformer adapter integration.

Verify that:
1. Real model loads
2. Adapter converts output correctly
3. Inference produces legal moves (0 illegal moves)
"""

import sys
import os
sys.path.insert(0, '/home/mateusz/dev/chessformer')

import torch
import numpy as np
from pathlib import Path

os.environ['PYTHONPATH'] = '/home/mateusz/dev/gymnasium'

from rl.env import ChessEnv
from rl.models import CheckpointLoader
from rl.env.action_space import sample_legal_action


def test_chessformer_loads():
    """Test that ChessTransformer loads correctly."""
    print("Test 1: Load ChessTransformer...")
    try:
        policy = CheckpointLoader.load_chessformer(device='cpu')
        print("  ✓ Model loaded successfully")
        return policy
    except Exception as e:
        print(f"  ❌ Failed to load: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_forward_pass(policy):
    """Test forward pass through adapter."""
    print("\nTest 2: Forward pass...")
    try:
        from rl.env import ChessEnv
        import chess
        
        env = ChessEnv()
        obs, info = env.reset()
        
        # Convert to tensor
        obs_tensor = torch.from_numpy(obs).float()
        mask_tensor = torch.from_numpy(info['legal_mask']).float()
        
        # Predict action
        action = policy.predict_action(obs_tensor, mask_tensor, deterministic=True)
        
        print(f"  ✓ Forward pass successful")
        print(f"    Input shape: {obs_tensor.shape}")
        print(f"    Action: {action}")
        print(f"    Mask sum: {mask_tensor.sum()}")
        
    except Exception as e:
        print(f"  ❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()


def test_legal_moves(policy, num_games: int = 20):
    """Test that adapter never produces illegal moves."""
    print(f"\nTest 3: Legal moves test ({num_games} games)...")
    
    from rl.env import ChessEnv
    from rl.env.action_space import sample_legal_action
    
    env = ChessEnv(max_plies=300, device=torch.device('cpu'))
    
    illegal_count = 0
    plies_total = 0
    
    for game_num in range(num_games):
        obs, info = env.reset()
        plies = 0
        
        for _ in range(500):
            obs_tensor = torch.from_numpy(obs).float()
            mask_tensor = torch.from_numpy(info['legal_mask']).float()
            
            # ChessTransformer for white
            if env.get_board().turn:
                try:
                    action = policy.predict_action(obs_tensor, mask_tensor, deterministic=False)
                except Exception as e:
                    print(f"    ❌ Game {game_num+1}: Policy error: {e}")
                    illegal_count += 1
                    break
            else:
                # Random for black
                action = sample_legal_action(env.get_board(), deterministic=False)
            
            # Execute
            try:
                obs, reward, terminated, truncated, info = env.step(action)
            except ValueError as e:
                print(f"    ❌ Game {game_num+1}: Illegal move: {e}")
                illegal_count += 1
                break
            
            plies += 1
            
            if terminated or truncated:
                break
        
        plies_total += plies
        if (game_num + 1) % 5 == 0:
            print(f"  ✓ Game {game_num+1}: {plies} plies")
    
    avg_plies = plies_total / num_games if num_games > 0 else 0
    print(f"\n  Total illegal moves: {illegal_count}")
    print(f"  Average plies: {avg_plies:.1f}")
    print(f"  Status: {'✓ PASS' if illegal_count == 0 else '❌ FAIL'}")
    
    return illegal_count == 0


if __name__ == '__main__':
    print("=" * 60)
    print("TEST: ChessTransformer Model Integration")
    print("=" * 60)
    
    # Test 1: Load model
    policy = test_chessformer_loads()
    if policy is None:
        sys.exit(1)
    
    # Test 2: Forward pass
    test_forward_pass(policy)
    
    # Test 3: Legal moves
    success = test_legal_moves(policy, num_games=20)
    
    print("\n" + "=" * 60)
    print("Overall Status: " + ("✓ PASS" if success else "❌ FAIL"))
    print("=" * 60)
    
    sys.exit(0 if success else 1)
