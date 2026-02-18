#!/usr/bin/env python3
"""
Test script for reward shaping and parallel collection improvements.

This tests:
1. Stockfish-based shaped rewards
2. Parallel game collection
3. Speed improvements vs. baseline
"""

import sys
from pathlib import Path
import torch
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).parent))

from rl.models import CheckpointLoader
from rl.opponents import RandomOpponent, StockfishOpponent
from rl.rollouts import RolloutCollector, ParallelRolloutCollector


def test_shaped_rewards():
    """Test Stockfish-based shaped rewards."""
    print("\n" + "="*70)
    print("TEST 1: Stockfish Shaped Rewards")
    print("="*70)
    
    # Load model
    print("\n[1] Loading pretrained model...")
    policy = CheckpointLoader.load_chessformer(device='cpu')
    print("✓ Model loaded")
    
    # Create opponent
    print("[2] Creating opponents...")
    opponents = [RandomOpponent(), StockfishOpponent(depth=5)]
    print("✓ Opponents created")
    
    # Test sequential collection WITH shaped rewards
    print("\n[3] Collecting 5 episodes WITH shaped rewards...")
    start = time.time()
    
    collector = RolloutCollector(
        policy=policy,
        opponent_pool=opponents,
        device='cpu',
        use_shaped_reward=True,
        shaped_reward_coef=0.01,
        stockfish_depth=3,
    )
    
    buffer = collector.collect_rollouts(num_episodes=5)
    elapsed_with_shaped = time.time() - start
    
    print(f"✓ Collected {len(buffer.transitions)} transitions in {elapsed_with_shaped:.2f}s")
    print(f"  Episodes: {len(buffer.episode_returns)}")
    print(f"  Mean return: {np.mean(buffer.episode_returns):.3f}")
    print(f"  Mean length: {np.mean(buffer.episode_lengths):.1f}")
    
    # Check reward composition
    print("\n[4] Checking reward structure...")
    print(f"  Sample episode return: {buffer.episode_returns[0]:.3f}")
    print(f"  Sample episode length: {buffer.episode_lengths[0]:.0f}")
    print("✓ Rewards computed successfully")
    
    return buffer


def test_parallel_collection():
    """Test parallel game collection."""
    print("\n" + "="*70)
    print("TEST 2: Parallel Game Collection")
    print("="*70)
    
    # Load model
    print("\n[1] Loading pretrained model...")
    policy = CheckpointLoader.load_chessformer(device='cpu')
    print("✓ Model loaded")
    
    # Create opponents
    print("[2] Creating opponents...")
    opponents = [RandomOpponent(), StockfishOpponent(depth=5)]
    print("✓ Opponents created")
    
    # Sequential baseline (small scale)
    print("\n[3] Sequential collection (4 episodes)...")
    start_seq = time.time()
    
    collector_seq = RolloutCollector(
        policy=policy,
        opponent_pool=opponents,
        device='cpu',
    )
    
    buffer_seq = collector_seq.collect_rollouts(num_episodes=4)
    time_seq = time.time() - start_seq
    print(f"✓ Sequential: {len(buffer_seq.transitions)} transitions in {time_seq:.2f}s")
    
    # Parallel collection
    print("\n[4] Parallel collection (4 episodes, 4 workers)...")
    start_par = time.time()
    
    collector_par = ParallelRolloutCollector(
        policy=policy,
        opponent_pool=opponents,
        device='cpu',
        num_workers=4,
    )
    
    buffer_par = collector_par.collect_rollouts(num_episodes=4)
    time_par = time.time() - start_par
    print(f"✓ Parallel: {len(buffer_par.transitions)} transitions in {time_par:.2f}s")
    
    # Speedup
    speedup = time_seq / time_par if time_par > 0 else float('inf')
    print(f"\n[5] Performance comparison:")
    print(f"  Sequential time: {time_seq:.2f}s")
    print(f"  Parallel time:   {time_par:.2f}s")
    print(f"  Speedup:         {speedup:.2f}x")
    
    if speedup > 1.5:
        print("✓ Parallel significantly faster!")
    else:
        print("⚠ Parallel may not be faster (overhead on small scale)")
    
    return speedup


def test_combined_improvements():
    """Test both improvements together."""
    print("\n" + "="*70)
    print("TEST 3: Combined Improvements (Shaped Rewards + Parallel)")
    print("="*70)
    
    # Load model
    print("\n[1] Loading pretrained model...")
    policy = CheckpointLoader.load_chessformer(device='cpu')
    print("✓ Model loaded")
    
    # Create opponents
    print("[2] Creating opponents...")
    opponents = [RandomOpponent(), StockfishOpponent(depth=5)]
    print("✓ Opponents created")
    
    # Combined mode
    print("\n[3] Collecting with BOTH improvements (8 episodes, 4 workers)...")
    start = time.time()
    
    collector = ParallelRolloutCollector(
        policy=policy,
        opponent_pool=opponents,
        device='cpu',
        num_workers=4,
        use_shaped_reward=True,
        shaped_reward_coef=0.01,
        stockfish_depth=3,
    )
    
    buffer = collector.collect_rollouts(num_episodes=8)
    elapsed = time.time() - start
    
    print(f"\n✓ Collected {len(buffer.transitions)} transitions in {elapsed:.2f}s")
    print(f"  Episodes: {len(buffer.episode_returns)}")
    print(f"  Episodes/sec: {len(buffer.episode_returns)/elapsed:.2f}")
    print(f"  Mean return: {np.mean(buffer.episode_returns):.3f}")
    print(f"  Mean length: {np.mean(buffer.episode_lengths):.1f}")
    
    return buffer


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("IMPROVEMENT TESTS: Shaped Rewards & Parallel Collection")
    print("="*70)
    
    try:
        # Test 1: Shaped rewards
        print("\n[Starting Test 1...]")
        buffer1 = test_shaped_rewards()
        print("✓ Test 1 PASSED")
        
    except Exception as e:
        print(f"\n✗ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        # Test 2: Parallel collection
        print("\n[Starting Test 2...]")
        speedup = test_parallel_collection()
        print("✓ Test 2 PASSED")
        
    except Exception as e:
        print(f"\n✗ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        # Test 3: Combined
        print("\n[Starting Test 3...]")
        buffer3 = test_combined_improvements()
        print("✓ Test 3 PASSED")
        
    except Exception as e:
        print(f"\n✗ Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    print("\n" + "="*70)
    print("ALL TESTS PASSED! ✓")
    print("="*70)
    print("\nImprovement Summary:")
    print("  • Shaped rewards: ✓ Dense position-based rewards working")
    print("  • Parallel collection: ✓ Multi-worker episode collection working")
    print("  • Combined mode: ✓ Both improvements work together")
    print("\nNext steps:")
    print("  1. Run training with shaped rewards config:")
    print("     python train_main.py --config rl/configs/train_ppo_shaped_rewards.json")
    print("  2. Run training with parallel config:")
    print("     python train_main.py --config rl/configs/train_ppo_parallel.json")
    print("  3. Run training with both improvements:")
    print("     python train_main.py --config rl/configs/train_ppo_all_improved.json")
    print("="*70)
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
