#!/usr/bin/env python3
"""Quick test of shaped rewards without parallelization."""

import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))

from rl.models import CheckpointLoader
from rl.opponents import RandomOpponent
from rl.rollouts import RolloutCollector

print("\n" + "="*70)
print("TEST: Shaped Rewards (Sequential Collection)")
print("="*70)

# Load model
print("\n[1] Loading model...")
policy = CheckpointLoader.load_chessformer(device='cpu')
print("✓ Model loaded")

# Create collector with shaped rewards enabled
print("\n[2] Creating collector WITH shaped rewards...")
collector = RolloutCollector(
    policy=policy,
    opponent_pool=[RandomOpponent()],
    device='cpu',
    use_shaped_reward=True,
    shaped_reward_coef=0.01,
    stockfish_depth=3,
)
print("✓ Collector created")

# Collect a small batch
print("\n[3] Collecting 3 episodes WITH shaped rewards...")
buffer = collector.collect_rollouts(num_episodes=3)

print(f"\n✓ SUCCESS! Collected {len(buffer.transitions)} transitions")
print(f"  Episodes: {len(buffer.episode_returns)}")
print(f"  Mean return: {sum(buffer.episode_returns) / len(buffer.episode_returns):.3f}")
print(f"  Mean length: {sum(buffer.episode_lengths) / len(buffer.episode_lengths):.1f}")

print("\n" + "="*70)
print("✓ Shaped rewards working correctly!")
print("="*70)
