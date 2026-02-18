#!/usr/bin/env python3
"""Quick test of rollout collection."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from rl.models import CheckpointLoader
from rl.opponents import RandomOpponent
from rl.rollouts import RolloutCollector

print("Loading model...")
policy = CheckpointLoader.load_chessformer(device='cpu')

print("Creating opponent pool...")
opponent_pool = [RandomOpponent()]

print("Creating collector...")
collector = RolloutCollector(policy, opponent_pool, device='cpu')

print("Collecting 1 episode...")
ret, length, buffer = collector.collect_episode()

print(f"\n✓ Episode complete!")
print(f"  Return: {ret:.2f}")
print(f"  Length: {length}")
print(f"  Transitions: {len(buffer.transitions)}")
print(f"  Buffer stats: {buffer.stats()}")
