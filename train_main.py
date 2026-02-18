#!/usr/bin/env python3
"""
Main PPO training script for chess RL.

Usage:
    python train_main.py --config rl/configs/train_ppo.json [--device cpu|cuda]
"""

import argparse
import json
import torch
import numpy as np
import sys
from pathlib import Path

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent))

from rl.models import CheckpointLoader
from rl.opponents import RandomOpponent, SnapshotOpponent
from rl.train import PPOTrainer


def main():
    parser = argparse.ArgumentParser(description="PPO training for chess RL")
    parser.add_argument(
        '--config',
        type=str,
        default='rl/configs/train_ppo.json',
        help='Path to training config'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Torch device'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/checkpoints',
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--stockfish-depth',
        type=int,
        default=None,
        help='Stockfish search depth (overrides config). E.g., --stockfish-depth 15'
    )
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Set seeds
    seed = config['training'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print("=" * 80)
    print(f"PPO Training for Chess RL")
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print("=" * 80)
    
    # Load pretrained model
    print("\n[1/5] Loading pretrained model...")
    checkpoint_path = config['checkpoints']['pretrained_path']
    policy = CheckpointLoader.load_chessformer(checkpoint_path, device=device)
    model_state_before = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in policy.adapter.transformer.state_dict().items()}
    print(f"✓ Loaded: {checkpoint_path}")
    
    # Create opponent pool
    print("[2/5] Setting up opponent pool...")
    opponent_pool = []
    
    opponent_types = config['opponent_pool'].get('types', ['random'])
    
    if 'random' in opponent_types:
        opponent_pool.append(RandomOpponent())
        print("  • RandomOpponent (random legal moves)")
    
    if 'stockfish' in opponent_types or config['opponent_pool'].get('stockfish', {}).get('enabled', False):
        try:
            from rl.opponents import StockfishOpponent
            sf_config = config['opponent_pool'].get('stockfish', {})
            
            # Use command-line arg if provided, otherwise use config
            sf_depth = args.stockfish_depth if args.stockfish_depth is not None else sf_config.get('depth', 10)
            sf_movetime = sf_config.get('movetime_ms', None)
            
            stockfish_opp = StockfishOpponent(depth=sf_depth, movetime_ms=sf_movetime)
            opponent_pool.append(stockfish_opp)
            print(f"  • StockfishOpponent (depth={sf_depth})")
        except Exception as e:
            print(f"  ✗ Failed to initialize Stockfish: {e}")
    
    if 'snapshot' in opponent_types:
        # Add snapshot opponents if any checkpoints exist
        snapshot_dir = Path(config['checkpoints'].get('trained_output_dir', 'data/checkpoints'))
        snapshots = list(snapshot_dir.glob('trained_ppo_step_*.pth'))
        for snap_path in sorted(snapshots)[-3:]:  # Last 3 snapshots
            try:
                opp = SnapshotOpponent(str(snap_path), device=device)
                opponent_pool.append(opp)
                print(f"  • {opp.get_name()}")
            except Exception as e:
                print(f"  ✗ Failed to load {snap_path}: {e}")
    
    if not opponent_pool:
        opponent_pool.append(RandomOpponent())
        print("  • RandomOpponent (default)")
    
    # Create trainer
    print("[3/5] Initializing PPO trainer...")
    trainer = PPOTrainer(
        policy=policy,
        opponent_pool=opponent_pool,
        config=config['ppo'],
        device=device,
    )
    print(f"✓ Trainer ready (lr={config['ppo']['learning_rate']}, clip={config['ppo']['clip_ratio']})")
    
    # Training loop
    print("[4/5] Starting training loop...")
    print(f"  • Total steps: {config['training']['total_steps']}")
    print(f"  • Steps per update: {config['training']['steps_per_update']}")
    print(f"  • Checkpoint interval: {config['training']['checkpoint_interval']}")
    print()
    
    total_steps = config['training']['total_steps']
    steps_per_update = config['training']['steps_per_update']
    log_interval = config['training']['log_interval']
    checkpoint_interval = config['training']['checkpoint_interval']
    
    num_episodes = config['rollout']['num_episodes_per_update']
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        while trainer.num_steps < total_steps:
            # Train step
            metrics = trainer.train_step(num_episodes=num_episodes)
            
            # Logging
            if trainer.num_updates % log_interval == 0:
                print(f"[Update {trainer.num_updates:04d}] "
                      f"Steps: {metrics.get('step', 0):6d} | "
                      f"Eps: {metrics.get('n_episodes', 0):3d} | "
                      f"MeanRet: {metrics.get('mean_return', 0):6.2f} | "
                      f"MeanLen: {metrics.get('mean_length', 0):5.0f} | "
                      f"Loss: {metrics.get('total_loss', 0):7.4f} | "
                      f"KL: {metrics.get('approx_kl', 0):7.4f}")
            
            # Checkpoint
            if trainer.num_updates % checkpoint_interval == 0:
                ckpt_path = output_dir / f"trained_ppo_step_{trainer.num_steps}.pth"
                trainer.save_checkpoint(str(ckpt_path))
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    
    # Final checkpoint
    print("\n[5/5] Saving final checkpoint...")
    final_ckpt = output_dir / f"trained_ppo_final.pth"
    trainer.save_checkpoint(str(final_ckpt))
    
    # Check if model was updated
    model_state_after = policy.adapter.transformer.state_dict()
    weights_changed = not all(
        torch.allclose(model_state_before[k], model_state_after[k])
        for k in model_state_before.keys()
        if 'running_mean' not in k and 'running_var' not in k and 'num_batches_tracked' not in k
    )
    
    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    print(f"Total steps: {trainer.num_steps}")
    print(f"Total updates: {trainer.num_updates}")
    print(f"Final checkpoint: {final_ckpt}")
    print(f"Model weights updated: {'Yes ✓' if weights_changed else 'No (frozen transformer)'}")
    print("=" * 80)


if __name__ == '__main__':
    main()
