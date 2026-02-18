#!/usr/bin/env python3
"""
Quick test of the full PPO training pipeline.

This runs a minimal training loop to verify all components work together.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rl.models import CheckpointLoader
from rl.opponents import RandomOpponent
from rl.train import PPOTrainer

def test_training_pipeline():
    """Test the full training pipeline."""
    device = 'cpu'
    
    print("=" * 80)
    print("Testing PPO Training Pipeline")
    print("=" * 80)
    
    # 1. Load model
    print("\n[1/3] Loading pretrained model...")
    try:
        policy = CheckpointLoader.load_chessformer('data/checkpoints/pretrained.pth', device=device)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # 2. Create opponent pool
    print("\n[2/3] Setting up opponent pool...")
    opponent_pool = [RandomOpponent()]
    print(f"✓ Created opponent pool with {len(opponent_pool)} opponent(s)")
    
    # 3. Create trainer
    print("\n[3/3] Initializing PPO trainer...")
    config = {
        'learning_rate': 3e-4,
        'n_epochs': 1,
        'batch_size': 64,
        'clip_ratio': 0.2,
        'entropy_coef': 0.01,
        'value_coef': 0.5,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'max_grad_norm': 0.5,
    }
    
    try:
        trainer = PPOTrainer(
            policy=policy,
            opponent_pool=opponent_pool,
            config=config,
            device=device,
        )
        print("✓ Trainer initialized")
    except Exception as e:
        print(f"✗ Failed to initialize trainer: {e}")
        return False
    
    # 4. Quick training test (just 1 update with 2 episodes)
    print("\n" + "=" * 80)
    print("Running Quick Training Test (2 episodes, 1 update)")
    print("=" * 80)
    
    try:
        metrics = trainer.train_step(num_episodes=2)
        print(f"\n✓ Training step completed successfully!")
        print(f"  • Steps collected: {metrics.get('step', 0)}")
        print(f"  • Episodes: {metrics.get('n_episodes', 0)}")
        print(f"  • Mean return: {metrics.get('mean_return', 0):.2f}")
        print(f"  • Mean length: {metrics.get('mean_length', 0):.0f}")
        print(f"  • Loss: {metrics.get('total_loss', 0):.4f}")
        print(f"  • KL divergence: {metrics.get('approx_kl', 0):.4f}")
    except Exception as e:
        print(f"\n✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("✓ All tests passed! Training pipeline is ready.")
    print("=" * 80)
    return True

if __name__ == '__main__':
    success = test_training_pipeline()
    sys.exit(0 if success else 1)
