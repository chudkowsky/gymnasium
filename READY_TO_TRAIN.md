# Ready to Train! 🚀

## Current Status: Stage D + E Complete ✅

Stage 0: Environment + Model Inference ✅  
Stage D: Opponent Pool ✅  
Stage E: PPO Training Pipeline ✅  

## Quick Start

### 1. Test Everything Works (2 minutes)
```bash
python test_ppo_pipeline.py
```
Expected output: All components initialized, ~170 steps collected, 2 episodes, loss=0.04

### 2. Start Real PPO Training
```bash
python train_main.py --config rl/configs/train_ppo.json [--device cpu|cuda]
```

The training script will:
- ✓ Load pretrained ChessTransformer model
- ✓ Create opponent pool (starts with RandomOpponent)
- ✓ Initialize PPO trainer
- ✓ Collect rollouts from games
- ✓ Update policy via PPO loss
- ✓ Save checkpoints periodically
- ✓ Log training metrics

### 3. Monitor Training
Logs appear in terminal with:
```
[Update 0100] Steps:  2048 | Eps:  32 | MeanRet:  0.50 | MeanLen: 170 | Loss: 0.0391 | KL: 0.0000
[Update 0200] Steps:  4096 | Eps:  32 | MeanRet:  0.45 | MeanLen: 169 | Loss: 0.0385 | KL: 0.0001
```

Checkpoints saved to: `data/checkpoints/trained_ppo_step_XXXX.pth`

## Architecture Overview

```
ChessEnv (board state, legal moves)
    ↓
RolloutCollector (samples episodes vs opponents)
    ↓
EPISODE: [obs, action, reward, next_obs, done]
    ↓
RolloutBuffer (stores trajectories, computes GAE)
    ↓
PPOTrainer (updates policy via clipped PPO loss)
    ↓
ChessformerPolicyWrapper
    ├─ Frozen ChessTransformer (policy logits, 4100 actions)
    └─ Trainable Value Head (estimates state values)
```

## What Gets Updated During Training?

- ✅ Value head: 2-layer network predicting state values
- ❌ ChessTransformer: frozen (contains 50M parameters)
- ❌ Encoding layers: frozen

For full model training, you'd need to unfreeze the transformer in `rl/models/chessformer_adapter.py`.

## Output: Upgraded Model

After training:
- **Input**: `data/checkpoints/pretrained.pth` (107 MB)
- **Output**: `data/checkpoints/trained_ppo_step_XXXXX.pth` (includes updated value head)

The value head learns:
- What positions are winning
- What positions are losing  
- Risk/reward assessment for different game states

This helps the agent:
- Play stronger via better state understanding
- Improve via self-play when combined with the transformer

## Next Steps After Training

### Option 1: Evaluate Performance
```python
from rl.models import CheckpointLoader
from rl.opponents import SnapshotOpponent

trained_policy = CheckpointLoader.load_checkpoint(
    'data/checkpoints/trained_ppo_step_100000.pth'
)
vs_random = SnapshotOpponent(
    'data/checkpoints/trained_ppo_step_100000.pth'
)
# Play evaluation matches...
```

### Option 2: Continue Training
```bash
# Trained model becomes new baseline
# Can add it to opponent pool automatically
python train_main.py --config rl/configs/train_ppo.json
```

### Option 3: Unfreeze Transformer (Advanced)
Edit `rl/models/chessformer_adapter.py` and remove the gradient freeze:
```python
# for param in self.transformer.parameters():
#     param.requires_grad = False  # <-- Comment this out
```
Then optimizer will update all 50M parameters (needs GPU for efficiency).

## Configuration

Edit `rl/configs/train_ppo.json` to adjust:
- **learning_rate**: 3e-4 (smaller for stability)
- **batch_size**: 64 (larger = more stable but slower)
- **n_epochs**: 3 (3-5 typically good)
- **clip_ratio**: 0.2 (standard PPO value)
- **entropy_coef**: 0.01 (encourage exploration)
- **gamma**: 0.99 (long-term credit)
- **gae_lambda**: 0.95 (advantage smoothing)

## Troubleshooting

**Q: Training runs but value head never improves?**
A: Value prediction is hard early on. Monitor `mean_return` instead - if that improves, training is working.

**Q: Games end too fast / too slow?**
A: Adjust `max_episode_length` in `rl/env/chess_env.py` (currently 300 plies).

**Q: Out of memory?**
A: Reduce `num_episodes_per_update` in config from 32 → 16 or 8.

**Q: Want CPU-only for weeks without GPU?**
A: Yes, that's fine. Just slower. Full pretraining on CPU possible but would take ~weeks.

## Files Reference

**Training Files**
- `train_main.py` - Main entry point
- `test_ppo_pipeline.py` - Validation script
- `rl/train/ppo.py` - PPO algorithm (policy + value updates)
- `rl/train/losses.py` - PPO loss functions

**Environment & Inference**
- `rl/env/chess_env.py` - Gym environment
- `rl/env/encoding.py` - Board → 15-plane tensor
- `rl/env/action_space.py` - Move ↔ Action mapping
- `rl/models/chessformer_adapter.py` - Frozen transformer + trainable value head

**Opponents**
- `rl/opponents/base.py` - Interface
- `rl/opponents/random_player.py` - Random moves
- `rl/opponents/snapshot_player.py` - Checkpoint-based

**Rollouts**
- `rl/rollouts/runner.py` - Trajectory collection
- `rl/rollouts/buffers.py` - Episode storage + GAE

---

**You're ready!** Run `python test_ppo_pipeline.py` first, then `python train_main.py` 🎯
