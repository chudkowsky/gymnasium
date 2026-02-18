# Chess RL — PPO Training with Chessformer

A reinforcement learning framework for training a chess agent via PPO, built on top of a pretrained **Chessformer** transformer model and a [Gymnasium](https://gymnasium.farama.org/)-compatible chess environment.

## What it does

The agent fine-tunes a pretrained chess transformer (Chessformer) using Proximal Policy Optimization (PPO). It learns by playing games against a pool of opponents (random, Stockfish, and past versions of itself). Legal move masking is enforced at every step — the agent never samples an illegal move.

**Training flow:**
1. Collect rollouts by playing games against the opponent pool
2. Compute returns and advantages using GAE
3. Update the policy with PPO (clipped objective + value loss + entropy bonus)
4. Save checkpoints periodically; load snapshots back into the opponent pool

## Requirements

- Python 3.10+
- Stockfish installed and on `$PATH`

```bash
pip install -r requirements.txt
```

Dependencies: `torch`, `gymnasium`, `python-chess`, `numpy`, [`chessformer`](https://github.com/chudkowsky/chessformer)

## Pretrained model

Place the Chessformer checkpoint at:
```
data/checkpoints/pretrained.pth
```

The checkpoint path is configured in `rl/configs/train_ppo.json`.

## Training

```bash
export PYTHONPATH=/path/to/gymnasium

python train_main.py --config rl/configs/train_ppo.json --device cpu
# or for GPU:
python train_main.py --config rl/configs/train_ppo.json --device cuda
```

Checkpoints are saved to `data/checkpoints/` at regular intervals.

### Key config options (`rl/configs/train_ppo.json`)

| Field | Default | Description |
|---|---|---|
| `training.total_steps` | 500 000 | Total environment steps |
| `training.steps_per_update` | 4096 | Steps between PPO updates |
| `ppo.learning_rate` | 3e-5 | Adam LR |
| `ppo.clip_ratio` | 0.2 | PPO clip epsilon |
| `rollout.num_episodes_per_update` | 32 | Games per update |
| `rollout.shaped_reward.enabled` | true | Stockfish-based dense rewards |
| `opponent_pool.types` | `["random", "stockfish"]` | Opponents to use |
| `opponent_pool.stockfish.depth` | 4 | Stockfish search depth |

## Architecture

### Model
- **Chessformer** (12-layer transformer, d_model=512) — pretrained on chess games
- **ChessformerAdapter** wraps it for RL: converts board obs → action logits + value
- Logit for action `(from, to)`: `score[from] + score[to]`

### Environment
- Obs: `(15, 8, 8)` float32 — 6 white piece planes, 6 black piece planes, side-to-move, repetition, halfmove clock
- Actions: 4100 discrete — `from*64 + to` (0–4095) plus 4 promotion types (4096–4099)
- Reward: +1 white wins, 0 draw, -1 black wins (optionally augmented with Stockfish eval delta)

## Project structure

```
train_main.py              # Entry point
rl/
  env/
    chess_env.py           # Gymnasium ChessEnv
    action_space.py        # move <-> action mapping
    encoding.py            # board -> 15-plane tensor
  models/
    checkpoint.py          # CheckpointLoader, PolicyWrapper
    chessformer_adapter.py # ChessformerAdapter, ChessformerPolicyWrapper
  opponents/               # RandomOpponent, SnapshotOpponent, StockfishOpponent
  rollouts/
    runner.py              # Sequential rollout collector
    parallel_runner.py     # Parallel rollout collector (multiprocessing)
  train/
    ppo.py                 # PPOTrainer
  configs/
    train_ppo.json         # Default training config
data/checkpoints/          # Pretrained + trained model checkpoints
```

## License

MIT
