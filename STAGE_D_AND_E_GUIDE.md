# Next Steps: Stages D → E

After completing Step 0, you can now begin RL training. This guide outlines the next phases.

---

## Stage D: Opponent Pool & Evaluation (Prerequisite for RL)

### Objective
Build opponent implementations and evaluation harness needed for training.

### 1. RandomOpponent ([rl/opponents/random_player.py](rl/opponents/random_player.py))
```python
class RandomOpponent:
    def select_action(self, board: chess.Board) -> int:
        """Select a uniformly random legal action."""
        # Use action_space.sample_legal_action()
```

**Smoke test**: Play 10 games vs model, verify legal moves only.

### 2. SnapshotOpponent ([rl/opponents/snapshot_player.py](rl/opponents/snapshot_player.py))
```python
class SnapshotOpponent:
    def __init__(self, checkpoint_path: str):
        """Load a saved checkpoint and play deterministically."""
        # Use PolicyWrapper from models/checkpoint.py
        
    def select_action(self, board: chess.Board, mask: np.ndarray) -> int:
        """Use policy to select action."""
```

**Smoke test**: Play 10 games vs earlier version of model, verify it wins sometimes.

### 3. Evaluation Harness ([rl/eval/eval_match.py](rl/eval/eval_match.py))
```python
def eval_match(player1, player2, num_games: int = 10) -> dict:
    """Run a match and return results."""
    return {
        'player1_wins': ...,
        'player2_wins': ...,
        'draws': ...,
        'avg_ply': ...,
    }
```

**Usage**:
```python
results = eval_match(current_model, RandomOpponent(), num_games=100)
print(f"Win rate vs random: {results['player1_wins']/100:.1%}")
```

---

## Stage E: RL Training Loops

Choose **one** of these approaches:

### Option A: PPO (Simpler, Faster to Implement)

#### Components
1. **Rollout Runner** ([rl/rollouts/runner.py](rl/rollouts/runner.py))
   - Vectorized environment collection
   - Collect obs, action, logprob, reward, mask, done
   - Experience buffers for batch training

2. **PPO Training Loop** ([rl/train/ppo.py](rl/train/ppo.py))
   - Policy + Value head
   - Advantage computation (GAE)
   - Clipped PPO loss
   - Gradient updates

3. **Safeguards** (Critical for Chess)
   - KL regularization to pretrained policy
   - Entropy bonus (small, prevent collapse)
   - Gradient clipping
   - Regular evaluation snapshots

#### PPO Pseudo-code
```python
for epoch in range(num_epochs):
    # Collect trajectories
    trajectories = rollout_runner.collect(env, policy, num_steps=2048)
    
    # Compute advantages
    advantages = compute_gae(trajectories, value_fn)
    
    # PPO update
    for update in range(num_ppo_updates):
        loss = ppo_loss(
            logits=trajectories['logits'],
            actions=trajectories['actions'],
            advantages=advantages,
            old_logprobs=trajectories['logprobs'],
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
    
    # Evaluate
    if epoch % 10 == 0:
        eval_reward = evaluate(model, RandomOpponent())
        print(f"Epoch {epoch}: {eval_reward}")
        if eval_reward > best_reward:
            save_checkpoint(model, f"step_{epoch}.pth")
```

### Option B: AlphaZero Self-Play (Stronger, More Complex)

#### Components
1. **MCTS Search** ([rl/train/mcts.py](rl/train/mcts.py))
   - Expand tree guided by policy logits
   - Backup with value estimates
   - Return improved policy targets (visit counts)

2. **Self-Play Runner** ([rl/rollouts/selfplay_runner.py](rl/rollouts/selfplay_runner.py))
   - Generate data via MCTS self-play
   - Store (obs, improved_policy, outcome)

3. **AZ Training Loop** ([rl/train/az_selfplay.py](rl/train/az_selfplay.py))
   - Policy loss: Cross-entropy to MCTS policy
   - Value loss: MSE to outcome
   - Regularization (weight decay)

#### AZ Pseudo-code
```python
best_model = initial_model

for iteration in range(num_iterations):
    # Self-play
    games = self_play(current_model, mcts_sims=100, num_games=256)
    
    # Train on collected data
    for epoch in range(num_epochs):
        batch = sample_batch(games)
        
        loss = (
            cross_entropy(logits, batch['improved_policy']) +
            mse(values, batch['outcome']) +
            regularization_loss
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluate
    if evaluate(current_model, best_model) > threshold:
        best_model = current_model
```

---

## Recommended: Start with PPO

**Why PPO first?**
- ✓ Simpler to implement (fewer hyperparameters)
- ✓ Faster training on CPU
- ✓ Better for fine-tuning pretrained model
- ✓ Proven on chess (works well with masking)

**Progression**:
1. **Day 1**: Implement RandomOpponent + evaluation
2. **Day 2**: Implement rollout runner
3. **Day 3**: Implement PPO loop
4. **Day 4+**: Train and evaluate

---

## Config Files (Create These)

### [rl/configs/train_ppo.json](rl/configs/train_ppo.json)
```json
{
  "batch_size": 128,
  "rollout_steps": 2048,
  "num_epochs": 10,
  "learning_rate": 3e-4,
  "gamma": 0.99,
  "gae_lambda": 0.95,
  "ppo_clip_ratio": 0.2,
  "value_coef": 0.5,
  "entropy_coef": 0.01,
  "kl_coef": 0.1,
  "max_grad_norm": 0.5,
  "eval_interval": 10,
  "checkpoint_dir": "data/checkpoints"
}
```

### [rl/configs/train_az.json](rl/configs/train_az.json)
```json
{
  "mcts_sims": 50,
  "num_games_per_iter": 256,
  "batch_size": 128,
  "num_epochs": 10,
  "learning_rate": 1e-3,
  "weight_decay": 1e-4,
  "policy_weight": 1.0,
  "value_weight": 1.0,
  "eval_threshold": 0.55,
  "checkpoint_dir": "data/checkpoints"
}
```

---

## Key Files to Implement (In Order)

### Priority 1 (Required for any training)
```
rl/opponents/random_player.py
rl/eval/eval_match.py
```

### Priority 2 (Required for PPO)
```
rl/rollouts/runner.py
rl/train/ppo.py
```

### Priority 3 (Optional, for AlphaZero)
```
rl/train/mcts.py
rl/rollouts/selfplay_runner.py
rl/train/az_selfplay.py
```

---

## Testing Checklist

After implementing each stage:

- [ ] **After RandomOpponent**: 
  - Run 10 games vs random
  - Verify all moves legal
  - Check win rate is ~5-15% (randomness)

- [ ] **After Evaluation Harness**:
  - Run eval_match between two snapshots
  - Verify scores add up to num_games
  - Check timing reasonable

- [ ] **After PPO Rollout Runner**:
  - Collect one batch
  - Verify shapes: obs, action, logprob, reward, mask
  - Check no NaNs in trajectories

- [ ] **After PPO Training**:
  - Run 1 epoch (should not crash)
  - Check loss decreasing over batches
  - Verify checkpoint saved
  - Play evaluation games (verify legal moves)

- [ ] **After MCTS** (if AlphaZero):
  - Run search from few positions
  - Verify visit counts sum to budget
  - Check policy targets are probability distributions

---

## Training Script Template ([rl/scripts/run_ppo.py](rl/scripts/run_ppo.py))

```python
#!/usr/bin/env python3
"""Run PPO training for chess."""

import json
import torch
import argparse
from rl.env import ChessEnv
from rl.models import SimpleTransformerPolicy, PolicyWrapper, CheckpointLoader
from rl.train.ppo import PPOTrainer
from rl.opponents.random_player import RandomOpponent
from rl.eval.eval_match import eval_match

def main(config_path: str, device: str = 'cpu'):
    # Load config
    with open(config_path) as f:
        config = json.load(f)
    
    # Create environment and model
    env = ChessEnv(max_plies=300, device=torch.device(device))
    model = SimpleTransformerPolicy(n_actions=4100, hidden_dim=256, device=device)
    
    # PPO trainer
    trainer = PPOTrainer(model, env, config, device=device)
    
    # Training loop
    best_elo = -100
    for epoch in range(config['num_epochs']):
        loss = trainer.train_epoch()
        print(f"Epoch {epoch}: loss={loss:.4f}")
        
        # Evaluate every N epochs
        if epoch % config['eval_interval'] == 0:
            policy = PolicyWrapper(model, device=device)
            results = eval_match(policy, RandomOpponent(), num_games=50)
            win_rate = results['wins'] / results['total']
            print(f"  Win rate vs random: {win_rate:.1%}")
            
            # Save checkpoint
            torch.save({
                'state_dict': model.state_dict(),
                'config': config,
                'epoch': epoch,
            }, f"data/checkpoints/ppo_epoch_{epoch:04d}.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='rl/configs/train_ppo.json')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    
    main(args.config, device=args.device)
```

---

## Performance Targets

Aim for these metrics to know training is working:

- **Day 1 (Random opponent)**: ~10% win rate (baseline)
- **Week 1**: ~30% win rate vs random
- **Week 2**: ~50% win rate vs random
- **Week 3+**: Beats older snapshots, improves Elo vs Stockfish

---

## Troubleshooting

**Problem**: PPO loss is NaN after first step
- **Check**: Gradient clipping, learning rate too high, mask has no legal actions

**Problem**: Model always plays same moves
- **Check**: Entropy bonus disabled? Temperature too low?

**Problem**: Training crashes after 100 epochs
- **Check**: Memory leak? Use `torch.no_grad()` in forward passes

**Problem**: Model forgets chess after first 10 epochs
- **Check**: KL regularization needed to prevent catastrophic forgetting

---

## Resources

- **PPO**: Schulman et al. 2017 (https://arxiv.org/abs/1707.06347)
- **AlphaZero**: Silver et al. 2018 (https://arxiv.org/abs/1805.08318)
- **Masked PPO**: Huang et al. 2022 (https://arxiv.org/abs/2105.05628)

---

Good luck with training! Reach out if you hit blockers.
