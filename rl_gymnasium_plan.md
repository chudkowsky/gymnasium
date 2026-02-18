# RL with Gymnasium + Pretrained Transformer — Agent Execution Plan (CPU-first)

This is a step-by-step, **followable plan for an AI agent** (or you) to implement RL training around your existing pretrained chess transformer checkpoint (`.pth`).  
Primary milestone: **run full games end-to-end (model vs opponent) with legal-move masking**.  
Only after that: PPO or AlphaZero-style self-play loops.

---

## 0. Definitions (freeze these first)

### 0.1 Repository structure (recommended)
```
rl/
  env/
    chess_env.py
    encoding.py
    action_space.py
  models/
    policy.py
    checkpoint.py
  opponents/
    random_player.py
    stockfish_player.py
    snapshot_player.py
  rollouts/
    runner.py
    buffers.py
  train/
    ppo.py
    az_selfplay.py
    losses.py
  eval/
    eval_match.py
    metrics.py
  configs/
    env.json
    train_ppo.json
    train_az.json
  scripts/
    smoke_env.py
    smoke_model_vs_random.py
    smoke_model_vs_stockfish.py
    run_ppo.py
    run_selfplay.py
data/
  pgn/
  checkpoints/
  manifests/
```

### 0.2 Non-negotiable contracts
Freeze these. Version them. Never change silently.

1) **Observation encoding**: `board -> obs_tensor`
- Must match what your transformer was trained on.

2) **Action space**: `action_id <-> chess.Move`
- Must be a fixed-size discrete space `N`.
- Must support promotions.

3) **Legal action mask**: `mask(board) -> {0,1}^N`
- The agent must never sample illegal moves (mask them).

4) **Model I/O**
- `forward(obs) -> logits[N]` (and optionally `value`)

---

## 1. Step 0 Milestone (the only goal of this stage)

✅ **Goal:** Load your `.pth`, create `ChessEnv`, and finish at least 1 full game:  
- model vs random  
- no crashes  
- no illegal moves chosen  
- correct termination

Do NOT start PPO or self-play training before this is stable.

---

## 2. Environment setup (dependencies)

### 2.1 Minimal Python deps
- python 3.10+ (3.11 ok)
- gymnasium
- python-chess
- torch
- numpy

Optional later:
- stable-baselines3 (PPO)
- wandb or tensorboard
- stockfish binary (UCI)

### 2.2 Stockfish binary
- Install Stockfish and make its path configurable (env var or config).
- Do not integrate Stockfish until after the step-0 milestone.

---

## 3. Action space design (most important implementation)

### 3.1 Choose action mapping (recommended)
Use:
- `from` square (0..63)
- `to` square (0..63)
- promotions (4 options: q,r,b,n) for pawn moves to last rank

Recommended mapping:
- base moves: `from*64 + to` => 4096
- promotions: separate contiguous block: `PROMO_OFFSET + promo_index`

You must implement:
- `move_to_action(move, board) -> int`
- `action_to_move(action, board) -> chess.Move`
- `legal_action_mask(board) -> np.ndarray[N] (0/1)`

### 3.2 Promotion handling
Promotions are the most common source of bugs.

Implementation rule:
- If `move.promotion` is set, it must map to the promo block.
- `action_to_move` must reconstruct the promotion piece.
- `legal_action_mask` must set promo actions only when legal.

### 3.3 En-passant / castling
Your mapping must support:
- en-passant (normal from-to is enough; legality via python-chess)
- castling (normal from-to is enough)

### 3.4 Validation tests for action space
Create `scripts/smoke_action_space.py`:

Tests:
1) For random boards (generate by random legal play for 20 plies):
   - enumerate legal moves via python-chess
   - convert each to action id
   - ensure in range `[0, N)`
2) For each such action:
   - convert back to move
   - ensure move is legal
3) mask correctness:
   - for each legal move `m`, mask[action(m)] == 1
   - mask has no ones outside legal action set (optional strict check)

---

## 4. Observation encoding (must match your pretrained model)

### 4.1 Implement `board_to_tensor(board) -> torch.Tensor`
This depends on how your transformer was trained. The agent must:
- locate existing encoding code in your repo (best)
- re-use it directly (preferred)
- or replicate it exactly (risky)

### 4.2 Required invariants
- tensor dtype and shape exactly as expected by model
- correct orientation (white perspective vs side-to-move perspective)
- include side-to-move info somehow (plane, scalar, token)

### 4.3 Smoke test for encoding
`scripts/smoke_encoding.py`:
- encode initial board
- encode after 1 move e2e4
- verify shapes, dtypes, no NaNs
- optionally verify piece-plane counts match expected

---

## 5. Gymnasium environment (ChessEnv)

### 5.1 `ChessEnv.reset()`
Responsibilities:
- set `self.board = chess.Board()` OR from sampled opening/FEN (later)
- compute `obs = board_to_tensor(board)`
- compute `mask = legal_action_mask(board)`
- return `(obs, info)` where:
  - info contains `fen`, `legal_mask`

### 5.2 `ChessEnv.step(action)`
Responsibilities:
1) Convert action to move: `m = action_to_move(action, board)`
2) Assert legality:
   - In production RL: agent uses mask so illegals should not occur.
   - For safety: if illegal, either:
     - replace with random legal move (debug mode), OR
     - terminate with large negative reward (training mode)
3) `board.push(m)`
4) Determine termination:
   - `board.is_game_over(claim_draw=True)`
5) Reward:
   - default: 0 during game
   - terminal: +1 white win / -1 white loss / 0 draw
   - If training from side-to-move, adapt reward accordingly.
6) Return `(obs, reward, terminated, truncated, info)`
   - info should include: `fen`, `move_uci`, `move_san`, `legal_mask`, `result`

### 5.3 Truncation
Add `max_plies` (e.g. 300) and truncate if exceeded.

---

## 6. Model integration (pretrained `.pth`)

### 6.1 Loader
`models/checkpoint.py` should:
- load weights
- load config (action space size, obs shape, model dims)
- validate compatibility:
  - logits output size == `N`

### 6.2 Policy wrapper
`models/policy.py` exposes:
- `predict_logits(obs_batch) -> logits_batch`
- `predict_action(obs, mask, deterministic=False) -> action`

### 6.3 Masked sampling (required)
Implement:
- `masked_logits = logits + log(mask)` where mask is 0/1
  - equivalently set illegal logits to `-inf`
- sampling:
  - training: categorical sample
  - eval: argmax

Edge case:
- ensure mask always has at least 1 legal action.

---

## 7. Step-0 smoke runs (must pass before RL)

### 7.1 Run model vs random
`scripts/smoke_model_vs_random.py`:
- env reset
- alternate turns:
  - model chooses masked action
  - random chooses random legal action
- stop at game end or truncation
- log moves and result

Success criteria:
- 100 games, 0 illegal moves, 0 crashes

### 7.2 Run model vs itself (self-play inference)
`scripts/smoke_selfplay.py`:
- model plays both sides using masked sampling (or argmax)
- success criteria: stable + completes games

### 7.3 Optional: run vs Stockfish (evaluation only)
`scripts/smoke_model_vs_stockfish.py`:
- Stockfish uses fixed depth or nodes
- keep it slow only for eval (not training)

---

## 8. Opponents module (for curriculum)

### 8.1 RandomOpponent
- chooses uniformly from legal moves

### 8.2 SnapshotOpponent
- loads a previous checkpoint
- plays masked actions

### 8.3 StockfishOpponent
- UCI process per worker
- fixed nodes/movetime
- optional Elo limit for curriculum

---

## 9. RL Track A: PPO (Gym-friendly, simpler)

### 9.1 Preconditions
- you have a value head or can add one
- env + mask stable
- rollout runner works

### 9.2 PPO data flow
1) Run vectorized envs for `T` steps collecting:
   - obs
   - mask
   - action
   - logprob
   - value
   - reward
   - done
2) Compute advantages (GAE)
3) Optimize policy+value loss with PPO clipping

### 9.3 Must-have PPO features for chess
- action masking in policy distribution
- entropy bonus (small)
- KL monitoring
- gradient clipping
- checkpointing snapshots for opponent pool

### 9.4 Catastrophic forgetting prevention
Pick one:
- KL penalty to pretrained policy (recommended)
- freeze lower layers for first N updates
- mix in supervised batches periodically

### 9.5 PPO milestone ladder
- beats random reliably
- beats older snapshot reliably
- improves vs Stockfish Elo ladder

---

## 10. RL Track B: AlphaZero-style self-play (stronger)

### 10.1 Preconditions
- policy head + value head
- can run search (even small MCTS)
- self-play runner stable

### 10.2 Self-play loop
For each position:
- run MCTS guided by current net
- sample move from visit counts
- store:
  - obs
  - improved_policy_target (visit count distribution over actions)
  - side-to-move
At end of game:
- assign outcome z ∈ {+1,0,-1} to each position

Train net on:
- policy loss (cross-entropy to MCTS policy)
- value loss (MSE or CE to outcome)
- optional regularization

### 10.3 Snapshot + evaluation
- maintain best model
- promote only if wins a match vs best by margin

---

## 11. Logging + reproducibility (mandatory)

### 11.1 Determinism controls
- set seeds in python, numpy, torch
- log git commit hash
- log env/model config hashes

### 11.2 Artifact outputs
- checkpoints: `data/checkpoints/step_XXXX.pth`
- eval results: `data/manifests/eval_step_XXXX.json`
- optional games: `data/pgn/eval_step_XXXX.pgn`

---

## 12. Performance roadmap (CPU-first, then scale)

### 12.1 CPU-first tips
- Use batched inference: predict logits for all envs at once
- Use `Threads=1` per Stockfish process if used
- Prefer `VectorEnv` to reduce python overhead

### 12.2 Scaling later (GPU training)
- rollout workers on CPU
- central learner on GPU
- trajectories shipped via queue (redis/zmq/files)

---

## 13. Agent checklist (execution order)

### Stage A — Contracts
- [ ] Determine obs encoding and reuse existing code
- [ ] Define action space size `N` and mapping including promotions
- [ ] Write env config `configs/env.json`

### Stage B — Core implementation
- [ ] Implement `action_space.py` (move<->action, mask)
- [ ] Implement `encoding.py` (board_to_tensor)
- [ ] Implement `chess_env.py` (reset/step)

### Stage C — Smoke tests
- [ ] `smoke_action_space.py` passes
- [ ] `smoke_encoding.py` passes
- [ ] Load `.pth` successfully
- [ ] `smoke_model_vs_random.py` runs 100 games, 0 illegal moves
- [ ] `smoke_selfplay.py` runs 100 games, stable

### Stage D — Opponents + eval
- [ ] Implement RandomOpponent, SnapshotOpponent
- [ ] Implement evaluation harness (Elo ladder optional)

### Stage E — RL
- [ ] PPO loop OR AlphaZero self-play loop
- [ ] checkpointing + opponent pool
- [ ] periodic evaluation

---

## 14. Acceptance criteria for “Step 0 done”
Step 0 is complete when:
- Model loads and outputs logits sized `N`
- Environment runs and terminates correctly
- Legal mask prevents illegal moves
- 100+ full games run without error
- PGN output for games is correct (optional at step 0, required later)

---

## 15. Notes on PGN (optional in step 0, required later)
When you later store games:
- collect SAN moves each ply
- store tags (Date, Result, etc.)
- store optional per-move comments (eval, depth) if available

---

## Appendix: Common failure points (debug order)
1) Action mapping broken (promotions)
2) Mask broken (illegal sampling)
3) Encoding mismatch (model expects different planes)
4) Side-to-move mismatch (reward sign wrong)
5) Model logits not aligned to action ids
6) Inference not batched (too slow)
