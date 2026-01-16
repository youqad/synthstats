# SynthStats

GFlowNet training for PyMC program synthesis. The model learns to sample programs proportional to their reward (ELPD-LOO score) rather than collapsing to a single mode.

## Setup

```bash
git clone https://github.com/youqad/synthstats.git
cd synthstats
uv sync --extra ml --extra pymc
```

## Training

```bash
# basic run (dugongs is the default)
uv run python scripts/train_skyrl.py model=qwen3_0.6b

# different environment
uv run python scripts/train_skyrl.py task.env=peregrines model=qwen3_0.6b

# with overrides
uv run python scripts/train_skyrl.py \
    task.env=dugongs \
    model=qwen3_4b \
    +trainer.batch_size=4 \
    +trainer.episodes=100
```

Available environments: `dugongs`, `peregrines`, `eight_schools`, `surgical`

Available models: `qwen3_0.6b` (fast), `qwen3_4b` (production), `mock` (testing)

## How it works

1. Policy generates PyMC program
2. Executor runs it in sandbox, gets MCMC samples
3. Judge scores with ELPD-LOO
4. SubTB loss updates policy toward reward-proportional sampling

The key difference from standard RL: we're not maximizing reward, we're matching a distribution. High-reward programs get sampled more often, but low-reward ones aren't eliminated entirely.

## Project structure

```
src/synthstats/
  core/           # protocols: Task, Policy, Executor, Judge
  envs/           # BoxingGym wrappers
  policies/       # HFPolicy (transformers + LoRA)
  executors/      # PyMC sandbox
  training/       # TrainingLoop, SubTB loss, replay buffer
  trainers/       # SkyRLSubTBTrainer

scripts/
  train_skyrl.py  # main entry point

configs/          # Hydra configs for envs, models, training
```

## Trajectory balance loss

The core idea (simplified):

```
L = (logZ + sum(log_pi) - log_R)^2
```

- `logZ`: learned partition function (separate high LR, ~0.1)
- `log_pi`: sum of token log-probs for generated program
- `log_R`: log reward from judge

The actual implementation uses SubTB (sub-trajectory balance) which computes this over partial trajectories with geometric weighting. See `training/losses/trajectory_balance.py` for details.

The loss pushes `logZ + log_pi` toward `log_R`. When balanced, sampling frequency matches reward.

## Replay buffer

Uses on-sample re-scoring: stores action sequences only, recomputes log-probs with current policy when sampled. Eliminates off-policy bias that would break the TB objective.

## Tests

```bash
uv run pytest tests/ -x -q
```

## Status

Research code. The training loop works, loss converges, but we haven't yet validated that learned policies actually match the target distribution. That's the next milestone.
