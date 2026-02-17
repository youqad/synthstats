# SynthStats

GFlowNet training for PyMC program synthesis. The model learns to sample programs proportional to their reward (ELPD-LOO score) rather than collapsing to a single mode.

## Setup

```bash
git clone https://github.com/youqad/synthstats.git
cd synthstats
uv sync --group dev --extra ml --extra pymc
```

## Training

```bash
# basic local run (dugongs is the default)
uv run synthstats-train runner=local env=dugongs policy=hf_qwen3_0_6b

# different environment
uv run synthstats-train runner=local env=peregrines policy=hf_qwen3_0_6b

# with overrides
uv run synthstats-train runner=local env=dugongs policy=hf_qwen3_4b \
    runner.train.batch_size=4 \
    runner.train.steps=1000 \
    logging=wandb
```

SFT warm-start (seed replay with demonstrations):

```bash
uv run synthstats-train runner=local env=dugongs policy=hf_qwen3_0_6b \
    sft_warmstart.enabled=true \
    sft_warmstart.data_path=/path/to/sft.jsonl \
    sft_warmstart.compute_rewards=true
```

Notes:
- `sft_warmstart.strip_thinking` defaults to `false` (preserves `<think>` as a latent variable for TB/SubTB).
- Optional knobs: `sft_warmstart.max_examples`, `sft_warmstart.log_clamp=[-700,700]`, `sft_warmstart.show_progress=false`.

Available environments include BoxingGym tasks like `dugongs`, `peregrines`, `eight_schools`, `surgical`.
See `configs/env/` for the full list.

Available policies: `hf_qwen3_0_6b` (fast), `hf_qwen3_4b` (production), `mock` (testing), `tinker` (API backend)

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
  cli/            # unified training CLI (synthstats-train)
  envs/           # BoxingGym wrappers / text environments
  policies/       # HFPolicy (transformers + LoRA) + mocks
  executors/      # PyMC sandbox
  train/          # runner-based training stack (collect → learn → log → checkpoint)

scripts/
  train_skyrl.py  # SkyRL training script

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

The actual implementation uses SubTB (sub-trajectory balance) which computes this over partial trajectories with geometric weighting. See `train/objectives/trajectory_balance.py` for details.

The loss pushes `logZ + log_pi` toward `log_R`. When balanced, sampling frequency matches reward.

## Replay buffer

Uses on-sample re-scoring: stores action sequences only, recomputes log-probs with current policy when sampled. Avoids stale log-probs that would introduce off-policy bias into the TB loss.

## Tests

```bash
uv run pytest tests/ -x -q
```

## Status

Research code. The training loop works, loss converges, but we haven't yet validated that learned policies actually match the target distribution. That's the next milestone.
