<p align="center">
  <img src="docs/assets/images/logo.png" alt="SynthStats Logo" width="200">
</p>

<h1 align="center">SynthStats</h1>

<p align="center">
  <a href="https://github.com/youqad/synthstats/actions/workflows/ci.yml"><img src="https://github.com/youqad/synthstats/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.12%20|%203.13-blue.svg" alt="Python 3.12-3.13">
  <img src="https://img.shields.io/badge/docs-mkdocs-blue.svg" alt="Documentation">
</p>

---

## Overview

SynthStats is research code exploring GFlowNet objectives for training language-model policies to generate probabilistic programs (PyMC). It scores proposed programs with task-specific rewards (often likelihood-based) and aims to maintain a diverse set of high-reward candidates rather than optimizing a single best output. The repository includes training loops (TB/SubTB), replay buffers, and benchmark tasks adapted from BoxingGym.

## Status

**Experimental research code.** Interfaces, defaults, and APIs may change. Not intended for production.

## What's Included

- **Training:** GFlowNet objectives (Sub-Trajectory Balance, Trajectory Balance) and prioritized replay buffers
- **Execution:** Protocol-based environment for PyMC programs with AST-based sandboxing
- **Benchmarks:** Tasks adapted from BoxingGym (Dugongs, Peregrines, Eight Schools, Surgical)
- **Architecture:** Plugin system separating Task, Policy, Executor, and Judge

## Sandboxing Note

Generated code is executed with best-effort constraints (AST checks, subprocess isolation). This is **not** a hardened security sandbox. Do not run on untrusted inputs.

## Experimental: SkyRL Compatibility

This repo includes experimental SkyRL-compatible trainer configs and loss registration, but does **not** use SkyRL's `BasePPOExp` training stack. Multi-node/SLURM execution depends on your cluster configuration.

## Installation

Requires Python 3.12 or 3.13:

```bash
git clone https://github.com/youqad/synthstats.git
cd synthstats

# Install dependencies
uv sync

# Install dev tools (ruff, mypy, pytest-cov)
uv sync --group dev

# With ML + PyMC extras
uv sync --extra ml --extra pymc
```

## Quick Start

Local training on Dugongs:

```bash
uv run synthstats-train runner=local env=dugongs policy=hf_qwen3_0_6b
```

Common overrides via Hydra:

```bash
uv run synthstats-train runner=local env=peregrines policy=hf_qwen3_4b \
  runner.train.batch_size=8 logging=wandb
```

SFT warm-start (pre-populate replay with demonstrations):

```bash
uv run synthstats-train runner=local env=dugongs policy=hf_qwen3_0_6b \
  sft_warmstart.enabled=true \
  sft_warmstart.data_path=/path/to/sft.jsonl \
  sft_warmstart.compute_rewards=true
```

Notes:
- `sft_warmstart.strip_thinking` defaults to `false` (preserves `<think>` as a latent variable for TB/SubTB).
- `sft_warmstart.compute_rewards=true` can be slow (executes programs to score them).
- Optional knobs: `sft_warmstart.max_examples`, `sft_warmstart.log_clamp=[-700,700]`, `sft_warmstart.show_progress=false`.

AB-SubTB (action-boundary SubTB) can be enabled via objective overrides:

```bash
uv run synthstats-train runner=local env=dugongs policy=hf_qwen3_0_6b \
  objective.loss_type=ab_subtb \
  objective.ab_subtb_alpha=0.1 \
  objective.use_boundary_critic=true
```

AB-SubTB summary:
- Keeps vanilla TB as anchor and adds an action-boundary SubTB regularizer.
- Optional boundary critic can be enabled with:
  - `objective.use_boundary_critic=true`
  - `objective.boundary_critic_hidden_dim=32`
  - `objective.boundary_critic_loss_coef=1.0`
- Main knobs in `configs/objective/subtb.yaml`:
  - `loss_type`: `tb | modified_subtb | ab_subtb`
  - `subtb_lambda`
  - `ab_subtb_alpha`
  - boundary critic knobs above

Backend support:
- Local objective path: supported.
- `SkyRLSubTBTrainer` wrapper path: supported.
- Tinker TB/endpoint-SubTB path: unchanged by default.

Checkpointing:
- Objective state includes boundary critic parameters when enabled.
- `SkyRLSubTBTrainer` state now restores full objective state with backward-compatible
  fallback for older `logZ`-only checkpoints.

Commit-visible details are in `AB_SUBTB.md`.
Note: `docs/` is currently ignored in this repository, so `README.md` and
`AB_SUBTB.md` are the tracked sources of truth.

Run tests:

```bash
uv run pytest
```

Developer checks:

```bash
uv run ruff check .
uv run mypy src/synthstats
uv run pytest --cov=src/synthstats
```

## Project Structure

```
src/synthstats/
├── cli/            # Unified CLI entrypoints (synthstats-train)
├── core/           # Protocol definitions (Task, Policy, Executor, Judge)
├── train/          # Runner → Learner → Objective training stack
├── tasks/          # Task plugins (Boxing, etc.)
├── judges/         # Reward components (ELPD-LOO, formatting)
├── policies/       # Policy implementations (HuggingFace, mock)
├── executors/      # Execution sandboxes (PyMC)
├── envs/           # BoxingGym wrappers / text environments
└── integrations/   # Tinker API adapter
```

## Links

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/youqad/synthstats/issues)
