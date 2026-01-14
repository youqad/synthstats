<p align="center">
  <img src="docs/assets/images/logo.png" alt="SynthStats Logo" width="200">
</p>

<h1 align="center">SynthStats</h1>
<p align="center">GFlowNet-steered probabilistic program synthesis for safer AI</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12%20|%203.13-blue.svg" alt="Python 3.12-3.13">
  <img src="https://img.shields.io/badge/docs-mkdocs-blue.svg" alt="Documentation">
</p>

---

## Overview

SynthStats trains language models to sample from distributions over probabilistic programs (PyMC) proportionally to their likelihood. Unlike standard RL methods that mode-seek (collapsing to a single high-reward program), GFlowNet training with Sub-Trajectory Balance (SubTB) loss preserves diversity across the entire reward distribution.

This enables uncertainty quantification through sampling, exploration of alternative models, and interpretable Bayesian reasoning. Each generated program is a human-readable statistical model, not a black-box neural network.

The architecture is built on a plugin system (Task/Policy/Executor/Judge protocols) that separates environment logic from training infrastructure, making it easy to adapt SynthStats to new domains beyond probabilistic programming.

## Key Features

- **SubTB Loss with Learned logZ** — Flow matching across all sub-trajectory lengths for faster convergence
- **Plugin Architecture** — Swap tasks, policies, executors, and judges without touching training code
- **BoxingGym Benchmarks** — Dugongs, Peregrines, Eight Schools, and Surgical environments
- **Safe Execution** — AST filtering blocks dangerous operations, subprocess isolation contains failures
- **SkyRL Integration** — Scale from local runs to multi-node SLURM clusters
- **GFN Replay Buffer** — On-sample re-scoring eliminates off-policy bias
- **ELPD-LOO Rewards** — Automatic Bayesian model scoring via leave-one-out cross-validation

## Installation

Requires Python 3.12 or 3.13:

```bash
# Clone repository
git clone https://github.com/youdar/synthstats.git
cd synthstats

# Install dependencies (uv is recommended)
uv sync

# Install with ML dependencies
uv sync --extra ml --extra pymc
```

## Quick Start

Train on the Dugongs environment:

```bash
uv run python scripts/train.py env=dugongs model=qwen3_0.6b
```

Override config via Hydra:

```bash
uv run python scripts/train.py env=peregrines model=qwen3_4b \
  +trainer.batch_size=8 +wandb.project=my_experiment
```

Run tests:

```bash
uv run pytest
```

See the [full documentation](docs/) for architecture details, training guides, and API references.

## Project Structure

```
src/synthstats/
├── core/           # Protocol definitions (Task, Policy, Executor, Judge)
├── training/       # GFlowNet losses (SubTB, TB), trainers, replay buffers
├── tasks/          # Task plugins (Boxing, SynthStats, ARC, SWE)
├── judges/         # Reward components (ELPD-LOO, formatting, LLM critic)
├── policies/       # Policy implementations (HuggingFace, mock)
├── runtime/        # Execution sandboxes (Python AST, Docker)
└── integrations/   # SkyRL, Tinker API
```

## Links

- **Documentation**: [/docs](docs/) (MkDocs site)
- **Issues**: [GitHub Issues](https://github.com/youdar/synthstats/issues)
