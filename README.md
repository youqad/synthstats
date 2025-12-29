# SynthStats

GFlowNet-steered probabilistic program synthesis for safer AI.

## Overview

SynthStats trains LLMs to sample from distributions over probabilistic programs (PyMC) proportionally to their likelihood, using GFlowNet fine-tuning with Sub-Trajectory Balance loss.

## Installation

```bash
uv sync
uv sync --extra dev  # for development
```

## Usage

```bash
# Train on BoxingGym environment
uv run python scripts/train.py env=dugongs model=qwen3_0.6b

# Run tests
uv run pytest
```

## Project Structure

- `src/synthstats/core/` - Protocol definitions (Task, Policy, Executor, Judge)
- `src/synthstats/training/` - GFlowNet training (SubTB loss, trainer)
- `src/synthstats/tasks/` - Task plugins (Boxing, SynthStats, ARC, SWE)
- `src/synthstats/judges/` - Reward components (likelihood, formatting, LLM critic)

## Team

Sam Staton (PI), Nikolay Malkin, Younesse Kaddar, Zihuiwen Ye, Jacek Karwowski, Daniella
