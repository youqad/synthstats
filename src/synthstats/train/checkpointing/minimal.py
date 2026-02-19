"""Minimal checkpointing for backends that own their state (SkyRL/Ray, Tinker)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from synthstats.train.checkpointing.base import (
    CheckpointState,
    _BaseCheckpointManager,
    extract_logZ,
)

logger = logging.getLogger(__name__)


class MinimalCheckpoint(_BaseCheckpointManager):
    """Saves step count and logZ only; backend manages the rest."""

    def __init__(
        self,
        save_dir: str | Path = "checkpoints",
        every_steps: int = 0,
        resume_from: str | Path | None = None,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.every_steps = every_steps
        self.resume_from = Path(resume_from) if resume_from else None

    def save(
        self,
        step: int,
        learner: Any,
        policy: Any | None = None,
        replay_buffer: Any | None = None,
        metrics_history: list[dict[str, float]] | None = None,
    ) -> Path:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / f"checkpoint_{step:06d}.pt"

        logZ = extract_logZ(learner)

        state = {
            "step_count": step,
            "logZ": logZ,
            "metrics_history": metrics_history or [],
        }

        torch.save(state, path)
        logger.info(f"Saved minimal checkpoint to {path}")
        return path

    def load(self, path: str | Path) -> CheckpointState:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        data = torch.load(path, weights_only=True)

        return CheckpointState(
            step_count=data.get("step_count", 0),
            logZ=data.get("logZ", 0.0),
            model_state_dict=None,
            optimizer_state_dict=None,
            rng_states={},
            replay_buffer=None,
            config={},
            metrics_history=data.get("metrics_history", []),
        )
