"""Minimal checkpointing for backends that own their state.

Used with SkyRL/Ray or Tinker where the backend manages most state.
Only saves step count and minimal metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from synthstats.train.checkpointing.base import CheckpointState

logger = logging.getLogger(__name__)


class MinimalCheckpoint:
    """Minimal checkpoint manager.

    For backends that own their state (SkyRL/Ray, Tinker).
    Only saves step count and minimal metadata.

    Args:
        save_dir: Directory for checkpoints
        every_steps: Save interval (0 = never)
        resume_from: Path to resume from (optional)
    """

    def __init__(
        self,
        save_dir: str | Path = "checkpoints",
        every_steps: int = 0,
        resume_from: str | Path | None = None,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.every_steps = every_steps
        self.resume_from = Path(resume_from) if resume_from else None

    def maybe_save(
        self,
        step: int,
        learner: Any,
        policy: Any | None = None,
        replay_buffer: Any | None = None,
        metrics_history: list[dict[str, float]] | None = None,
    ) -> Path | None:
        """Save if interval met (usually disabled for minimal)."""
        if self.every_steps <= 0:
            return None
        if step % self.every_steps != 0:
            return None
        return self.save(step, learner, policy, replay_buffer, metrics_history)

    def save(
        self,
        step: int,
        learner: Any,
        policy: Any | None = None,
        replay_buffer: Any | None = None,
        metrics_history: list[dict[str, float]] | None = None,
    ) -> Path:
        """Save minimal checkpoint."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / f"checkpoint_{step:06d}.pt"

        logZ = learner.logZ if hasattr(learner, "logZ") else 0.0
        if hasattr(logZ, "item"):
            logZ = logZ.item()

        state = {
            "step_count": step,
            "logZ": logZ,
            "metrics_history": metrics_history or [],
        }

        torch.save(state, path)
        logger.info(f"Saved minimal checkpoint to {path}")
        return path

    def load(self, path: str | Path) -> CheckpointState:
        """Load checkpoint."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        data = torch.load(path, weights_only=False)

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

    def find_latest(self) -> Path | None:
        """Find most recent checkpoint."""
        if not self.save_dir.exists():
            return None

        checkpoints = sorted(
            self.save_dir.glob("checkpoint_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        return checkpoints[-1] if checkpoints else None
