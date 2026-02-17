"""Base protocol for checkpoint managers.

CheckpointManagers handle saving and loading training state:
- FullStateCheckpoint: Complete state (policy, optimizer, RNG, replay)
- MinimalCheckpoint: For backends that own their state
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


def extract_logZ(learner: Any) -> float:
    logZ = learner.logZ if hasattr(learner, "logZ") else 0.0
    if hasattr(logZ, "item"):
        logZ = logZ.item()
    return float(logZ)


def find_latest_checkpoint(
    save_dir: Path,
    pattern: str = "checkpoint_*.pt",
) -> Path | None:
    save_dir = Path(save_dir)
    if not save_dir.exists():
        return None
    checkpoints = sorted(
        save_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
    )
    return checkpoints[-1] if checkpoints else None


def should_save(step: int, every_steps: int) -> bool:
    if every_steps <= 0:
        return False
    return step % every_steps == 0


@dataclass
class CheckpointState:
    """Complete training state for checkpoint/resume.

    All fields are serializable via torch.save.
    """

    step_count: int
    logZ: float
    model_state_dict: dict[str, Any] | None
    optimizer_state_dict: dict[str, Any] | None
    rng_states: dict[str, Any]
    replay_buffer: dict[str, Any] | None
    config: dict[str, Any]
    metrics_history: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_count": self.step_count,
            "logZ": self.logZ,
            "model_state_dict": self.model_state_dict,
            "optimizer_state_dict": self.optimizer_state_dict,
            "rng_states": self.rng_states,
            "replay_buffer": self.replay_buffer,
            "config": self.config,
            "metrics_history": self.metrics_history,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointState:
        return cls(
            step_count=data["step_count"],
            logZ=data["logZ"],
            model_state_dict=data.get("model_state_dict"),
            optimizer_state_dict=data.get("optimizer_state_dict"),
            rng_states=data.get("rng_states", {}),
            replay_buffer=data.get("replay_buffer"),
            config=data.get("config", {}),
            metrics_history=data.get("metrics_history", []),
        )


@runtime_checkable
class CheckpointManager(Protocol):
    """Protocol for checkpoint management.

    Checkpoint managers handle:
    - Periodic saving of training state
    - Resuming from checkpoints
    - Cleanup of old checkpoints

    Example:
        >>> manager = FullStateCheckpoint(save_dir="checkpoints", every_steps=100)
        >>> manager.maybe_save(step=100, state=state)
        >>> state = manager.load("checkpoints/checkpoint_000100.pt")
    """

    def maybe_save(
        self,
        step: int,
        learner: Any,
        policy: Any | None = None,
        replay_buffer: Any | None = None,
        metrics_history: list[dict[str, float]] | None = None,
    ) -> Path | None:
        """Save checkpoint if conditions are met (e.g., step interval).

        Args:
            step: Current training step
            learner: Learner with logZ and optimizer
            policy: Policy with model state (optional)
            replay_buffer: Replay buffer (optional)
            metrics_history: Training metrics history (optional)

        Returns:
            Path to saved checkpoint, or None if not saved.
        """
        ...

    def save(
        self,
        step: int,
        learner: Any,
        policy: Any | None = None,
        replay_buffer: Any | None = None,
        metrics_history: list[dict[str, float]] | None = None,
    ) -> Path:
        """Force save a checkpoint.

        Returns:
            Path to saved checkpoint.
        """
        ...

    def load(self, path: str | Path) -> CheckpointState:
        """Load checkpoint from path.

        Returns:
            CheckpointState with restored training state.
        """
        ...

    def find_latest(self) -> Path | None:
        """Find the most recent checkpoint in save_dir.

        Returns:
            Path to latest checkpoint, or None if none exist.
        """
        ...


def save_checkpoint(path: Path, state: CheckpointState) -> None:
    """Save checkpoint to disk, creating parent directories if needed."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_dict = state.to_dict()
    torch.save(checkpoint_dict, path)

    logger.info(f"Saved checkpoint to {path} (step {state.step_count})")


def load_checkpoint(path: Path) -> CheckpointState:
    """Load checkpoint from disk.

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
    """
    import torch

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # weights_only=False: CheckpointState includes numpy/Python RNG states
    checkpoint_dict = torch.load(path, weights_only=False)
    state = CheckpointState.from_dict(checkpoint_dict)

    logger.info(f"Loaded checkpoint from {path} (step {state.step_count})")

    return state


def cleanup_old_checkpoints(
    checkpoint_dir: Path,
    keep_last_n: int,
    pattern: str = "checkpoint_*.pt",
) -> list[Path]:
    """Remove old checkpoints, keeping only the most recent *keep_last_n*."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []

    checkpoints = sorted(checkpoint_dir.glob(pattern), key=lambda p: p.stat().st_mtime)

    if keep_last_n <= 0 or len(checkpoints) <= keep_last_n:
        return []

    to_remove = checkpoints[:-keep_last_n]
    removed = []

    for ckpt in to_remove:
        try:
            ckpt.unlink()
            removed.append(ckpt)
            logger.debug(f"Removed old checkpoint: {ckpt}")
        except OSError as e:
            logger.warning(f"Failed to remove checkpoint {ckpt}: {e}")

    if removed:
        logger.info(f"Cleaned up {len(removed)} old checkpoints")

    return removed
