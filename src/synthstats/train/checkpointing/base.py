"""Base protocol for checkpoint managers.

CheckpointManagers handle saving and loading training state:
- FullStateCheckpoint: Complete state (policy, optimizer, RNG, replay)
- MinimalCheckpoint: For backends that own their state
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


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
        """Convert to dictionary for serialization."""
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
        """Create CheckpointState from dictionary."""
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
