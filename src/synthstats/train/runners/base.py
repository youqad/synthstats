"""Base protocol for training runners.

Runners own the execution backend:
- LocalRunner: Pure PyTorch loop
- SkyRLRayRunner: Ray + SkyRL distributed training
- TinkerRunner: Tinker API backend
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class RunResult:
    """Result of a training run.

    Attributes:
        metrics: Final metrics from training
        checkpoints: List of checkpoint paths saved during training
        interrupted: Whether training was interrupted (SIGTERM, etc.)
        error: Error message if training failed
    """

    metrics: dict[str, float] = field(default_factory=dict)
    checkpoints: list[str] = field(default_factory=list)
    interrupted: bool = False
    error: str | None = None

    @property
    def success(self) -> bool:
        """Return True if training completed without error."""
        return self.error is None


@runtime_checkable
class Runner(Protocol):
    """Protocol for training runners.

    Runners encapsulate the execution backend and orchestrate training.
    Each runner is responsible for:
    - Building components (env, policy, learner, etc.) from config
    - Running the training loop
    - Handling checkpointing and logging
    - Graceful shutdown on signals

    Example:
        >>> runner = LocalRunner(cfg)
        >>> result = runner.run()
        >>> print(f"Final loss: {result.metrics['loss']}")
    """

    def run(self) -> RunResult:
        """Execute the training run.

        Returns:
            RunResult with final metrics and checkpoint paths.
        """
        ...

    def state_dict(self) -> dict[str, Any]:
        """Serialize runner state for checkpointing.

        Returns full training state including:
        - Step count
        - Learner state (logZ, optimizer)
        - Policy state
        - Replay buffer contents
        - RNG states
        """
        ...

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore runner state from checkpoint."""
        ...
