"""Base protocol for learners.

Learners update model parameters from trajectory batches:
- SubTBTorchLearner: PyTorch optimizer-based updates
- SubTBTinkerLearner: Tinker API-based updates
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Learner(Protocol):
    """Protocol for parameter update implementations.

    Learners take batches of trajectories and update model parameters.
    They own the optimizer and handle gradient computation.

    Example:
        >>> learner = SubTBTorchLearner(objective, policy, cfg)
        >>> metrics = learner.update(batch)
        >>> print(f"Loss: {metrics['loss']}")
    """

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        """Update parameters from a batch.

        Args:
            batch: Dict with keys like log_probs, log_reward, mask, entropy

        Returns:
            Dict with metrics (loss, logZ, etc.)
        """
        ...

    def state_dict(self) -> dict[str, Any]:
        """Serialize learner state for checkpointing.

        Returns:
            Dict with optimizer state, logZ value, etc.
        """
        ...

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore learner state from checkpoint."""
        ...

    @property
    def logZ(self) -> float:
        """Current logZ value."""
        ...
