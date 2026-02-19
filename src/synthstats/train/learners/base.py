"""Base protocol for learners."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Learner(Protocol):
    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        """Update parameters from a batch, return metrics."""
        ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...

    @property
    def logZ(self) -> float:
        """Current logZ value."""
        ...
