"""Base protocol for training runners."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class RunResult:
    """Result of a training run."""

    metrics: dict[str, float] = field(default_factory=dict)
    checkpoints: list[str] = field(default_factory=list)
    interrupted: bool = False
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None


@runtime_checkable
class Runner(Protocol):
    """Protocol for training execution backends."""

    def run(self) -> RunResult: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...
