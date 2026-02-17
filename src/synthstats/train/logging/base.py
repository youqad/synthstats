"""Base protocol for logging sinks."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LoggerSink(Protocol):
    """Protocol for logging implementations."""

    def log(self, step: int, metrics: dict[str, Any]) -> None: ...

    def close(self) -> None: ...
