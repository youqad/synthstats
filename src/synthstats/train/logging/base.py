"""Base protocol for logging sinks.

LoggerSinks handle metric logging to various backends:
- StdoutLogger: Console output
- WandbLogger: Weights & Biases
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LoggerSink(Protocol):
    """Protocol for logging implementations.

    Logger sinks receive metrics at each training step and forward them
    to their backend (console, W&B, TensorBoard, etc.).

    Example:
        >>> logger = WandbLogger(project="synthstats")
        >>> logger.log(step=100, metrics={"loss": 0.5, "logZ": 1.2})
        >>> logger.close()
    """

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        """Log metrics for a training step.

        Args:
            step: Current training step
            metrics: Dict of metric names to values
        """
        ...

    def close(self) -> None:
        """Close the logger and flush any pending data."""
        ...


class NullLogger:
    """No-op logger for when logging is disabled."""

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        """Do nothing."""
        pass

    def close(self) -> None:
        """Do nothing."""
        pass
