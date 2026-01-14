"""Executor Protocol - defines the interface for safe tool runtimes.

Executors run tool calls in sandboxed environments with safety checks.
Examples: Python sandbox, PyMC sandbox, Bash/Docker for SWE.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from synthstats.core.types import ToolCall


@dataclass
class ToolResult:
    """Result of executing a tool call."""

    output: str
    success: bool
    error: str | None = None


@runtime_checkable
class Executor(Protocol):
    """Safe tool runtime interface.

    Executors are shared infrastructure:
    - Python sandbox executor
    - PyMC executor (for boxing + synthstats)
    - Bash/Docker executor (for SWE)

    They must NOT depend on any particular task.
    """

    name: str

    def execute(self, payload: ToolCall, *, timeout_s: float) -> ToolResult:
        """Execute a tool call safely.

        Args:
            payload: The tool call to execute.
            timeout_s: Maximum execution time in seconds.

        Returns:
            ToolResult with output, success flag, and optional error.
        """
        ...
