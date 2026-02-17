"""Executor protocol."""

from __future__ import annotations

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


def execute_tool_call(
    action: ToolCall,
    executors: dict[str, Executor],
    *,
    timeout_s: float,
    unknown_prefix: str = "Error: ",
    unknown_available_label: str = "Available tools",
) -> ToolResult:
    """Shared tool execution helper with consistent error handling."""
    executor = executors.get(action.name)
    if executor is None:
        available = list(executors.keys())
        return ToolResult(
            output=(
                f"{unknown_prefix}Unknown tool '{action.name}'. "
                f"{unknown_available_label}: {available}"
            ),
            success=False,
            error=f"Unknown tool: {action.name}",
        )

    try:
        return executor.execute(action, timeout_s=timeout_s)
    except Exception as e:
        return ToolResult(
            output=f"Error executing {action.name}: {e}",
            success=False,
            error=str(e),
        )
