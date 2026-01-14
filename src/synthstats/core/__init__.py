"""Core module - types and protocols."""

from synthstats.core.executor import Executor, ToolResult
from synthstats.core.judge import Judge
from synthstats.core.policy import GenConfig, Generation, Policy, TokenLogProbs
from synthstats.core.task import Task
from synthstats.core.types import (
    Action,
    FinalAnswer,
    Message,
    Program,
    Reward,
    StepResult,
    ToolCall,
    Trajectory,
)

__all__ = [
    # Types
    "Action",
    "FinalAnswer",
    "Message",
    "Program",
    "Reward",
    "StepResult",
    "ToolCall",
    "Trajectory",
    # Protocols
    "Executor",
    "Judge",
    "Policy",
    "Task",
    # Supporting types
    "GenConfig",
    "Generation",
    "TokenLogProbs",
    "ToolResult",
]
