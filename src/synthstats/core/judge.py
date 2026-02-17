"""Judge protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from synthstats.core.types import Reward, Trajectory


@runtime_checkable
class Judge(Protocol):
    """Reward computation interface.

    SynthStats needs multi-part reward:
    - likelihood / posterior score
    - format validity
    - LLM critique / process reward model
    - uncertainty/entropy penalties

    Judges are composed via CompositeJudge.
    """

    def score(self, *, task_name: str, trajectory: Trajectory, artifacts: dict) -> Reward:
        """Compute reward for a trajectory.

        Args:
            task_name: Name of the task (for task-specific scoring).
            trajectory: Complete episode trajectory.
            artifacts: Additional data from task steps (e.g., execution outputs).

        Returns:
            Reward with total, components breakdown, and info.
        """
        ...
