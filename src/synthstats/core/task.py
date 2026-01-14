"""Task Protocol - defines the interface for domain plugins.

A Task represents a specific problem domain (BoxingGym, SynthStats, ARC, SWE).
It manages episode state and provides observations to the policy.
"""

from typing import Any, Protocol, runtime_checkable

from synthstats.core.types import Action, Message, StepResult


@runtime_checkable
class Task(Protocol):
    """Domain plugin interface.

    Responsibilities:
    - Define what an "episode" is
    - Provide observations (messages/context) to the policy
    - Consume structured Actions (not raw text)
    - Decide done/next state
    - Provide artifacts needed for judging
    """

    name: str

    def reset(self, seed: int | None = None) -> Any:
        """Reset the task to initial state.

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            Initial task state (task-specific type).
        """
        ...

    def observe(self, state: Any) -> list[Message]:
        """Generate observation messages for the current state.

        Args:
            state: Current task state.

        Returns:
            List of messages to send to the policy.
        """
        ...

    def step(self, state: Any, action: Action) -> StepResult:
        """Execute an action and transition to next state.

        Args:
            state: Current task state.
            action: Structured action from the policy.

        Returns:
            StepResult with next_state, done flag, and artifacts.
        """
        ...
