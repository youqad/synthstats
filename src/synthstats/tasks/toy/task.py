"""Toy task for smoke tests."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from synthstats.core.types import Message, StepResult


@dataclass
class ToyState:

    step: int = 0
    done: bool = False


class ToyTask:

    name = "toy"

    def reset(self, seed: int | None = None) -> ToyState:
        if seed is not None:
            random.seed(seed)
        return ToyState(step=0, done=False)

    def observe(self, state: ToyState) -> list[Message]:
        return [
            Message(role="system", content="You are a test agent."),
            Message(role="user", content="Please provide a JSON answer."),
        ]

    def step(self, state: ToyState, action: Any) -> StepResult:
        return StepResult(
            next_state=ToyState(step=state.step + 1, done=True),
            done=True,
            artifacts={"completed": True},
        )
