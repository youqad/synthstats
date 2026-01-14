"""Reusable Hypothesis strategies for SynthStats property tests."""

from __future__ import annotations

from typing import Any

import torch
from hypothesis import strategies as st

from synthstats.core.types import FinalAnswer, Message, Program, Reward, ToolCall, Trajectory
from synthstats.training.buffers.gfn_replay import BufferEntry


@st.composite
def st_finite_floats(
    draw: st.DrawFn,
    min_value: float = -1e6,
    max_value: float = 1e6,
) -> float:
    """Generate finite floats (no NaN/Inf)."""
    return draw(
        st.floats(
            min_value=min_value,
            max_value=max_value,
            allow_nan=False,
            allow_infinity=False,
        )
    )


@st.composite
def st_log_probs(
    draw: st.DrawFn,
    min_length: int = 1,
    max_length: int = 20,
) -> torch.Tensor:
    """Generate valid log probability tensors (negative values)."""
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    values = draw(
        st.lists(
            st.floats(
                min_value=-20.0,
                max_value=0.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=length,
            max_size=length,
        )
    )
    return torch.tensor(values, dtype=torch.float32)


@st.composite
def st_positive_rewards(
    draw: st.DrawFn,
    min_value: float = 1e-6,
    max_value: float = 100.0,
) -> float:
    """Generate positive reward values."""
    return draw(
        st.floats(
            min_value=min_value,
            max_value=max_value,
            allow_nan=False,
            allow_infinity=False,
        )
    )


@st.composite
def st_message(draw: st.DrawFn) -> Message:
    """Generate a random Message."""
    role = draw(st.sampled_from(["system", "user", "assistant", "tool"]))
    content = draw(
        st.text(
            min_size=1,
            max_size=100,
            alphabet=st.characters(blacklist_categories=("Cs",)),
        )
    )
    return Message(role=role, content=content)


@st.composite
def st_messages(
    draw: st.DrawFn,
    min_size: int = 1,
    max_size: int = 5,
) -> list[Message]:
    """Generate a list of Messages."""
    return draw(st.lists(st_message(), min_size=min_size, max_size=max_size))


@st.composite
def st_reward(
    draw: st.DrawFn,
    min_total: float = -10.0,
    max_total: float = 10.0,
) -> Reward:
    """Generate a Reward object."""
    total = draw(
        st.floats(
            min_value=min_total,
            max_value=max_total,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    return Reward(total=total, components={}, info={})


@st.composite
def st_trajectory(
    draw: st.DrawFn,
    min_steps: int = 1,
    max_steps: int = 5,
) -> Trajectory:
    """Generate a valid Trajectory."""
    n_steps = draw(st.integers(min_value=min_steps, max_value=max_steps))

    messages = draw(st_messages(min_size=1, max_size=n_steps + 1))

    # token_ids: one list per step
    token_ids = [
        draw(st.lists(st.integers(min_value=0, max_value=50000), min_size=1, max_size=20))
        for _ in range(n_steps)
    ]

    # token_logprobs: matching shape
    token_logprobs = [
        draw(
            st.lists(
                st.floats(
                    min_value=-20.0,
                    max_value=0.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=len(token_ids[i]),
                max_size=len(token_ids[i]),
            )
        )
        for i in range(n_steps)
    ]

    # loss_mask: matching shape
    loss_mask = [
        draw(st.lists(st.booleans(), min_size=len(token_ids[i]), max_size=len(token_ids[i])))
        for i in range(n_steps)
    ]

    reward = draw(st_reward())

    return Trajectory(
        messages=messages,
        token_ids=token_ids,
        token_logprobs=token_logprobs,
        loss_mask=loss_mask,
        reward=reward,
    )


@st.composite
def st_action_dict(draw: st.DrawFn) -> dict[str, Any]:
    """Generate a valid action dictionary."""
    action_type = draw(
        st.sampled_from(["query", "compute_eig", "submit_program", "answer"])
    )
    payload = draw(
        st.text(
            min_size=0,
            max_size=50,
            alphabet=st.characters(blacklist_categories=("Cs",)),
        )
    )
    return {"type": action_type, "payload": payload}


@st.composite
def st_buffer_entry(draw: st.DrawFn) -> BufferEntry:
    """Generate a valid BufferEntry for GFNReplayBuffer."""
    n_actions = draw(st.integers(min_value=1, max_value=5))
    actions = [draw(st_action_dict()) for _ in range(n_actions)]
    observations = [
        draw(
            st.text(
                min_size=1,
                max_size=50,
                alphabet=st.characters(blacklist_categories=("Cs",)),
            )
        )
        for _ in range(n_actions)
    ]
    log_reward = draw(
        st.floats(
            min_value=-20.0,
            max_value=0.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    policy_version = draw(st.integers(min_value=0, max_value=100))
    temperature = draw(
        st.floats(
            min_value=0.1,
            max_value=2.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )

    return BufferEntry(
        actions=actions,
        observations=observations,
        log_reward=log_reward,
        policy_version=policy_version,
        temperature=temperature,
    )


# action type strategies for codec tests

@st.composite
def st_final_answer(draw: st.DrawFn) -> FinalAnswer:
    """Generate a FinalAnswer action."""
    text = draw(
        st.text(
            min_size=0,
            max_size=100,
            alphabet=st.characters(blacklist_categories=("Cs",)),
        )
    )
    return FinalAnswer(text=text)


@st.composite
def st_tool_call(draw: st.DrawFn) -> ToolCall:
    """Generate a ToolCall action with JSON-serializable input."""
    name = draw(st.text(min_size=1, max_size=30, alphabet="abcdefghijklmnopqrstuvwxyz_"))

    # generate simple JSON-serializable dict
    n_keys = draw(st.integers(min_value=0, max_value=3))
    input_dict: dict[str, Any] = {}
    for _ in range(n_keys):
        key = draw(st.text(min_size=1, max_size=15, alphabet="abcdefghijklmnopqrstuvwxyz_"))
        value = draw(
            st.one_of(
                st.text(
                    min_size=0,
                    max_size=30,
                    alphabet=st.characters(blacklist_categories=("Cs",)),
                ),
                st.integers(min_value=-1000, max_value=1000),
                st.booleans(),
            )
        )
        input_dict[key] = value

    return ToolCall(name=name, input=input_dict, raw="")


@st.composite
def st_program(draw: st.DrawFn) -> Program:
    """Generate a Program action."""
    code = draw(
        st.text(
            min_size=1,
            max_size=200,
            alphabet=st.characters(blacklist_categories=("Cs",)),
        )
    )
    # matches Program dataclass: "pymc" | "numpyro" | "lazyppl"
    language = draw(st.sampled_from(["pymc", "numpyro", "lazyppl"]))
    return Program(code=code, language=language)
