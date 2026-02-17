"""Rollout loop for running episodes.

The rollout_episode function is the main entry point for running a single
episode with any task, policy, codec combination.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from synthstats.core.executor import Executor, ToolResult, execute_tool_call
from synthstats.core.judge import Judge
from synthstats.core.policy import GenConfig, Policy
from synthstats.core.task import Task
from synthstats.core.types import (
    Message,
    Reward,
    ToolCall,
    Trajectory,
)
from synthstats.runtime.codecs import ActionCodec, ParseError


@dataclass
class RolloutConfig:
    """Configuration for rollout episodes."""

    max_steps: int = 20
    tool_timeout_s: float = 30.0
    seed: int | None = None
    gen_config: GenConfig = field(default_factory=GenConfig)


def rollout_episode(
    task: Task,
    policy: Policy,
    codec: ActionCodec,
    executors: dict[str, Executor],
    judge: Judge,
    cfg: RolloutConfig,
) -> Trajectory:
    """Run a single episode: reset -> observe -> generate -> parse -> step -> score.

    Returns:
        Trajectory with messages, token data, and reward.
    """
    # initialize episode
    state = task.reset(seed=cfg.seed)
    messages: list[Message] = []
    token_ids: list[list[int]] = []
    token_logprobs: list[list[float]] = []
    loss_mask: list[list[bool]] = []
    eos_logprobs: list[list[float]] = []
    accumulated_artifacts: dict[str, Any] = {}
    done = False
    steps = 0

    while not done and steps < cfg.max_steps:
        obs_messages = task.observe(state)
        messages.extend(obs_messages)

        # generate response
        generation = policy.generate(messages, gen=cfg.gen_config)

        messages.append(Message(role="assistant", content=generation.text))
        token_ids.append(generation.token_ids)
        token_logprobs.append(generation.token_logprobs)
        loss_mask.append(_build_loss_mask(policy, generation.text, generation.token_ids))
        eos_logprobs.append(generation.eos_logprobs)

        # parse action
        try:
            action = codec.parse(generation.text)
        except ParseError as e:
            # handle parse error by adding error message and continuing
            error_msg = f"Parse error: {e}. Please provide a valid action."
            messages.append(Message(role="user", content=error_msg))
            steps += 1
            continue

        # execute tool calls
        if isinstance(action, ToolCall):
            tool_result = _execute_tool(action, executors, cfg.tool_timeout_s)
            messages.append(
                Message(
                    role="tool",
                    content=tool_result.output,
                    tool_call_id=action.name,
                )
            )

        # step the task
        step_result = task.step(state, action)
        state = step_result.next_state
        done = step_result.done
        accumulated_artifacts.update(step_result.artifacts)
        steps += 1

    # build trajectory (without final reward yet)
    trajectory = Trajectory(
        messages=messages,
        token_ids=token_ids,
        token_logprobs=token_logprobs,
        loss_mask=loss_mask,
        reward=Reward(total=0.0, components={}, info={}),  # placeholder
        eos_logprobs=eos_logprobs,
    )

    # score with judge
    final_reward = judge.score(
        task_name=task.name,
        trajectory=trajectory,
        artifacts=accumulated_artifacts,
    )

    return Trajectory(
        messages=trajectory.messages,
        token_ids=trajectory.token_ids,
        token_logprobs=trajectory.token_logprobs,
        loss_mask=trajectory.loss_mask,
        reward=final_reward,
        eos_logprobs=trajectory.eos_logprobs,
    )


def _build_loss_mask(policy: Policy, text: str, token_ids: list[int]) -> list[bool]:
    """Build per-token loss mask.

    TB/SubTB optimize over the full latent trajectory (Z, Y), so all generated
    tokens (including <think> reasoning) are included in the loss by default.
    """
    del policy, text
    return [True] * len(token_ids)


def _execute_tool(
    action: ToolCall,
    executors: dict[str, Executor],
    timeout_s: float,
) -> ToolResult:
    """Execute a tool call."""
    return execute_tool_call(
        action,
        executors,
        timeout_s=timeout_s,
        unknown_prefix="Error: ",
        unknown_available_label="Available tools",
    )
