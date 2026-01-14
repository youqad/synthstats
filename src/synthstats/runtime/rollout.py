"""Rollout loop for running episodes.

The rollout_episode function is the main entry point for running a single
episode with any task, policy, codec combination.
"""

from dataclasses import dataclass, field
from typing import Any

from synthstats.core.executor import Executor, ToolResult
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
    """Run a single episode and return the trajectory.

    This is the unified rollout loop used by all tasks. The loop:
    1. task.reset(seed) - initialize episode state
    2. task.observe(state) - get messages for policy
    3. policy.generate(messages) - sample from LLM
    4. codec.parse(generation.text) - parse into Action
    5. if ToolCall: executor.execute() - run tool, append result to messages
    6. task.step(state, action) - update state
    7. if not done, goto 2
    8. judge.score() - compute reward
    9. return Trajectory

    Args:
        task: Task instance providing reset/observe/step.
        policy: Policy instance for generation.
        codec: ActionCodec for parsing LLM output.
        executors: Dict mapping tool names to Executor instances.
        judge: Judge for computing reward.
        cfg: Rollout configuration.

    Returns:
        Trajectory with messages, token data, and reward.
    """
    # initialize episode
    state = task.reset(seed=cfg.seed)
    messages: list[Message] = []
    token_ids: list[list[int]] = []
    token_logprobs: list[list[float]] = []
    loss_mask: list[list[bool]] = []
    accumulated_artifacts: dict[str, Any] = {}
    done = False
    steps = 0

    while not done and steps < cfg.max_steps:
        # get observations
        obs_messages = task.observe(state)
        messages.extend(obs_messages)

        # generate response
        generation = policy.generate(messages, gen=cfg.gen_config)

        # add assistant message
        messages.append(Message(role="assistant", content=generation.text))
        token_ids.append(generation.token_ids)
        token_logprobs.append(generation.token_logprobs)
        # default: all tokens contribute to loss
        loss_mask.append([True] * len(generation.token_ids))

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
    )

    # score with judge
    final_reward = judge.score(
        task_name=task.name,
        trajectory=trajectory,
        artifacts=accumulated_artifacts,
    )

    # return trajectory with final reward
    return Trajectory(
        messages=trajectory.messages,
        token_ids=trajectory.token_ids,
        token_logprobs=trajectory.token_logprobs,
        loss_mask=trajectory.loss_mask,
        reward=final_reward,
    )


def _execute_tool(
    action: ToolCall,
    executors: dict[str, Executor],
    timeout_s: float,
) -> ToolResult:
    """Execute a tool call using the appropriate executor.

    Args:
        action: ToolCall to execute.
        executors: Dict mapping tool names to Executor instances.
        timeout_s: Timeout for execution.

    Returns:
        ToolResult from execution or error result if executor not found.
    """
    executor = executors.get(action.name)
    if executor is None:
        available = list(executors.keys())
        return ToolResult(
            output=(
                f"Error: Unknown tool '{action.name}'. "
                f"Available tools: {available}"
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
