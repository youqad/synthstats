"""SkyRL BaseTextEnv wrapper for SynthStats tasks.

Bridges Task/ActionCodec/Executor to SkyRL's conversation interface.
Import-safe: works without SkyRL installed.
"""

from __future__ import annotations

from typing import Any

from synthstats.core.constants import REWARD_FLOOR_DEFAULT
from synthstats.core.executor import Executor, ToolResult, execute_tool_call
from synthstats.core.task import Task
from synthstats.core.types import Message, ToolCall
from synthstats.runtime.codecs import ActionCodec, ParseError

# import-safe SkyRL types
try:
    from skyrl_gym.envs.base_text_env import (
        BaseTextEnv,
        BaseTextEnvStepOutput,
        ConversationType,
    )
except Exception:
    BaseTextEnv = object  # type: ignore[assignment,misc]
    BaseTextEnvStepOutput = dict[str, Any]  # type: ignore[misc,assignment]
    ConversationType = list[dict[str, str]]  # type: ignore[misc,assignment]


class SynthStatsTextEnv(BaseTextEnv):  # type: ignore[misc]
    """SkyRL BaseTextEnv wrapping a SynthStats Task."""

    def __init__(
        self,
        task: Task,
        codec: ActionCodec,
        executors: dict[str, Executor] | None = None,
        *,
        max_turns: int = 20,
        reward_floor: float = REWARD_FLOOR_DEFAULT,
    ) -> None:
        if hasattr(super(), "__init__"):
            super().__init__()

        self.task = task
        self.codec = codec
        self.executors: dict[str, Executor] = executors or {}
        self.max_turns = max_turns
        self.reward_floor = reward_floor

        self.chat_history: ConversationType = []
        self._state: Any = None
        self._turns: int = 0
        self._done: bool = False
        self._last_observation: str = ""

    def init(
        self, prompt: ConversationType | None = None
    ) -> tuple[ConversationType, dict[str, Any]]:
        """Initialize a new episode.

        Args:
            prompt: Optional custom system prompt (list of role/content dicts).
                    If None, uses messages from task.observe().

        Returns:
            Tuple of (chat_history, info_dict)
        """
        self._turns = 0
        self._done = False

        # reset task and get initial state
        self._state = self.task.reset()

        messages = self.task.observe(self._state)

        # convert to SkyRL conversation format
        base_prompt: ConversationType
        if prompt:
            base_prompt = list(prompt)
        else:
            base_prompt = []

        for msg in messages:
            base_prompt.append(self._message_to_dict(msg))

        self.chat_history = base_prompt

        # save last user message for error recovery
        for msg in reversed(messages):
            if msg.role == "user":
                self._last_observation = msg.content
                break

        return self.chat_history, {}

    def step(self, action: str) -> BaseTextEnvStepOutput:  # type: ignore[misc]
        """Take a step in the environment.

        Args:
            action: Raw action text from the policy (to be parsed by codec)

        Returns:
            Dict with keys: observations, reward, done, metadata
        """
        if self._done:
            return {"observations": [], "reward": 0.0, "done": True, "metadata": {}}

        self._turns += 1

        # add assistant message to history
        self.chat_history.append({"role": "assistant", "content": action})

        # check turn limit
        if self._turns > self.max_turns:
            self._done = True
            return {"observations": [], "reward": 0.0, "done": True, "metadata": {}}

        error: str | None = None
        reward = 0.0
        info: dict[str, Any] = {}

        # parse action
        try:
            parsed_action = self.codec.parse(action)
        except ParseError as e:
            error = str(e)
            obs_content = f"Parse error: {error}"
            if self._last_observation:
                obs_content = f"{obs_content}\n\n{self._last_observation}"
            new_obs = {"role": "user", "content": obs_content}
            self.chat_history.append(new_obs)
            info["error"] = error
            info["parse_error"] = True
            return {
                "observations": [new_obs],
                "reward": 0.0,
                "done": False,
                "metadata": info,
            }

        # execute tool calls if needed
        if isinstance(parsed_action, ToolCall):
            tool_result = self._execute_tool(parsed_action)
            tool_msg = {"role": "tool", "content": tool_result.output}
            self.chat_history.append(tool_msg)
            if not tool_result.success:
                info["tool_error"] = tool_result.error

        # step the task
        step_result = self.task.step(self._state, parsed_action)
        self._state = step_result.next_state
        done = step_result.done
        info.update(step_result.artifacts)

        # check turn limit again
        if self._turns >= self.max_turns:
            done = True

        if done:
            self._done = True
            # extract reward from artifacts if available
            if "reward" in step_result.artifacts:
                reward = float(step_result.artifacts["reward"])
            elif hasattr(step_result, "reward"):
                reward = float(getattr(step_result, "reward", 0.0))
            return {
                "observations": [],
                "reward": max(float(reward), self.reward_floor),
                "done": True,
                "metadata": info,
            }

        next_messages = self.task.observe(self._state)
        new_observations: list[dict[str, str]] = []
        for msg in next_messages:
            # skip system messages in subsequent observations
            if msg.role == "system":
                continue
            obs_dict = self._message_to_dict(msg)
            self.chat_history.append(obs_dict)
            new_observations.append(obs_dict)
            if msg.role == "user":
                self._last_observation = msg.content

        return {
            "observations": new_observations,
            "reward": float(reward),
            "done": False,
            "metadata": info,
        }

    def _execute_tool(self, action: ToolCall) -> ToolResult:
        """Execute a tool call."""
        return execute_tool_call(
            action,
            self.executors,
            timeout_s=30.0,
            unknown_prefix="Error: ",
            unknown_available_label="Available",
        )

    @staticmethod
    def _message_to_dict(msg: Message) -> dict[str, str]:
        """Convert SynthStats Message to SkyRL conversation dict."""
        return {"role": msg.role, "content": msg.content}

    def close(self) -> None:
        """Clean up resources."""
        pass

    def get_metrics(self) -> dict[str, Any]:
        """Return metrics for this environment."""
        return {
            "turns": self._turns,
            "done": self._done,
        }
