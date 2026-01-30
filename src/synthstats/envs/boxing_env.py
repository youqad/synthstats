"""Native BoxingGym environment using SkyRL's BaseTextEnv.

This is the canonical environment for BoxingGym tasks, integrating:
- Task: State machine for Boxing environments (Dugongs, Peregrines, etc.)
- ActionCodec: Parsing LLM output to structured actions
- Executors: PyMC sandbox, query execution
- Judge: Reward computation (ELPD-LOO, format checking)

Usage:
    from synthstats.envs import BoxingEnv
    from synthstats.tasks.boxing import DugongsTask
    from synthstats.runtime.codecs import BoxingCodec
    from synthstats.judges import LikelihoodJudge

    env = BoxingEnv(
        task=DugongsTask(),
        codec=BoxingCodec(),
        executors={"pymc": PyMCExecutor()},
        judge=LikelihoodJudge(),
    )

    obs, info = env.init()
    while True:
        action = policy.generate(obs)
        result = env.step(action)
        if result["done"]:
            break
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from synthstats.core.executor import Executor, ToolResult
    from synthstats.core.judge import Judge
    from synthstats.core.task import Task
    from synthstats.core.types import ToolCall
    from synthstats.runtime.codecs import ActionCodec

# import-safe SkyRL types
try:
    from skyrl_gym.envs.base_text_env import (
        BaseTextEnv,
        BaseTextEnvStepOutput,
        ConversationType,
    )

    SKYRL_AVAILABLE = True
except ImportError:
    BaseTextEnv = object  # type: ignore[assignment,misc]
    BaseTextEnvStepOutput = dict[str, Any]  # type: ignore[misc,assignment]
    ConversationType = list[dict[str, str]]  # type: ignore[misc,assignment]
    SKYRL_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class BoxingEnvConfig:
    """Configuration for BoxingEnv."""

    max_turns: int = 20
    reward_floor: float = 1e-4
    reward_scale: float = 1.0
    include_format_reward: bool = True
    format_reward_weight: float = 0.1


class BoxingEnv(BaseTextEnv):  # type: ignore[misc]
    """Native BoxingGym environment for SkyRL training.

    This environment wraps a BoxingGym Task with integrated Judge support
    for computing rewards. It implements SkyRL's BaseTextEnv interface.

    Key differences from SynthStatsTextEnv:
    - Judge is integrated and called automatically on episode completion
    - Reward computation includes format checking and likelihood scoring
    - Trajectory history is maintained for Judge input

    Args:
        task: BoxingGym Task instance
        codec: ActionCodec for parsing LLM output
        executors: Dict mapping tool names to Executor instances
        judge: Judge instance for reward computation (optional)
        config: BoxingEnvConfig with hyperparameters
    """

    def __init__(
        self,
        task: Task,
        codec: ActionCodec,
        executors: dict[str, Executor] | None = None,
        judge: Judge | None = None,
        config: BoxingEnvConfig | None = None,
    ) -> None:
        if hasattr(super(), "__init__"):
            super().__init__()

        self.task = task
        self.codec = codec
        self.executors: dict[str, Executor] = executors or {}
        self.judge = judge
        self.config = config or BoxingEnvConfig()

        # set max_turns from config
        self.max_turns = self.config.max_turns

        # episode state
        self.chat_history: ConversationType = []
        self._state: Any = None
        self._turns: int = 0
        self._done: bool = False
        self._trajectory_messages: list[dict[str, str]] = []
        self._artifacts: dict[str, Any] = {}

    def init(
        self, prompt: ConversationType | None = None
    ) -> tuple[ConversationType, dict[str, Any]]:
        """Initialize a new episode.

        Args:
            prompt: Optional initial prompt (list of role/content dicts).
                    If None, uses messages from task.observe().

        Returns:
            Tuple of (chat_history, metadata)
        """
        self._turns = 0
        self._done = False
        self._trajectory_messages = []
        self._artifacts = {}

        # reset task
        self._state = self.task.reset()

        # get initial observations
        messages = self.task.observe(self._state)

        # build chat history
        if prompt:
            self.chat_history = list(prompt)
        else:
            self.chat_history = []

        for msg in messages:
            msg_dict = {"role": msg.role, "content": msg.content}
            self.chat_history.append(msg_dict)
            self._trajectory_messages.append(msg_dict)

        return self.chat_history, {"task_name": self.task.name}

    def step(self, action: str) -> BaseTextEnvStepOutput:  # type: ignore[misc]
        """Take a step in the environment.

        Args:
            action: Raw action text from the policy

        Returns:
            BaseTextEnvStepOutput with observations, reward, done, metadata
        """
        from synthstats.core.types import ToolCall
        from synthstats.runtime.codecs import ParseError

        if self._done:
            return {
                "observations": [],
                "reward": 0.0,
                "done": True,
                "metadata": {"already_done": True},
            }

        self._turns += 1

        # record assistant action
        assistant_msg = {"role": "assistant", "content": action}
        self.chat_history.append(assistant_msg)
        self._trajectory_messages.append(assistant_msg)

        # check turn limit
        if self._turns > self.max_turns:
            self._done = True
            reward = self._compute_final_reward()
            return {
                "observations": [],
                "reward": reward,
                "done": True,
                "metadata": {"max_turns_exceeded": True, **self._artifacts},
            }

        # parse action
        try:
            parsed_action = self.codec.parse(action)
        except ParseError as e:
            error_msg = f"Parse error: {e}"
            obs = {"role": "user", "content": error_msg}
            self.chat_history.append(obs)
            self._trajectory_messages.append(obs)
            return {
                "observations": [obs],
                "reward": 0.0,
                "done": False,
                "metadata": {"parse_error": str(e)},
            }

        # execute tool calls (skip "query" — handled by task.step)
        if isinstance(parsed_action, ToolCall) and parsed_action.name != "query":
            result = self._execute_tool(parsed_action)
            tool_msg = {"role": "tool", "content": result.output}
            self.chat_history.append(tool_msg)
            self._trajectory_messages.append(tool_msg)
            if not result.success:
                self._artifacts["tool_error"] = result.error

        # step task
        step_result = self.task.step(self._state, parsed_action)
        self._state = step_result.next_state
        self._artifacts.update(step_result.artifacts)

        done = step_result.done or self._turns >= self.max_turns

        if done:
            self._done = True
            reward = self._compute_final_reward()
            return {
                "observations": [],
                "reward": reward,
                "done": True,
                "metadata": self._artifacts,
            }

        # get next observation
        next_messages = self.task.observe(self._state)
        new_observations: list[dict[str, str]] = []
        for msg in next_messages:
            if msg.role == "system":
                continue
            obs_dict = {"role": msg.role, "content": msg.content}
            self.chat_history.append(obs_dict)
            self._trajectory_messages.append(obs_dict)
            new_observations.append(obs_dict)

        return {
            "observations": new_observations,
            "reward": 0.0,
            "done": False,
            "metadata": {},
        }

    def _execute_tool(self, action: ToolCall) -> ToolResult:
        """Execute a tool call."""
        from synthstats.core.executor import ToolResult

        executor = self.executors.get(action.name)
        if executor is None:
            available = list(self.executors.keys())
            return ToolResult(
                output=f"Unknown tool '{action.name}'. Available: {available}",
                success=False,
                error=f"Unknown tool: {action.name}",
            )

        try:
            return executor.execute(action, timeout_s=30.0)
        except Exception as e:
            return ToolResult(
                output=f"Error executing {action.name}: {e}",
                success=False,
                error=str(e),
            )

    def _execute_program_for_elpd(self, code: str) -> float | None:
        """Execute PyMC program in subprocess and compute ELPD-LOO.

        Uses rlimits to prevent runaway processes (30s CPU, 2GB memory).
        """
        import os as _os
        import re as _re
        import signal
        import subprocess
        import sys
        import tempfile

        # reduce sample count for training speed
        modified = _re.sub(
            r"pm\.sample\((\d+)",
            "pm.sample(200, chains=2, cores=1",
            code,
        )

        elpd_snippet = """
# --- synthstats: compute ELPD-LOO ---
import arviz as _az
try:
    if not hasattr(idata, 'log_likelihood'):
        import pymc as _pm
        _pm.compute_log_likelihood(idata, model=model)
    _loo = _az.loo(idata, pointwise=True)
    print(f"__ELPD__:{_loo.elpd_loo}")
except Exception as _e:
    print(f"__ELPD_ERR__:{_e}")
"""

        full_code = modified + "\n" + elpd_snippet

        def _set_rlimits() -> None:
            """Unix resource limits for subprocess."""
            try:
                import resource
                import sys as _sys

                resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
                if _sys.platform == "linux":
                    mem_limit = 2 * 1024 * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
            except (ImportError, AttributeError, ValueError, OSError):
                pass

        try:
            sub_env = _os.environ.copy()
            sub_env["PYTENSOR_FLAGS"] = "device=cpu,cxx="
            for var in [
                "SSH_AUTH_SOCK",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_ACCESS_KEY_ID",
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "HF_TOKEN",
                "WANDB_API_KEY",
            ]:
                sub_env.pop(var, None)

            with tempfile.TemporaryDirectory(prefix="synthstats_") as tmpdir:
                proc = subprocess.Popen(
                    [sys.executable, "-"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=tmpdir,
                    env=sub_env,
                    preexec_fn=_set_rlimits,
                    start_new_session=True,
                )
                try:
                    stdout, stderr = proc.communicate(input=full_code, timeout=180.0)
                except subprocess.TimeoutExpired:
                    # kill entire process group (start_new_session=True creates new pgid)
                    # NOTE: Avoid hanging in cleanup if something goes wrong.
                    try:
                        if hasattr(_os, "killpg") and hasattr(_os, "getpgid"):
                            pgid = _os.getpgid(proc.pid)
                            for sig in (signal.SIGTERM, signal.SIGKILL):
                                try:
                                    _os.killpg(pgid, sig)
                                except (ProcessLookupError, OSError):
                                    pass
                                try:
                                    proc.wait(timeout=2.0)
                                    break
                                except subprocess.TimeoutExpired:
                                    continue
                        else:
                            proc.kill()
                            proc.wait(timeout=2.0)
                    except Exception:
                        # best-effort fallback: kill the parent process
                        try:
                            proc.kill()
                        except Exception:
                            pass
                        try:
                            proc.wait(timeout=2.0)
                        except Exception:
                            pass
                    logger.warning("Program timed out (180s)")
                    return None

            for line in stdout.split("\n"):
                if line.startswith("__ELPD__:"):
                    try:
                        return float(line.split(":", 1)[1])
                    except ValueError:
                        pass
                elif line.startswith("__ELPD_ERR__:"):
                    logger.warning(f"ELPD error: {line}")

            if proc.returncode != 0:
                logger.warning(f"Program failed: {stderr[:300]}")
        except Exception as e:
            logger.warning(f"Program exec error: {e}")

        return None

    def _compute_final_reward(self) -> float:
        """Compute final reward using Judge if available."""
        # execute submitted program to compute ELPD
        if "program" in self._artifacts and "elpd" not in self._artifacts:
            elpd = self._execute_program_for_elpd(self._artifacts["program"])
            if elpd is not None:
                self._artifacts["elpd"] = elpd
                logger.info(f"Program ELPD-LOO: {elpd:.4f}")

        if self.judge is None:
            # fall back to artifact reward
            reward = self._artifacts.get("reward", 0.0)
            return max(float(reward), self.config.reward_floor)

        # build minimal trajectory for Judge
        from synthstats.core.types import Message, Reward

        messages = [
            Message(role=m["role"], content=m["content"]) for m in self._trajectory_messages
        ]

        # create trajectory stub for Judge
        # Judge only needs messages and artifacts
        trajectory_stub = type(
            "TrajectoryStub",
            (),
            {
                "messages": messages,
                "reward": Reward(0.0, {}, {}),
                "token_ids": [],
                "token_logprobs": [],
                "loss_mask": [],
                "eos_logprobs": [],
            },
        )()

        try:
            reward_obj = self.judge.score(
                task_name=self.task.name,
                trajectory=trajectory_stub,  # type: ignore
                artifacts=self._artifacts,
            )
            reward = reward_obj.total * self.config.reward_scale
            self._artifacts["reward_components"] = reward_obj.components
            self._artifacts["reward_info"] = reward_obj.info
        except Exception as e:
            logger.warning(f"Judge scoring failed: {e}")
            reward = self._artifacts.get("reward", 0.0)

        return max(float(reward), self.config.reward_floor)

    def close(self) -> None:
        """Clean up resources."""
        pass

    def get_metrics(self) -> dict[str, Any]:
        """Return episode metrics."""
        metrics = {
            "turns": self._turns,
            "done": self._done,
            "task_name": self.task.name,
        }
        if "reward_components" in self._artifacts:
            metrics["reward_components"] = self._artifacts["reward_components"]
        return metrics


__all__ = ["BoxingEnv", "BoxingEnvConfig", "SKYRL_AVAILABLE"]
