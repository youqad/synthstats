"""BoxingGym environment for SkyRL's BaseTextEnv.

Wraps Task + ActionCodec + Executor + Judge into a single SkyRL-compatible env.
Import-safe: works without SkyRL installed.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from synthstats.core.constants import REWARD_FLOOR_DEFAULT
from synthstats.core.executor import execute_tool_call
from synthstats.core.process import cleanup_process_group
from synthstats.executors.pymc_sandbox import check_code_safety

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


def _rewrite_pm_sample_calls(code: str) -> str:
    """Rewrite pm.sample calls to bounded settings using AST (not regex)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    class _SampleCallRewriter(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            self.generic_visit(node)
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "pm"
                and node.func.attr == "sample"
            ):
                args = list(node.args)
                keywords = [
                    kw for kw in node.keywords if kw.arg not in {"draws", "chains", "cores"}
                ]

                if args:
                    args[0] = ast.Constant(value=200)
                else:
                    keywords.append(ast.keyword(arg="draws", value=ast.Constant(value=200)))

                keywords.append(ast.keyword(arg="chains", value=ast.Constant(value=2)))
                keywords.append(ast.keyword(arg="cores", value=ast.Constant(value=1)))
                node.args = args
                node.keywords = keywords
            return node

    try:
        rewritten = _SampleCallRewriter().visit(tree)
        ast.fix_missing_locations(rewritten)
        return ast.unparse(rewritten)
    except Exception:
        return code


@dataclass
class BoxingEnvConfig:
    """Configuration for BoxingEnv."""

    max_turns: int = 20
    reward_floor: float = REWARD_FLOOR_DEFAULT
    reward_scale: float = 1.0
    include_format_reward: bool = True
    format_reward_weight: float = 0.1


class BoxingEnv(BaseTextEnv):  # type: ignore[misc]
    """BoxingGym env with integrated Judge for automatic reward computation.

    Unlike SynthStatsTextEnv, calls Judge on episode completion and maintains
    trajectory history for reward scoring.
    """

    def __init__(
        self,
        task: Task,
        codec: ActionCodec,
        executors: dict[str, Executor] | None = None,
        judge: Judge | None = None,
        config: BoxingEnvConfig | None = None,
        *,
        name: str | None = None,
        max_steps: int | None = None,
    ) -> None:
        if hasattr(super(), "__init__"):
            super().__init__()

        self.task = task
        self.codec = codec
        self.executors: dict[str, Executor] = executors or {}
        self.judge = judge
        self.config = config or BoxingEnvConfig()

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

        self._state = self.task.reset()
        messages = self.task.observe(self._state)

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

        # execute tool calls (skip "query", handled by task.step)
        if isinstance(parsed_action, ToolCall) and parsed_action.name != "query":
            result = self._execute_tool(parsed_action)
            tool_msg = {"role": "tool", "content": result.output}
            self.chat_history.append(tool_msg)
            self._trajectory_messages.append(tool_msg)
            if not result.success:
                self._artifacts["tool_error"] = result.error

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
        return execute_tool_call(
            action,
            self.executors,
            timeout_s=30.0,
            unknown_prefix="",
            unknown_available_label="Available",
        )

    @staticmethod
    def _force_pymc_sample_limits(code: str) -> str:
        """Public helper for tests: rewrite pm.sample calls to fixed bounded settings."""
        return _rewrite_pm_sample_calls(code)

    def _execute_program_for_elpd(self, code: str) -> float | None:
        """Execute PyMC program in subprocess and compute ELPD-LOO.

        Uses rlimits to prevent runaway processes (30s CPU, 2GB memory).
        """
        import os as _os
        import subprocess
        import sys
        import tempfile

        is_safe, error = check_code_safety(code)
        if not is_safe:
            logger.warning("AST safety check failed: %s", error)
            return None

        # reduce sample count for training speed
        modified = self._force_pymc_sample_limits(code)

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
                "TINKER_API_KEY",
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
                    cleanup_process_group(proc)
                    logger.warning("Program timed out (180s)")
                    return None
                else:
                    cleanup_process_group(proc)

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

    def score_program(self, program: str) -> float:
        """Score a standalone program using the configured Judge.

        This is used for offline scoring, e.g. SFT warm-start reward computation.

        The scoring path mirrors online episodes:
        1. Execute the program to compute ELPD (best-effort).
        2. Delegate reward shaping/clipping to the configured Judge.
        3. Apply environment-level `reward_scale` and `reward_floor`.

        Returns:
            Final reward (after env reward scaling and reward floor).
        """
        # Compute ELPD via the same subprocess path used during online episodes.
        elpd = self._execute_program_for_elpd(program)
        if elpd is None:
            # If we cannot compute ELPD, treat as a failure and return a floor reward.
            return float(self.config.reward_floor)

        if self.judge is None:
            logger.warning("score_program called without a Judge; returning reward_floor")
            return float(self.config.reward_floor)

        from synthstats.core.types import Reward

        artifacts: dict[str, Any] = {"program": program, "elpd": elpd}

        # Judge only needs messages and artifacts. Provide an empty trajectory stub.
        trajectory_stub = type(
            "TrajectoryStub",
            (),
            {
                "messages": [],
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
                artifacts=artifacts,
            )
            reward = reward_obj.total * self.config.reward_scale
        except Exception as e:
            logger.warning(f"Judge scoring failed: {e}")
            reward = 0.0

        return max(float(reward), self.config.reward_floor)

    def _compute_final_reward(self) -> float:
        """Compute final reward using Judge if available."""
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

        # judge only needs messages and artifacts
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
