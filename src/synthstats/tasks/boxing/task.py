"""BoxingTask - BoxingGym Task implementation.

The task presents a scientific discovery problem where the agent must:
1. Query the environment to collect observations
2. Submit a probabilistic program that explains the data
"""

from __future__ import annotations

from dataclasses import dataclass, field

from synthstats.core.types import Action, Message, Program, StepResult, ToolCall


@dataclass
class BoxingState:
    """State for a BoxingGym episode."""

    observations: list[str] = field(default_factory=list)
    step: int = 0
    done: bool = False


class BoxingTask:
    """BoxingGym Task implementation.

    The task presents a scientific discovery problem where the agent must:
    1. Query the environment to collect observations
    2. Submit a probabilistic program that explains the data
    """

    name = "boxing"

    def __init__(self, env_name: str = "dugongs", max_steps: int = 10):
        self.env_name = env_name
        self.max_steps = max_steps
        self.env = self._load_env(env_name)

    def _load_env(self, env_name: str):
        """Load the specified environment."""
        # Prefer boxing_gym environments when available.
        try:
            from synthstats.tasks.boxing.envs.boxing_gym_adapter import (
                load_boxing_gym_env,
            )

            boxing_env = load_boxing_gym_env(env_name)
            if boxing_env is not None:
                return boxing_env
        except Exception:
            # Fall back to stub envs below.
            pass

        if env_name == "dugongs":
            from synthstats.tasks.boxing.envs.dugongs_env import DugongsEnv

            return DugongsEnv()
        elif env_name == "peregrines":
            from synthstats.tasks.boxing.envs.peregrines_env import PeregrinesEnv

            return PeregrinesEnv()
        elif env_name == "eight_schools":
            from synthstats.tasks.boxing.envs.eight_schools_env import EightSchoolsEnv

            return EightSchoolsEnv()
        elif env_name == "surgical":
            from synthstats.tasks.boxing.envs.surgical_env import SurgicalEnv

            return SurgicalEnv()
        raise ValueError(
            f"Unknown environment: {env_name}. "
            "If this is a BoxingGym env, ensure boxing_gym is importable and "
            "dependencies like pymc/arviz are installed."
        )

    def reset(self, seed: int | None = None) -> BoxingState:
        """Reset the task to initial state.

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            Initial BoxingState.
        """
        self.env.reset(seed=seed)
        return BoxingState(observations=[], step=0, done=False)

    def system_prompt(self) -> str:
        """Return the system prompt for this task."""
        return (
            f"You are a scientist discovering a statistical model for {self.env_name}.\n"
            "\n"
            "## Actions\n"
            "\n"
            "1. Query the environment to collect data (vary the age each time):\n"
            '<tool_call>{"name": "query", "input": {"query": "age=3"}}</tool_call>\n'
            '<tool_call>{"name": "query", "input": {"query": "age=15"}}</tool_call>\n'
            '<tool_call>{"name": "query", "input": {"query": "age=25"}}</tool_call>\n'
            "\n"
            "2. When you have enough data, submit a PyMC program:\n"
            "<submit_program>\n"
            "import pymc as pm\n"
            "import numpy as np\n"
            "\n"
            "ages = np.array([...])  # your observed ages\n"
            "lengths = np.array([...])  # your observed lengths\n"
            "\n"
            "with pm.Model() as model:\n"
            '    alpha = pm.Normal("alpha", mu=3, sigma=1)\n'
            '    beta = pm.Normal("beta", mu=1, sigma=1)\n'
            '    lam = pm.Beta("lam", alpha=2, beta=2)\n'
            '    sigma = pm.HalfNormal("sigma", sigma=1)\n'
            "    mu = alpha - beta * lam**ages\n"
            '    obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=lengths)\n'
            "    idata = pm.sample(1000, return_inferencedata=True)\n"
            "</submit_program>\n"
            "\n"
            "Collect 2-3 data points at DIFFERENT ages, then submit your PyMC model immediately."
        )

    def observe(self, state: BoxingState) -> list[Message]:
        """Generate observation messages for the current state.

        Args:
            state: Current BoxingState.

        Returns:
            List of messages (system prompt + observations).
        """
        system_msg = Message(role="system", content=self.system_prompt())
        obs_content = (
            "\n".join(state.observations) if state.observations else "No observations yet."
        )
        user_msg = Message(role="user", content=f"Observations:\n{obs_content}")
        return [system_msg, user_msg]

    def step(self, state: BoxingState, action: Action) -> StepResult:
        """Execute an action and transition to next state.

        Args:
            state: Current BoxingState.
            action: Structured action (ToolCall or Program).

        Returns:
            StepResult with next_state, done flag, and artifacts.
        """
        if isinstance(action, Program):
            # submit program - episode ends
            next_state = BoxingState(
                observations=list(state.observations),
                step=state.step + 1,
                done=True,
            )
            return StepResult(
                next_state=next_state,
                done=True,
                artifacts={"program": action.code},
            )
        elif isinstance(action, ToolCall) and action.name == "query":
            # run experiment
            query = action.input.get("query", "")
            result = self.env.query(query)
            new_observations = list(state.observations)
            new_observations.append(f"Query: {query}\nResult: {result}")
            new_step = state.step + 1
            done = new_step >= self.max_steps
            next_state = BoxingState(
                observations=new_observations,
                step=new_step,
                done=done,
            )
            return StepResult(
                next_state=next_state,
                done=done,
                artifacts={},
            )
        else:
            # unknown action - no change
            return StepResult(next_state=state, done=False, artifacts={})
