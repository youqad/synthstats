"""Trajectory collectors for training.

Collectors generate trajectories by running episodes with a policy.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import torch

from synthstats.core.types import Action, FinalAnswer, Program, ToolCall

# type aliases
PolicyFnLegacy = Callable[[str], tuple[dict[str, Any], float | torch.Tensor, float | torch.Tensor]]
PolicyFnWithTemp = Callable[
    [str, float], tuple[dict[str, Any], float | torch.Tensor, float | torch.Tensor]
]
PolicyFn = PolicyFnLegacy | PolicyFnWithTemp
ScoreFnLegacy = Callable[[str, dict[str, Any]], tuple[float | torch.Tensor, float | torch.Tensor]]
ScoreFnWithTemp = Callable[
    [str, dict[str, Any], float], tuple[float | torch.Tensor, float | torch.Tensor]
]
ScoreFn = ScoreFnLegacy | ScoreFnWithTemp


@dataclass
class CollectedTrajectory:
    """A collected trajectory with policy data."""

    observations: list[str]
    actions: list[dict[str, Any]]
    log_probs: torch.Tensor  # [T]
    entropy: torch.Tensor  # [T]
    reward: float
    ref_log_probs: torch.Tensor | None = None
    temperature: float = 1.0
    eos_logprobs: torch.Tensor | None = None  # [T] log P(EOS|state)
    prompts: list[str] | None = None
    completions: list[str] | None = None

    def detach(self) -> CollectedTrajectory:
        """Return copy with detached tensors on CPU."""
        ref_lp = self.ref_log_probs.detach().cpu() if self.ref_log_probs is not None else None
        eos_lp = self.eos_logprobs.detach().cpu() if self.eos_logprobs is not None else None
        return CollectedTrajectory(
            observations=self.observations,
            actions=self.actions,
            log_probs=self.log_probs.detach().cpu(),
            ref_log_probs=ref_lp,
            entropy=self.entropy.detach().cpu(),
            reward=self.reward,
            temperature=self.temperature,
            eos_logprobs=eos_lp,
            prompts=self.prompts,
            completions=self.completions,
        )


class TrajectoryCollector:
    """Collector that generates trajectories from env + policy.

    Args:
        env: Environment with init() and step() methods
        policy_fn: Policy callable
        score_fn: Optional scoring function for ref-policy correction
    """

    def __init__(
        self,
        env: Any,
        policy_fn: PolicyFn,
        score_fn: ScoreFn | None = None,
    ) -> None:
        self.env = env
        self.policy_fn = policy_fn

        if score_fn is None and hasattr(policy_fn, "score_action"):
            self.score_fn = policy_fn.score_action
            self._score_fn_is_default = True
        else:
            self.score_fn = score_fn
            self._score_fn_is_default = False

        # detect signatures
        try:
            sig = inspect.signature(policy_fn)
            self._policy_accepts_temp = len(sig.parameters) >= 2
        except (ValueError, TypeError):
            self._policy_accepts_temp = False

        if self.score_fn is not None:
            try:
                sig = inspect.signature(self.score_fn)
                self._score_accepts_temp = len(sig.parameters) >= 3
            except (ValueError, TypeError):
                self._score_accepts_temp = False
        else:
            self._score_accepts_temp = False

    def collect(
        self,
        episodes: int = 1,
        *,
        temperature: float = 1.0,
        seed: int | None = None,
        compute_ref_log_probs: bool = False,
    ) -> list[CollectedTrajectory]:
        """Collect trajectories by running episodes.

        Args:
            episodes: Number of episodes
            temperature: Sampling temperature
            seed: Optional seed
            compute_ref_log_probs: Compute reference policy log-probs

        Returns:
            List of CollectedTrajectory
        """
        trajectories: list[CollectedTrajectory] = []

        if compute_ref_log_probs:
            if self.score_fn is None:
                raise ValueError("compute_ref_log_probs requires score_fn")
            if self._score_fn_is_default:
                raise ValueError("compute_ref_log_probs requires explicit ref policy score_fn")

        for _ in range(episodes):
            chat_history, _ = self.env.init()

            obs_list: list[str] = []
            acts: list[dict[str, Any]] = []
            logps: list[torch.Tensor] = []
            ref_logps: list[torch.Tensor] = []
            ents: list[torch.Tensor] = []
            eos_logps: list[torch.Tensor] = []
            done = False
            total_reward = 0.0

            obs = self._extract_observation(chat_history)

            while not done:
                if self._policy_accepts_temp:
                    action, logp, ent = cast(PolicyFnWithTemp, self.policy_fn)(obs, temperature)
                else:
                    action, logp, ent = cast(PolicyFnLegacy, self.policy_fn)(obs)

                eos_logp = getattr(self.policy_fn, "_last_eos_logprob_final", None)
                if eos_logp is not None:
                    eos_logps.append(self._to_tensor(eos_logp))

                if compute_ref_log_probs:
                    with torch.no_grad():
                        if self._score_accepts_temp:
                            ref_logp, _ = cast(ScoreFnWithTemp, self.score_fn)(
                                obs, action, temperature
                            )
                        else:
                            ref_logp, _ = cast(ScoreFnLegacy, self.score_fn)(obs, action)
                    ref_logps.append(self._to_tensor(ref_logp).detach())

                action_text = self._render_action(action)
                result = self.env.step(action_text)

                obs_list.append(obs)
                acts.append(action)
                logps.append(self._to_tensor(logp))
                ents.append(self._to_tensor(ent))

                total_reward += result["reward"]
                done = result["done"]

                if not done:
                    chat_hist = getattr(self.env, "chat_history", None)
                    if chat_hist:
                        obs = self._extract_observation(chat_hist)
                    elif result["observations"]:
                        obs = self._extract_observation(result["observations"])

            eos_logprobs_t: torch.Tensor | None = None
            if eos_logps and len(eos_logps) == len(logps):
                eos_logprobs_t = torch.stack(eos_logps)

            trajectories.append(
                CollectedTrajectory(
                    observations=obs_list,
                    actions=acts,
                    log_probs=torch.stack(logps) if logps else torch.zeros(0),
                    ref_log_probs=torch.stack(ref_logps) if compute_ref_log_probs else None,
                    entropy=torch.stack(ents) if ents else torch.zeros(0),
                    reward=total_reward,
                    temperature=temperature,
                    eos_logprobs=eos_logprobs_t,
                )
            )

        return trajectories

    def replay_entry(
        self,
        entry: Any,
        temperature: float = 1.0,
    ) -> CollectedTrajectory | None:
        """Replay a buffer entry with current policy scoring.

        Used by GFNReplayBuffer for on-sample re-scoring.
        Also captures EOS logprobs to match fresh trajectories from collect().
        """
        if self.score_fn is None:
            raise ValueError("replay_entry requires score_fn")

        if len(entry.actions) == 0 or len(entry.observations) == 0:
            return None

        n_actions = len(entry.actions)
        observations = entry.observations[:n_actions]
        if len(observations) < n_actions:
            observations = observations + [observations[-1]] * (n_actions - len(observations))

        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        eos_logps: list[torch.Tensor] = []

        # resolve EOS source once: for bound methods (policy.score_action),
        # the attribute lives on the instance, not the method wrapper.
        # NOTE: _last_eos_logprob_final is a side-channel — not thread-safe.
        # Safe for single-threaded LocalRunner; would need explicit return
        # from score_action if parallelized.
        eos_source = getattr(self.score_fn, "__self__", self.score_fn)

        for obs, action in zip(observations, entry.actions, strict=True):
            if self._score_accepts_temp:
                logp, ent = cast(ScoreFnWithTemp, self.score_fn)(obs, action, temperature)
            else:
                logp, ent = cast(ScoreFnLegacy, self.score_fn)(obs, action)
            log_probs.append(self._to_tensor(logp))
            entropies.append(self._to_tensor(ent))

            eos_logp = getattr(eos_source, "_last_eos_logprob_final", None)
            if eos_logp is not None:
                eos_logps.append(self._to_tensor(eos_logp))

        # clamp to prevent overflow (log_reward can reach ±700)
        reward = math.exp(min(entry.log_reward, 700.0))

        # only include eos_logprobs if we got one for every action
        eos_logprobs_t: torch.Tensor | None = None
        if eos_logps and len(eos_logps) == len(log_probs):
            eos_logprobs_t = torch.stack(eos_logps)

        return CollectedTrajectory(
            observations=observations,
            actions=list(entry.actions),
            log_probs=torch.stack(log_probs) if log_probs else torch.zeros(0),
            entropy=torch.stack(entropies) if entropies else torch.zeros(0),
            reward=reward,
            temperature=temperature,
            eos_logprobs=eos_logprobs_t,
        )

    @staticmethod
    def _extract_observation(messages: list[dict[str, str]]) -> str:
        """Serialize messages as JSON."""
        import json

        return json.dumps(messages)

    def _render_action(self, action: dict[str, Any] | Action | str) -> str:
        """Render action to text."""
        if isinstance(action, str):
            return action
        if isinstance(action, Action):
            return self._render_with_codec(action)
        if isinstance(action, dict):
            parsed = self._dict_to_action(action)
            if parsed is not None:
                return self._render_with_codec(parsed)
        return str(action)

    def _render_with_codec(self, action: Action) -> str:
        """Render using env's codec."""
        codec = getattr(self.env, "codec", None)
        if codec is None:
            return str(action)
        if hasattr(codec, "render"):
            return codec.render(action)
        if hasattr(codec, "format"):
            return codec.format(action)
        return str(action)

    @staticmethod
    def _dict_to_action(action: dict[str, Any]) -> Action | None:
        """Convert dict to Action."""
        if "query" in action and len(action) <= 2 and "name" not in action:
            return ToolCall(name="query", input={"query": str(action["query"])}, raw="")
        if "tool" in action or "name" in action:
            name = action.get("tool") or action.get("name")
            raw_input = action.get("input", action.get("args", {}))
            input_payload: dict[str, Any]
            if isinstance(raw_input, dict):
                input_payload = raw_input
            elif isinstance(raw_input, str):
                input_payload = {"query": raw_input} if name == "query" else {"value": raw_input}
            else:
                input_payload = {}
            return ToolCall(name=str(name), input=input_payload, raw="")
        if "answer" in action:
            return FinalAnswer(text=str(action["answer"]))
        if "program" in action or "code" in action:
            code = action.get("program", action.get("code", ""))
            language = action.get("language", "pymc")
            return Program(code=str(code), language=str(language))
        if "type" in action:
            action_type = str(action.get("type"))
            payload = action.get("payload", "")
            if action_type in {"answer", "final", "final_answer"}:
                return FinalAnswer(text=str(payload))
            if action_type in {"program", "submit_program"}:
                return Program(code=str(payload), language=action.get("language", "pymc"))
            if isinstance(payload, dict):
                input_payload = payload
            elif isinstance(payload, str) and action_type == "query":
                input_payload = {"query": payload}
            elif isinstance(payload, str) and payload:
                input_payload = {"value": payload}
            else:
                input_payload = {}
            return ToolCall(name=action_type, input=input_payload, raw="")
        return None

    @staticmethod
    def _to_tensor(val: float | torch.Tensor) -> torch.Tensor:
        """Convert to scalar tensor."""
        if isinstance(val, torch.Tensor):
            if val.dim() != 0:
                val = val.reshape(())
            return val
        return torch.tensor(float(val), dtype=torch.float32)
