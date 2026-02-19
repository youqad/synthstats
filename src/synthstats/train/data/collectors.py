"""Trajectory collectors for training."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import torch

from synthstats.core.types import Action, FinalAnswer, Program, ToolCall

PolicyOut3 = tuple[dict[str, Any], float | torch.Tensor, float | torch.Tensor]
PolicyOut4 = tuple[dict[str, Any], float | torch.Tensor, float | torch.Tensor, float | torch.Tensor | None]
PolicyFnLegacy = Callable[[str], PolicyOut3 | PolicyOut4]
PolicyFnWithTemp = Callable[[str, float], PolicyOut3 | PolicyOut4]
PolicyFn = PolicyFnLegacy | PolicyFnWithTemp
ScoreOut2 = tuple[float | torch.Tensor, float | torch.Tensor]
ScoreOut3 = tuple[float | torch.Tensor, float | torch.Tensor, float | torch.Tensor | None]
ScoreFnLegacy = Callable[[str, dict[str, Any]], ScoreOut2 | ScoreOut3]
ScoreFnWithTemp = Callable[[str, dict[str, Any], float], ScoreOut2 | ScoreOut3]
ScoreFn = ScoreFnLegacy | ScoreFnWithTemp


@dataclass
class CollectedTrajectory:
    """Trajectory with log_probs, entropy, and optional ref/EOS."""

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
    """Runs episodes and collects trajectories with policy log-probs."""

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
        compute_ref_log_probs: bool = False,
    ) -> list[CollectedTrajectory]:
        """Run `episodes` episodes, optionally scoring against a ref policy."""
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
                action, logp, ent, eos_logp = self._sample_action(obs, temperature)
                if eos_logp is not None:
                    eos_logps.append(self._to_tensor(eos_logp))

                if compute_ref_log_probs:
                    with torch.no_grad():
                        ref_logp, _, _ = self._score_action(obs, action, temperature)
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
            if eos_logps:
                if len(eos_logps) == len(logps):
                    eos_logprobs_t = torch.stack(eos_logps)
                else:
                    import warnings

                    warnings.warn(
                        f"partial EOS logprobs ({len(eos_logps)}/{len(logps)}), discarding",
                        UserWarning,
                        stacklevel=2,
                    )

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
        """Re-score a buffer entry with current policy for on-policy gradients."""
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

        for obs, action in zip(observations, entry.actions, strict=True):
            logp, ent, eos_logp = self._score_action(obs, action, temperature)
            log_probs.append(self._to_tensor(logp))
            entropies.append(self._to_tensor(ent))

            if eos_logp is not None:
                eos_logps.append(self._to_tensor(eos_logp))

        # log→exp→log round-trip: BufferEntry stores log_reward from collection time,
        # but CollectedTrajectory.reward is a raw float that build_subtb_batch will
        # re-log-transform. Minor floating-point drift is acceptable.
        clamped = max(min(entry.log_reward, 700.0), -700.0)
        reward = math.exp(clamped)

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
        import json

        return json.dumps(messages)

    def _render_action(self, action: dict[str, Any] | Action | str) -> str:
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
        codec = getattr(self.env, "codec", None)
        if codec is None:
            return str(action)
        if hasattr(codec, "render"):
            return codec.render(action)
        return str(action)

    @staticmethod
    def _dict_to_action(action: dict[str, Any]) -> Action | None:
        if "query" in action and len(action) <= 2 and "name" not in action:
            return ToolCall(name="query", input={"query": str(action["query"])}, raw="")
        if "tool" in action or "name" in action:
            name = action.get("tool") or action.get("name")
            raw_input = action.get("input", action.get("args", {}))
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
        if isinstance(val, torch.Tensor):
            if val.dim() != 0:
                val = val.reshape(())
            return val
        return torch.tensor(float(val), dtype=torch.float32)

    def _sample_action(
        self,
        obs: str,
        temperature: float,
    ) -> tuple[
        dict[str, Any], float | torch.Tensor, float | torch.Tensor, float | torch.Tensor | None
    ]:
        sample_with_eos = getattr(self.policy_fn, "sample_with_eos", None)
        if callable(sample_with_eos):
            try:
                out = sample_with_eos(obs, temperature)
            except TypeError:
                out = sample_with_eos(obs)
            return self._unpack_policy_output(out)

        if self._policy_accepts_temp:
            out = cast(PolicyFnWithTemp, self.policy_fn)(obs, temperature)
        else:
            out = cast(PolicyFnLegacy, self.policy_fn)(obs)
        return self._unpack_policy_output(out)

    def _score_action(
        self,
        obs: str,
        action: dict[str, Any],
        temperature: float,
    ) -> tuple[float | torch.Tensor, float | torch.Tensor, float | torch.Tensor | None]:
        if self.score_fn is None:
            raise ValueError("_score_action requires score_fn")

        score_with_eos = getattr(self.score_fn, "score_action_with_eos", None)
        if callable(score_with_eos):
            try:
                out = score_with_eos(obs, action, temperature)
            except TypeError:
                out = score_with_eos(obs, action)
            return self._unpack_score_output(out)

        score_owner = getattr(self.score_fn, "__self__", None)
        owner_score_with_eos = getattr(score_owner, "score_action_with_eos", None)
        if callable(owner_score_with_eos):
            try:
                out = owner_score_with_eos(obs, action, temperature)
            except TypeError:
                out = owner_score_with_eos(obs, action)
            return self._unpack_score_output(out)

        if self._score_accepts_temp:
            out = cast(ScoreFnWithTemp, self.score_fn)(obs, action, temperature)
        else:
            out = cast(ScoreFnLegacy, self.score_fn)(obs, action)
        return self._unpack_score_output(out)

    @staticmethod
    def _unpack_policy_output(
        out: Any,
    ) -> tuple[
        dict[str, Any], float | torch.Tensor, float | torch.Tensor, float | torch.Tensor | None
    ]:
        if not isinstance(out, tuple):
            raise ValueError("policy_fn must return tuple(action, logp, entropy[, eos_logprob])")
        if len(out) == 3:
            action, logp, ent = out
            return cast(dict[str, Any], action), logp, ent, None
        if len(out) == 4:
            action, logp, ent, eos_logp = out
            return cast(dict[str, Any], action), logp, ent, eos_logp
        raise ValueError(
            f"policy_fn returned unsupported tuple length {len(out)}; expected 3 or 4 elements"
        )

    @staticmethod
    def _unpack_score_output(
        out: Any,
    ) -> tuple[float | torch.Tensor, float | torch.Tensor, float | torch.Tensor | None]:
        if not isinstance(out, tuple):
            raise ValueError("score_fn must return tuple(logp, entropy[, eos_logprob])")
        if len(out) == 2:
            logp, ent = out
            return logp, ent, None
        if len(out) == 3:
            logp, ent, eos_logp = out
            return logp, ent, eos_logp
        raise ValueError(
            f"score_fn returned unsupported tuple length {len(out)}; expected 2 or 3 elements"
        )
