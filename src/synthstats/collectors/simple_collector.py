"""Simple on-policy collector for SynthStats with text observations.

This is not optimized; it runs env steps sequentially and stores log-probs/entropy
from a provided policy callable.

Policy signatures (both supported):
- Legacy: policy_fn(obs_text) -> (action_dict, log_prob, entropy)
- New:    policy_fn(obs_text, temperature) -> (action_dict, log_prob, entropy)
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch.nn.utils.rnn import pad_sequence

from synthstats.core.types import Action, FinalAnswer, Program, ToolCall
from synthstats.envs.skyrl_text_env import SynthStatsTextEnv
from synthstats.training.buffers import BufferEntry

# type aliases for policy functions
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
    """A single collected trajectory with policy data."""

    observations: list[str]
    actions: list[dict[str, Any]]
    # per-step values, shape: [T]
    log_probs: torch.Tensor
    entropy: torch.Tensor
    reward: float
    # optional fields
    ref_log_probs: torch.Tensor | None = None
    temperature: float = 1.0  # temperature used during collection
    eos_logprobs: torch.Tensor | None = None  # [T] log P(EOS|state) for SubTB
    # Tinker-specific: raw text for API-based training
    prompts: list[str] | None = None  # full prompts at each step
    completions: list[str] | None = None  # generated completions at each step

    def detach(self) -> CollectedTrajectory:
        """Return a copy with detached tensors moved to CPU.

        Use this before storing in replay buffer to avoid retaining
        computation graphs and GPU memory.
        """
        ref_log_probs = (
            self.ref_log_probs.detach().cpu() if self.ref_log_probs is not None else None
        )
        eos_logprobs = self.eos_logprobs.detach().cpu() if self.eos_logprobs is not None else None
        return CollectedTrajectory(
            observations=self.observations,
            actions=self.actions,
            log_probs=self.log_probs.detach().cpu(),
            ref_log_probs=ref_log_probs,
            entropy=self.entropy.detach().cpu(),
            reward=self.reward,
            temperature=self.temperature,
            eos_logprobs=eos_logprobs,
            prompts=self.prompts,
            completions=self.completions,
        )


class SimpleCollector:
    """Simple sequential collector for SynthStats environments.

    Collects trajectories by running episodes with the provided policy function.
    Supports both legacy (obs-only) and new (obs, temperature) policy signatures.

    Args:
        env: SynthStatsTextEnv instance
        policy_fn: Policy callable with one of two signatures:
                   - Legacy: (obs_text) -> (action_dict, log_prob, entropy)
                   - New: (obs_text, temperature) -> (action_dict, log_prob, entropy)
        score_fn: Optional scoring callable for reference-policy correction.
                  Must be explicitly provided when compute_ref_log_probs=True.
    """

    def __init__(
        self,
        env: SynthStatsTextEnv,
        policy_fn: PolicyFn,
        score_fn: ScoreFn | None = None,
    ):
        self.env = env
        self.policy_fn = policy_fn

        if score_fn is None and hasattr(policy_fn, "score_action"):
            self.score_fn = policy_fn.score_action
            self._score_fn_is_default = True
        else:
            self.score_fn = score_fn
            self._score_fn_is_default = False

        # detect if policy supports temperature parameter
        try:
            sig = inspect.signature(policy_fn)
            self._policy_accepts_temp = len(sig.parameters) >= 2
        except (ValueError, TypeError):
            self._policy_accepts_temp = False

        # detect if score function supports temperature parameter
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
            episodes: Number of episodes to collect
            temperature: Sampling temperature for policy
            seed: Optional seed for environment reset
            compute_ref_log_probs: If True, compute ref policy log-probs for actions

        Returns:
            List of CollectedTrajectory
        """
        trajectories: list[CollectedTrajectory] = []

        if compute_ref_log_probs:
            if self.score_fn is None:
                raise ValueError(
                    "compute_ref_log_probs=True requires a score_fn (e.g., ref_policy.score_action)"
                )
            if self._score_fn_is_default:
                raise ValueError(
                    "compute_ref_log_probs=True requires an explicit score_fn "
                    "from a reference policy (not the behavior policy)"
                )

        for _ep in range(episodes):
            # reset environment
            chat_history, _ = self.env.init()

            obs_list: list[str] = []
            acts: list[dict[str, Any]] = []
            logps: list[torch.Tensor] = []
            ref_logps: list[torch.Tensor] = []
            ents: list[torch.Tensor] = []
            eos_logps: list[torch.Tensor] = []
            done = False
            total_reward = 0.0

            # extract initial observation from chat history
            obs = self._extract_observation(chat_history)

            while not done:
                # call policy
                if self._policy_accepts_temp:
                    action, logp, ent = cast(PolicyFnWithTemp, self.policy_fn)(obs, temperature)
                else:
                    action, logp, ent = cast(PolicyFnLegacy, self.policy_fn)(obs)

                # grab EOS logprob if policy tracks it
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

                # render action to text
                action_text = self._render_action(action)

                # step environment
                result = self.env.step(action_text)

                obs_list.append(obs)
                acts.append(action)

                # convert to tensors
                logp_t = self._to_tensor(logp)
                ent_t = self._to_tensor(ent)

                logps.append(logp_t)
                ents.append(ent_t)

                total_reward += result["reward"]
                done = result["done"]

                if not done:
                    # use full chat history for proper multi-turn context
                    chat_hist = getattr(self.env, "chat_history", None)
                    if chat_hist:
                        obs = self._extract_observation(chat_hist)
                    elif result["observations"]:
                        obs = self._extract_observation(result["observations"])

            # stack EOS logprobs if we got the full set
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
                    ref_log_probs=(torch.stack(ref_logps) if compute_ref_log_probs else None),
                    entropy=torch.stack(ents) if ents else torch.zeros(0),
                    reward=total_reward,
                    temperature=temperature,
                    eos_logprobs=eos_logprobs_t,
                )
            )

        return trajectories

    def replay_entry(
        self,
        entry: BufferEntry,
        temperature: float = 1.0,
    ) -> CollectedTrajectory | None:
        """Replay a BufferEntry and re-score with current policy.

        Re-computes log_probs for the stored action sequence using the
        current score_fn. This eliminates off-policy bias from stale
        log_probs stored at collection time.

        Note: eos_logprobs will be None for replayed trajectories (score_fn
        can't compute EOS probs). Use vanilla TB with replay, not modified_subtb.

        Args:
            entry: BufferEntry with actions and observations
            temperature: Sampling temperature for scoring

        Returns:
            CollectedTrajectory with fresh log_probs, or None on failure.

        Raises:
            ValueError: If score_fn is not available
        """
        if self.score_fn is None:
            raise ValueError("replay_entry requires a score_fn")

        if len(entry.actions) == 0 or len(entry.observations) == 0:
            return None

        # ensure we have matching observations for actions
        # (each action needs corresponding observation for scoring)
        n_actions = len(entry.actions)
        observations = entry.observations[:n_actions]

        if len(observations) < n_actions:
            # pad with last observation if needed
            observations = observations + [observations[-1]] * (n_actions - len(observations))

        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []

        # NOTE: No torch.no_grad() - we NEED gradients for policy learning!
        # The whole point of re-scoring is to get fresh, differentiable log_probs
        for obs, action in zip(observations, entry.actions, strict=True):
            if self._score_accepts_temp:
                logp, ent = cast(ScoreFnWithTemp, self.score_fn)(obs, action, temperature)
            else:
                logp, ent = cast(ScoreFnLegacy, self.score_fn)(obs, action)

            # Keep gradients attached for policy learning
            log_probs.append(self._to_tensor(logp))
            entropies.append(self._to_tensor(ent))

        # convert log_reward back to reward
        reward = math.exp(entry.log_reward)

        return CollectedTrajectory(
            observations=observations,
            actions=list(entry.actions),
            log_probs=torch.stack(log_probs) if log_probs else torch.zeros(0),
            entropy=torch.stack(entropies) if entropies else torch.zeros(0),
            reward=reward,
            temperature=temperature,
        )

    @staticmethod
    def _extract_observation(messages: list[dict[str, str]]) -> str:
        """Serialize full message history as JSON for chat-template-aware policies."""
        import json

        return json.dumps(messages)

    def _render_action(self, action: dict[str, Any] | Action | str) -> str:
        """Render action into the codec-compatible text format."""
        if isinstance(action, str):
            return action

        if isinstance(action, Action):
            return self._render_with_codec(action)

        if isinstance(action, dict):
            parsed = self._dict_to_action(action)
            if parsed is not None:
                return self._render_with_codec(parsed)

        # fallback: best-effort string conversion
        return str(action)

    def _render_with_codec(self, action: Action) -> str:
        """Render an Action using the env's codec (render/format)."""
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
        """Convert a dict action into a core Action when possible."""
        # bare query dict: {"query": "..."} — common when model skips XML wrapper
        if "query" in action and len(action) <= 2 and "name" not in action and "tool" not in action:
            return ToolCall(name="query", input={"query": str(action["query"])}, raw="")

        # JSONToolCodec-style
        if "tool" in action or "name" in action:
            name = action.get("tool") or action.get("name")
            raw_input = action.get("input", action.get("args", {}))
            input_payload: dict[str, Any]
            if isinstance(raw_input, dict):
                input_payload = raw_input
            elif isinstance(raw_input, str):
                if name == "query":
                    input_payload = {"query": raw_input}
                else:
                    input_payload = {"value": raw_input}
            else:
                input_payload = {}
            return ToolCall(name=str(name), input=input_payload, raw="")

        if "answer" in action:
            return FinalAnswer(text=str(action["answer"]))

        if "program" in action or "code" in action:
            code = action.get("program", action.get("code", ""))
            language = action.get("language", "pymc")
            return Program(code=str(code), language=str(language))

        # legacy format: {"type": ..., "payload": ...}
        if "type" in action:
            action_type = str(action.get("type"))
            payload = action.get("payload", "")
            if action_type in {"answer", "final", "final_answer"}:
                return FinalAnswer(text=str(payload))
            if action_type in {"program", "submit_program"}:
                language = action.get("language", "pymc")
                return Program(code=str(payload), language=str(language))
            # treat as tool call
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
        """Convert value to scalar tensor."""
        if isinstance(val, torch.Tensor):
            if val.dim() != 0:
                val = val.reshape(())
            return val
        return torch.tensor(float(val), dtype=torch.float32)


def build_subtb_batch(
    trajectories: list[CollectedTrajectory],
    *,
    reward_floor: float = 1e-4,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Build a padded batch suitable for SubTB/TB training.

    Variable-length episodes are padded to T_max and accompanied by a boolean mask.

    Args:
        trajectories: List of CollectedTrajectory
        reward_floor: Minimum reward value (to avoid log(0))
        device: Target device for tensors

    Returns:
        dict with:
          - log_probs: [B, T_max]
          - loss_mask: [B, T_max] (bool) where True indicates a real step
          - log_reward: [B] (no grad; terminal reward is treated as constant)
          - entropy: [B, T_max] (float), padded with 0.0
          - eos_logprobs: [B, T_max] (optional)

    Raises:
        ValueError: If trajectories is empty or has invalid data
    """
    if not trajectories:
        raise ValueError("trajectories must be non-empty")

    device_t = torch.device(device) if not isinstance(device, torch.device) else device

    if reward_floor <= 0:
        raise ValueError(f"reward_floor must be > 0, got {reward_floor}")

    log_prob_seqs: list[torch.Tensor] = []
    ent_seqs: list[torch.Tensor] = []
    eos_logprob_seqs: list[torch.Tensor] | None = None
    ref_log_prob_seqs: list[torch.Tensor] | None = None

    has_ref = [t.ref_log_probs is not None for t in trajectories]
    if any(has_ref) and not all(has_ref):
        raise ValueError(
            "mixed ref_log_probs: either provide ref_log_probs for all trajectories or none"
        )
    if all(has_ref):
        ref_log_prob_seqs = []

    has_eos = [t.eos_logprobs is not None for t in trajectories]
    if any(has_eos) and not all(has_eos):
        raise ValueError(
            "mixed eos_logprobs: either provide eos_logprobs for all trajectories or none"
        )
    if all(has_eos):
        eos_logprob_seqs = []

    for i, t in enumerate(trajectories):
        if not isinstance(t.log_probs, torch.Tensor):
            raise ValueError(
                f"trajectory[{i}].log_probs must be a torch.Tensor, "
                f"got {type(t.log_probs).__name__}"
            )
        if not isinstance(t.entropy, torch.Tensor):
            raise ValueError(
                f"trajectory[{i}].entropy must be a torch.Tensor, got {type(t.entropy).__name__}"
            )
        if t.log_probs.dim() != 1:
            raise ValueError(
                f"trajectory[{i}].log_probs must be 1D [T], got shape {tuple(t.log_probs.shape)}"
            )
        if t.entropy.dim() != 1:
            raise ValueError(
                f"trajectory[{i}].entropy must be 1D [T], got shape {tuple(t.entropy.shape)}"
            )
        if t.log_probs.numel() == 0:
            raise ValueError(
                f"trajectory[{i}] has empty log_probs (T=0); episodes must have >=1 step"
            )
        if t.entropy.numel() != t.log_probs.numel():
            raise ValueError(
                f"trajectory[{i}] length mismatch: log_probs has T={t.log_probs.numel()} "
                f"but entropy has T={t.entropy.numel()}"
            )

        log_prob_seqs.append(t.log_probs.to(device_t))
        ent_seqs.append(t.entropy.to(device_t))
        if ref_log_prob_seqs is not None:
            ref = t.ref_log_probs
            if ref is None:
                raise ValueError(f"trajectory[{i}] missing ref_log_probs while others provide it")
            if not isinstance(ref, torch.Tensor):
                raise ValueError(
                    f"trajectory[{i}].ref_log_probs must be a torch.Tensor, "
                    f"got {type(ref).__name__}"
                )
            if ref.dim() != 1:
                raise ValueError(
                    f"trajectory[{i}].ref_log_probs must be 1D [T], got shape {tuple(ref.shape)}"
                )
            if ref.numel() != t.log_probs.numel():
                raise ValueError(
                    f"trajectory[{i}] length mismatch: ref_log_probs has T={ref.numel()} "
                    f"but log_probs has T={t.log_probs.numel()}"
                )
            ref_log_prob_seqs.append(ref.to(device_t))

        if eos_logprob_seqs is not None:
            eos = t.eos_logprobs
            if eos is None:
                raise ValueError(f"trajectory[{i}] missing eos_logprobs while others provide it")
            if not isinstance(eos, torch.Tensor):
                raise ValueError(
                    f"trajectory[{i}].eos_logprobs must be a torch.Tensor, got {type(eos).__name__}"
                )
            if eos.dim() != 1:
                raise ValueError(
                    f"trajectory[{i}].eos_logprobs must be 1D [T], got shape {tuple(eos.shape)}"
                )
            if eos.numel() != t.log_probs.numel():
                raise ValueError(
                    f"trajectory[{i}] length mismatch: eos_logprobs has T={eos.numel()} "
                    f"but log_probs has T={t.log_probs.numel()}"
                )
            eos_logprob_seqs.append(eos.to(device_t))

    # pad sequences
    log_probs = pad_sequence(log_prob_seqs, batch_first=True, padding_value=0.0)
    entropy = pad_sequence(ent_seqs, batch_first=True, padding_value=0.0)
    ref_log_probs = None
    if ref_log_prob_seqs is not None:
        ref_log_probs = pad_sequence(ref_log_prob_seqs, batch_first=True, padding_value=0.0)
    eos_logprobs = None
    if eos_logprob_seqs is not None:
        eos_logprobs = pad_sequence(eos_logprob_seqs, batch_first=True, padding_value=0.0)

    # create mask
    mask = pad_sequence(
        [torch.ones_like(lp, dtype=torch.bool, device=device_t) for lp in log_prob_seqs],
        batch_first=True,
        padding_value=False,
    )

    # verify shapes
    if log_probs.shape != entropy.shape or log_probs.shape != mask.shape:
        raise RuntimeError(
            f"internal error: padded shapes diverged: log_probs={tuple(log_probs.shape)} "
            f"entropy={tuple(entropy.shape)} mask={tuple(mask.shape)}"
        )

    # compute log rewards (detached, with floor)
    rewards = torch.tensor(
        [max(float(t.reward), float(reward_floor)) for t in trajectories],
        device=device_t,
    )
    log_reward = torch.log(rewards).detach()

    result: dict[str, torch.Tensor] = {
        "log_probs": log_probs,
        "loss_mask": mask,
        "log_reward": log_reward,
        "entropy": entropy,
    }
    if ref_log_probs is not None:
        result["ref_log_probs"] = ref_log_probs
    if eos_logprobs is not None:
        result["eos_logprobs"] = eos_logprobs
    return result


def build_tinker_batch(
    trajectories: list[CollectedTrajectory],
    *,
    reward_floor: float = 1e-4,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Build batch for TinkerTrainer - preserves strings instead of tokenizing."""
    if not trajectories:
        raise ValueError("trajectories must be non-empty")

    device_t = torch.device(device) if not isinstance(device, torch.device) else device

    if reward_floor <= 0:
        raise ValueError(f"reward_floor must be > 0, got {reward_floor}")

    prompts: list[str] = []
    completions: list[str] = []

    for t in trajectories:
        if t.prompts is None or t.completions is None:
            prompt = _reconstruct_prompt(t.observations)
            completion = _reconstruct_completion(t.actions)
        else:
            prompt = "\n".join(t.prompts)
            completion = "\n".join(t.completions)

        prompts.append(prompt)
        completions.append(completion)

    rewards = torch.tensor(
        [max(float(t.reward), float(reward_floor)) for t in trajectories],
        device=device_t,
    )
    log_reward = torch.log(rewards).detach()

    result: dict[str, Any] = {
        "prompts": prompts,
        "completions": completions,
        "log_reward": log_reward,
    }

    # include mask if log_probs available
    if all(t.log_probs is not None and t.log_probs.numel() > 0 for t in trajectories):
        log_prob_seqs = [t.log_probs.to(device_t) for t in trajectories]
        mask = pad_sequence(
            [torch.ones_like(lp, dtype=torch.bool, device=device_t) for lp in log_prob_seqs],
            batch_first=True,
            padding_value=False,
        )
        result["loss_mask"] = mask

    return result


def _reconstruct_prompt(observations: list[str]) -> str:
    """Reconstruct prompt from observations when prompts field not available."""
    if not observations:
        return ""
    # join observations with newlines
    return "\n".join(observations)


def _reconstruct_completion(actions: list[dict[str, Any]]) -> str:
    """Reconstruct completion from actions when completions field not available."""
    import json

    if not actions:
        return ""
    # render actions as JSON strings
    return "\n".join(json.dumps(a) for a in actions)
