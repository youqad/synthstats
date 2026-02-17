"""Batch building utilities for training.

Converts lists of trajectories into batched tensors suitable for
SubTB/TB training.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from synthstats.core.constants import REWARD_FLOOR_DEFAULT
from synthstats.train.loop.collectors import CollectedTrajectory
from synthstats.train.utils.device import normalize_device


def extract_reward(obj: Any) -> float:
    """Extract scalar reward from a Trajectory, Reward, or float."""
    r = getattr(obj, "reward", obj)
    if hasattr(r, "total"):
        return float(r.total)
    return float(r)


def _validate_batch_inputs(
    trajectories: list[CollectedTrajectory],
    reward_floor: float,
    device: str | torch.device,
) -> torch.device:
    """Shared validation for batch builders.

    Returns:
        Normalized torch.device
    """
    if not trajectories:
        raise ValueError("trajectories must be non-empty")
    if reward_floor <= 0:
        raise ValueError(f"reward_floor must be > 0, got {reward_floor}")
    return normalize_device(device)


def _compute_log_rewards(
    trajectories: list[CollectedTrajectory],
    reward_floor: float,
    device: torch.device,
) -> torch.Tensor:
    """Compute floored log rewards from trajectories.

    Returns:
        log_reward tensor [B] on device
    """
    rewards = torch.tensor(
        [max(extract_reward(t), float(reward_floor)) for t in trajectories],
        device=device,
    )
    return torch.log(rewards).detach()


def _all_or_none_field(
    trajectories: list[CollectedTrajectory],
    *,
    field: str,
    label: str,
) -> bool:
    has_field = [getattr(t, field) is not None for t in trajectories]
    if any(has_field) and not all(has_field):
        raise ValueError(f"mixed {label}: provide for all or none")
    return all(has_field)


def _validate_base_tensors(t: CollectedTrajectory, index: int) -> None:
    if not isinstance(t.log_probs, torch.Tensor):
        raise ValueError(f"trajectory[{index}].log_probs must be torch.Tensor")
    if not isinstance(t.entropy, torch.Tensor):
        raise ValueError(f"trajectory[{index}].entropy must be torch.Tensor")
    if t.log_probs.dim() != 1:
        raise ValueError(f"trajectory[{index}].log_probs must be 1D [T]")
    if t.entropy.dim() != 1:
        raise ValueError(f"trajectory[{index}].entropy must be 1D [T]")
    if t.log_probs.numel() == 0:
        raise ValueError(f"trajectory[{index}] has empty log_probs")
    if t.entropy.numel() != t.log_probs.numel():
        raise ValueError(f"trajectory[{index}] length mismatch: log_probs vs entropy")


def _require_optional_seq(
    value: torch.Tensor | None,
    *,
    index: int,
    label: str,
    expected_len: int,
    device: torch.device,
) -> torch.Tensor:
    if value is None:
        raise ValueError(f"trajectory[{index}] missing {label}")
    if value.dim() != 1 or value.numel() != expected_len:
        raise ValueError(f"trajectory[{index}].{label} shape mismatch")
    return value.to(device)


def _collect_subtb_sequences(
    trajectories: list[CollectedTrajectory],
    *,
    device: torch.device,
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor] | None,
    list[torch.Tensor] | None,
]:
    include_ref = _all_or_none_field(trajectories, field="ref_log_probs", label="ref_log_probs")
    include_eos = _all_or_none_field(trajectories, field="eos_logprobs", label="eos_logprobs")

    log_prob_seqs: list[torch.Tensor] = []
    ent_seqs: list[torch.Tensor] = []
    ref_log_prob_seqs: list[torch.Tensor] | None = [] if include_ref else None
    eos_logprob_seqs: list[torch.Tensor] | None = [] if include_eos else None

    for i, t in enumerate(trajectories):
        _validate_base_tensors(t, i)
        log_prob = t.log_probs.to(device)
        entropy = t.entropy.to(device)
        log_prob_seqs.append(log_prob)
        ent_seqs.append(entropy)

        if ref_log_prob_seqs is not None:
            ref_log_prob_seqs.append(
                _require_optional_seq(
                    t.ref_log_probs,
                    index=i,
                    label="ref_log_probs",
                    expected_len=log_prob.numel(),
                    device=device,
                )
            )

        if eos_logprob_seqs is not None:
            eos_logprob_seqs.append(
                _require_optional_seq(
                    t.eos_logprobs,
                    index=i,
                    label="eos_logprobs",
                    expected_len=log_prob.numel(),
                    device=device,
                )
            )

    return log_prob_seqs, ent_seqs, ref_log_prob_seqs, eos_logprob_seqs


def _pad_optional(
    sequences: list[torch.Tensor] | None,
    *,
    padding_value: float = 0.0,
) -> torch.Tensor | None:
    if sequences is None:
        return None
    return pad_sequence(sequences, batch_first=True, padding_value=padding_value)


def _build_loss_mask(
    log_prob_seqs: list[torch.Tensor],
    *,
    device: torch.device,
) -> torch.Tensor:
    return pad_sequence(
        [torch.ones_like(lp, dtype=torch.bool, device=device) for lp in log_prob_seqs],
        batch_first=True,
        padding_value=False,
    )


def build_subtb_batch(
    trajectories: list[CollectedTrajectory],
    *,
    reward_floor: float = REWARD_FLOOR_DEFAULT,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Build padded batch for SubTB/TB training.

    Args:
        trajectories: List of CollectedTrajectory
        reward_floor: Minimum reward (avoid log(0))
        device: Target device

    Returns:
        Dict with:
          - log_probs: [B, T_max]
          - loss_mask: [B, T_max] (bool)
          - log_reward: [B]
          - entropy: [B, T_max]
          - ref_log_probs: optional [B, T_max]
          - eos_logprobs: optional [B, T_max]
    """
    device_t = _validate_batch_inputs(trajectories, reward_floor, device)

    log_prob_seqs, ent_seqs, ref_log_prob_seqs, eos_logprob_seqs = _collect_subtb_sequences(
        trajectories,
        device=device_t,
    )

    log_probs = pad_sequence(log_prob_seqs, batch_first=True, padding_value=0.0)
    entropy = pad_sequence(ent_seqs, batch_first=True, padding_value=0.0)
    ref_log_probs = _pad_optional(ref_log_prob_seqs)
    eos_logprobs = _pad_optional(eos_logprob_seqs)
    mask = _build_loss_mask(log_prob_seqs, device=device_t)

    log_reward = _compute_log_rewards(trajectories, reward_floor, device_t)

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
    reward_floor: float = REWARD_FLOOR_DEFAULT,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Build batch for Tinker training (string-based).

    Args:
        trajectories: List of CollectedTrajectory
        reward_floor: Minimum reward
        device: Target device

    Returns:
        Dict with prompts, completions, log_reward
    """
    device_t = _validate_batch_inputs(trajectories, reward_floor, device)

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

    log_reward = _compute_log_rewards(trajectories, reward_floor, device_t)

    result: dict[str, Any] = {
        "prompts": prompts,
        "completions": completions,
        "log_reward": log_reward,
    }

    return result


def _reconstruct_prompt(observations: list[str]) -> str:
    if not observations:
        return ""
    return "\n".join(observations)


def _reconstruct_completion(actions: list[dict[str, Any]]) -> str:
    import json

    if not actions:
        return ""
    return "\n".join(json.dumps(a) for a in actions)
