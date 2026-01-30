"""Batch building utilities for training.

Converts lists of trajectories into batched tensors suitable for
SubTB/TB training.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from synthstats.train.loop.collectors import CollectedTrajectory


def build_subtb_batch(
    trajectories: list[CollectedTrajectory],
    *,
    reward_floor: float = 1e-4,
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
    if not trajectories:
        raise ValueError("trajectories must be non-empty")

    device_t = torch.device(device) if not isinstance(device, torch.device) else device

    if reward_floor <= 0:
        raise ValueError(f"reward_floor must be > 0, got {reward_floor}")

    log_prob_seqs: list[torch.Tensor] = []
    ent_seqs: list[torch.Tensor] = []
    eos_logprob_seqs: list[torch.Tensor] | None = None
    ref_log_prob_seqs: list[torch.Tensor] | None = None

    # check consistency
    has_ref = [t.ref_log_probs is not None for t in trajectories]
    if any(has_ref) and not all(has_ref):
        raise ValueError("mixed ref_log_probs: provide for all or none")
    if all(has_ref):
        ref_log_prob_seqs = []

    has_eos = [t.eos_logprobs is not None for t in trajectories]
    if any(has_eos) and not all(has_eos):
        raise ValueError("mixed eos_logprobs: provide for all or none")
    if all(has_eos):
        eos_logprob_seqs = []

    for i, t in enumerate(trajectories):
        if not isinstance(t.log_probs, torch.Tensor):
            raise ValueError(f"trajectory[{i}].log_probs must be torch.Tensor")
        if not isinstance(t.entropy, torch.Tensor):
            raise ValueError(f"trajectory[{i}].entropy must be torch.Tensor")
        if t.log_probs.dim() != 1:
            raise ValueError(f"trajectory[{i}].log_probs must be 1D [T]")
        if t.entropy.dim() != 1:
            raise ValueError(f"trajectory[{i}].entropy must be 1D [T]")
        if t.log_probs.numel() == 0:
            raise ValueError(f"trajectory[{i}] has empty log_probs")
        if t.entropy.numel() != t.log_probs.numel():
            raise ValueError(f"trajectory[{i}] length mismatch: log_probs vs entropy")

        log_prob_seqs.append(t.log_probs.to(device_t))
        ent_seqs.append(t.entropy.to(device_t))

        if ref_log_prob_seqs is not None:
            ref = t.ref_log_probs
            if ref is None:
                raise ValueError(f"trajectory[{i}] missing ref_log_probs")
            if ref.dim() != 1 or ref.numel() != t.log_probs.numel():
                raise ValueError(f"trajectory[{i}].ref_log_probs shape mismatch")
            ref_log_prob_seqs.append(ref.to(device_t))

        if eos_logprob_seqs is not None:
            eos = t.eos_logprobs
            if eos is None:
                raise ValueError(f"trajectory[{i}] missing eos_logprobs")
            if eos.dim() != 1 or eos.numel() != t.log_probs.numel():
                raise ValueError(f"trajectory[{i}].eos_logprobs shape mismatch")
            eos_logprob_seqs.append(eos.to(device_t))

    # pad
    log_probs = pad_sequence(log_prob_seqs, batch_first=True, padding_value=0.0)
    entropy = pad_sequence(ent_seqs, batch_first=True, padding_value=0.0)

    ref_log_probs = None
    if ref_log_prob_seqs is not None:
        ref_log_probs = pad_sequence(ref_log_prob_seqs, batch_first=True, padding_value=0.0)

    eos_logprobs = None
    if eos_logprob_seqs is not None:
        eos_logprobs = pad_sequence(eos_logprob_seqs, batch_first=True, padding_value=0.0)

    # mask
    mask = pad_sequence(
        [torch.ones_like(lp, dtype=torch.bool, device=device_t) for lp in log_prob_seqs],
        batch_first=True,
        padding_value=False,
    )

    # rewards
    def _get_reward(t: CollectedTrajectory) -> float:
        r = t.reward
        if hasattr(r, "total"):
            return float(r.total)
        return float(r)

    rewards = torch.tensor(
        [max(_get_reward(t), float(reward_floor)) for t in trajectories],
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
    """Build batch for Tinker training (string-based).

    Args:
        trajectories: List of CollectedTrajectory
        reward_floor: Minimum reward
        device: Target device

    Returns:
        Dict with prompts, completions, log_reward
    """
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

    def _get_reward(t: CollectedTrajectory) -> float:
        r = t.reward
        if hasattr(r, "total"):
            return float(r.total)
        return float(r)

    rewards = torch.tensor(
        [max(_get_reward(t), float(reward_floor)) for t in trajectories],
        device=device_t,
    )
    log_reward = torch.log(rewards).detach()

    result: dict[str, Any] = {
        "prompts": prompts,
        "completions": completions,
        "log_reward": log_reward,
    }

    return result


def _reconstruct_prompt(observations: list[str]) -> str:
    """Reconstruct prompt from observations."""
    if not observations:
        return ""
    return "\n".join(observations)


def _reconstruct_completion(actions: list[dict[str, Any]]) -> str:
    """Reconstruct completion from actions."""
    import json

    if not actions:
        return ""
    return "\n".join(json.dumps(a) for a in actions)
