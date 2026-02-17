"""TB and Modified SubTB loss functions.

Canonical implementations shared by:
- train/objectives/subtb.py (SubTBObjective)
- train/objectives/trajectory_balance.py (SkyRL integration)
- integrations/tinker/losses.py (Tinker adapter)

Vanilla TB: L = (logZ + sum(log_pi) - log_R)^2
Modified SubTB: Lambda-weighted sub-trajectory flow matching with EOS potentials
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from synthstats.core.math_utils import sanitize_finite


def subtb_loss(
    log_probs: Tensor,
    loss_mask: Tensor,
    log_rewards: Tensor,
    logZ: nn.Parameter,
    *,
    max_residual: float = 100.0,
    ref_log_probs: Tensor | None = None,
    ref_weight: float = 1.0,
    normalize_by_length: bool = False,
) -> Tensor:
    """Trajectory Balance loss: (logZ + sum(masked_log_probs) - log_R)^2.

    Args:
        log_probs: Per-token log probabilities [B, T]
        loss_mask: Boolean mask, True = include in loss
        log_rewards: Log rewards [B]
        logZ: Learnable log partition function (scalar Parameter)
        max_residual: Clamp magnitude for numerical stability
        ref_log_probs: Reference-policy log-probs [B, T] or [B]
        ref_weight: Weight applied to ref_log_probs before subtraction
        normalize_by_length: Divide masked log-prob sums by trajectory length

    Returns:
        Scalar loss (mean over batch).
        inf/nan log_rewards replaced with -max_residual to preserve
        the "low reward" signal rather than treating failures as neutral.
    """
    mask_f = loss_mask.float()
    masked_logprobs = (log_probs * mask_f).sum(dim=-1)  # [B]

    masked_ref_logprobs = None
    if ref_log_probs is not None:
        if ref_log_probs.dim() == 1:
            masked_ref_logprobs = ref_log_probs
        elif ref_log_probs.dim() == 2:
            if ref_log_probs.shape != log_probs.shape:
                raise ValueError(
                    "ref_log_probs must match log_probs shape when 2D: "
                    f"ref_log_probs={tuple(ref_log_probs.shape)} log_probs={tuple(log_probs.shape)}"
                )
            masked_ref_logprobs = (ref_log_probs * mask_f).sum(dim=-1)
        else:
            raise ValueError(
                f"ref_log_probs must be 1D or 2D, got shape {tuple(ref_log_probs.shape)}"
            )

        if ref_weight != 1.0:
            masked_ref_logprobs = masked_ref_logprobs * ref_weight

    if normalize_by_length:
        lengths = mask_f.sum(dim=-1).clamp_min(1.0)
        masked_logprobs = masked_logprobs / lengths
        if masked_ref_logprobs is not None:
            masked_ref_logprobs = masked_ref_logprobs / lengths

    safe_log_rewards = sanitize_finite(log_rewards, -max_residual)

    if masked_ref_logprobs is not None:
        masked_logprobs = masked_logprobs - masked_ref_logprobs

    residual = logZ + masked_logprobs - safe_log_rewards  # [B]

    residual = residual.clamp(-max_residual, max_residual)

    loss = (residual**2).mean()

    return loss


def compute_modified_subtb_core(
    log_probs: Tensor,
    eos_logprobs: Tensor,
    log_rewards: Tensor,
    mask: Tensor,
    *,
    subtb_lambda: float = 0.9,
    max_residual: float = 100.0,
) -> Tensor:
    """Modified SubTB loss (from gfn-lm-tuning).

    delta[t] = log_pf[t] - eos_logprob[t] + eos_logprob[t+1]
    (last valid position uses log_R instead of eos_logprob[t+1])

    loss = sum_L lambda^(L-1) * (cumsum(delta)[t+L] - cumsum(delta)[t])^2

    Args:
        log_probs: Per-token log probabilities [B, T]
        eos_logprobs: EOS log probabilities (flow potentials) [B, T]
        log_rewards: Log rewards [B] or [B, T]
        mask: Loss mask [B, T]
        subtb_lambda: Decay factor for sub-trajectory lengths
        max_residual: Clamp magnitude for numerical stability

    Returns:
        Scalar loss (weighted mean over valid sub-trajectories)
    """
    B, T = log_probs.shape
    device = log_probs.device
    dtype = log_probs.dtype

    mask_f = mask.float() if mask.dtype != torch.float32 else mask

    if log_rewards.dim() == 1:
        log_rewards_2d = log_rewards.unsqueeze(1).expand(B, T)
    else:
        log_rewards_2d = log_rewards

    safe_log_rewards = sanitize_finite(log_rewards_2d, -max_residual)

    if T == 1:
        traj_reward = safe_log_rewards[:, 0:1]
        delta = log_probs - eos_logprobs + traj_reward
    else:
        eos_next = torch.cat([eos_logprobs[:, 1:], eos_logprobs[:, -1:]], dim=1)
        delta = log_probs - eos_logprobs + eos_next

        valid = mask_f > 0
        idxs = torch.arange(T, device=device, dtype=torch.long)
        last_valid_idx = torch.where(
            valid.any(dim=1),
            (valid.float() * idxs).long().max(dim=1).values,
            torch.zeros(B, device=device, dtype=torch.long),
        )

        traj_reward = safe_log_rewards.gather(1, last_valid_idx.unsqueeze(1)).squeeze(1)
        eos_next_at_last = eos_next.gather(1, last_valid_idx.unsqueeze(1)).squeeze(1)
        correction = traj_reward - eos_next_at_last

        correction_tensor = torch.zeros_like(delta)
        correction_tensor.scatter_(1, last_valid_idx.unsqueeze(1), correction.unsqueeze(1))
        delta = delta + correction_tensor

    delta = torch.where(mask_f > 0, delta, torch.zeros_like(delta))

    zeros = torch.zeros(B, 1, device=device, dtype=dtype)
    delta_cumsum = torch.cat([zeros, delta], dim=1).cumsum(dim=1)
    mask_cumsum = torch.cat([zeros, mask_f], dim=1).cumsum(dim=1)

    has_valid = (mask_f.sum(dim=1) > 0).float().unsqueeze(1)

    batch_loss = torch.tensor(0.0, device=device, dtype=dtype)
    total_weight = torch.tensor(0.0, device=device, dtype=dtype)

    for subtraj_len in range(1, T + 1):
        residual = delta_cumsum[:, subtraj_len:] - delta_cumsum[:, :-subtraj_len]
        residual = residual.clamp(-max_residual, max_residual)

        mask_sum = mask_cumsum[:, subtraj_len:] - mask_cumsum[:, :-subtraj_len]
        valid_mask = (mask_sum >= subtraj_len - 0.5).float() * has_valid

        residual = residual * valid_mask

        weight = subtb_lambda ** (subtraj_len - 1)
        batch_loss = batch_loss + weight * (residual**2).sum()
        total_weight = total_weight + weight * valid_mask.sum()

    return batch_loss / total_weight.clamp(min=1.0)
