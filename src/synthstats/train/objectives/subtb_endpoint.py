"""Endpoint-based SubTB loss for partial EOS availability.

Uses telescoped formula: residual_{i->j} = u[i] + sum log_pf[i:j] - u[j]
where u[t] = log_R[t] - eos_logprob[t]. Only needs EOS at endpoints.
"""

from __future__ import annotations

import torch
from torch import Tensor

from synthstats.core.constants import (
    LOG_SPARSE_REWARD_DEFAULT,
    SUBTB_LAMBDA_DEFAULT,
    TB_MAX_RESIDUAL_DEFAULT,
)
from synthstats.core.math_utils import sanitize_finite


def compute_endpoint_subtb_loss(
    log_pf: Tensor,
    log_reward: Tensor,
    eos_logprob: Tensor,
    eos_available: Tensor,
    *,
    loss_mask: Tensor | None = None,
    logZ: Tensor | float | None = None,
    subtb_lambda: float = SUBTB_LAMBDA_DEFAULT,
    max_residual: float = TB_MAX_RESIDUAL_DEFAULT,
    min_valid_subtrajs: int = 1,
) -> tuple[Tensor, dict[str, float]]:
    """Compute SubTB loss over subtrajectories with valid endpoints."""
    B, T = log_pf.shape
    device = log_pf.device
    dtype = log_pf.dtype

    if loss_mask is not None:
        if loss_mask.shape != log_pf.shape:
            raise ValueError(
                "loss_mask must match log_pf shape: "
                f"loss_mask={tuple(loss_mask.shape)} log_pf={tuple(log_pf.shape)}"
            )
        transition_mask = loss_mask.to(dtype=torch.bool, device=device)
    else:
        transition_mask = torch.ones(B, T, dtype=torch.bool, device=device)

    # cumsum tracks invalid transitions so subtrajectories crossing
    # masked positions (e.g. <think> regions) are excluded entirely
    invalid = (~transition_mask).to(dtype=torch.int32)
    zeros_int = torch.zeros(B, 1, device=device, dtype=torch.int32)
    invalid_cumsum = torch.cat([zeros_int, invalid.cumsum(dim=-1)], dim=-1)

    # sanitize NaN/Inf before computation (NaN * 0 = NaN)
    safe_log_pf = sanitize_finite(log_pf, -max_residual)
    safe_log_reward = sanitize_finite(log_reward, -max_residual)
    safe_eos_logprob = sanitize_finite(eos_logprob, -max_residual)

    u = safe_log_reward - safe_eos_logprob

    # u[0] = logZ (start-state flow / log normalizing constant)
    if logZ is not None:
        if not isinstance(logZ, Tensor):
            logZ = torch.tensor(logZ, device=device, dtype=dtype)
        else:
            logZ = logZ.to(device=device, dtype=dtype)
        u = u.clone()  # avoid in-place modification
        u[:, 0] = logZ.expand(B) if logZ.dim() == 0 else logZ

    zeros = torch.zeros(B, 1, device=device, dtype=dtype)
    log_pf_cumsum = torch.cat([zeros, safe_log_pf.cumsum(dim=-1)], dim=-1)

    total_loss = torch.zeros((), device=device, dtype=dtype)
    total_weight = torch.zeros((), device=device, dtype=dtype)
    total_valid_count = torch.zeros((), device=device, dtype=torch.int64)

    for length in range(1, T + 1):
        weight = float(subtb_lambda ** (length - 1))

        i_idx = torch.arange(T + 1 - length, device=device)
        j_idx = i_idx + length

        eos_valid = eos_available[:, i_idx] & eos_available[:, j_idx]
        # no flow consistency across masked (<think>) regions: the reward
        # signal doesn't reach latent tokens, so SubTB can't balance there
        no_invalid_transitions = (invalid_cumsum[:, j_idx] - invalid_cumsum[:, i_idx]) == 0
        valid = eos_valid & no_invalid_transitions
        valid_f = valid.to(dtype=dtype)

        log_pf_sum = log_pf_cumsum[:, j_idx] - log_pf_cumsum[:, i_idx]
        residual = u[:, i_idx] + log_pf_sum - u[:, j_idx]
        # mask before clamping to avoid NaN propagation
        residual = torch.where(valid, residual, torch.zeros_like(residual))
        residual = residual.clamp(-max_residual, max_residual)

        total_loss = total_loss + weight * (residual.square()).sum()
        total_weight = total_weight + weight * valid_f.sum()
        total_valid_count = total_valid_count + valid.sum()

    valid_count = int(total_valid_count.item())
    if valid_count < min_valid_subtrajs:
        metrics = {
            "subtb_loss": 0.0,
            "subtb_valid_count": valid_count,
            "subtb_coverage": 0.0,
            "subtb_warning": 1.0,
        }
        # graph-connected zero for DDP safety (matches trajectory_balance.py);
        # multiply before sum so fp16 can't overflow on long sequences
        zero = (safe_log_pf * 0.0).sum()
        if not zero.requires_grad:
            zero = zero.requires_grad_()
        return zero, metrics

    loss = total_loss / total_weight.clamp(min=1e-8)
    coverage = eos_available.float().mean().item()

    metrics = {
        "subtb_loss": loss.item(),
        "subtb_valid_count": valid_count,
        "subtb_coverage": coverage,
        "subtb_warning": 0.0,
    }

    return loss, metrics


def broadcast_terminal_reward(
    log_reward: Tensor,
    seq_len: int,
    *,
    log_sparse_reward: float | None = None,
) -> Tensor:
    """Broadcast terminal reward to per-prefix format [B, T+1].

    Incomplete prefixes get log_sparse_reward (default log(1e-4)),
    terminal position gets the actual reward.
    """
    B = log_reward.shape[0]
    device = log_reward.device
    dtype = log_reward.dtype

    if log_sparse_reward is None:
        log_sparse_reward = LOG_SPARSE_REWARD_DEFAULT
    per_prefix = torch.full((B, seq_len + 1), log_sparse_reward, device=device, dtype=dtype)
    per_prefix[:, -1] = log_reward

    return per_prefix


def create_eos_unavailable_mask(
    batch_size: int,
    seq_len: int,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Create placeholder EOS tensors when top-k extraction is unavailable."""
    eos_logprob = torch.full((batch_size, seq_len + 1), -1e6, device=device, dtype=torch.float32)
    eos_available = torch.zeros(batch_size, seq_len + 1, dtype=torch.bool, device=device)

    return eos_logprob, eos_available
