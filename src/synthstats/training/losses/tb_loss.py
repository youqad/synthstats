"""Standalone TB loss for non-SkyRL training loops.

Implements Trajectory Balance loss with learned logZ:
    L = (logZ + sum(log_pi) - log_R)^2

Optional features:
- Reference-policy correction (KL regularization)
- Length normalization

For SubTB with termination probabilities or SkyRL integration,
see trajectory_balance.py instead.
"""

import torch
import torch.nn as nn
from torch import Tensor


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
    """Trajectory Balance loss with learned logZ.

    Computes the squared error between the log-space trajectory balance:
        (logZ + sum(masked_log_probs) - log_R)^2

    Args:
        log_probs: Per-token log probabilities, shape [batch, seq_len]
        loss_mask: Boolean mask for which tokens to include in loss.
                   True = include, False = exclude (e.g., <think> tokens)
        log_rewards: Log of rewards, shape [batch]
        logZ: Learnable log partition function (scalar Parameter)
        max_residual: Clamp residual magnitude for numerical stability (default: 100.0).
                      Prevents exploding gradients from extreme reward/logprob values.
        ref_log_probs: Optional reference-policy log-probs, shape [B, T] or [B].
                       If provided, the TB residual uses log_pi - ref_log_probs.
        ref_weight: Weight applied to ref_log_probs before subtraction.
        normalize_by_length: If True, divide masked log-prob sums by trajectory length.

    Returns:
        Scalar loss tensor (mean over batch)

    Note:
        Handles inf/nan in log_rewards gracefully by replacing with -max_residual.
        This preserves the "low reward" signal for GFlowNets; using zeros would
        incorrectly treat failures as neutral outcomes. Combined with residual
        clamping, this ensures stable gradients while still penalizing bad trajectories.

    Example:
        >>> log_probs = torch.tensor([[-0.5, -0.3]])
        >>> loss_mask = torch.ones(1, 2, dtype=torch.bool)
        >>> log_rewards = torch.tensor([0.0])
        >>> logZ = nn.Parameter(torch.tensor(0.8))
        >>> loss = subtb_loss(log_probs, loss_mask, log_rewards, logZ)
    """
    # apply mask and sum log probs per trajectory
    # mask: [B, T] bool -> float, log_probs: [B, T]
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

    # handle NaN/inf in log_rewards (from zero or invalid rewards)
    # use -max_residual as fallback to preserve "low reward" signal
    # (using 0.0 would treat failures as neutral, which is wrong for GFlowNets)
    safe_log_rewards = torch.where(
        torch.isfinite(log_rewards),
        log_rewards,
        torch.full_like(log_rewards, -max_residual),
    )

    # TB loss: (logZ + log_pi - log_R)^2, averaged over batch
    if masked_ref_logprobs is not None:
        masked_logprobs = masked_logprobs - masked_ref_logprobs

    residual = logZ + masked_logprobs - safe_log_rewards  # [B]

    # clamp for numerical stability
    residual = residual.clamp(-max_residual, max_residual)

    loss = (residual**2).mean()

    return loss
