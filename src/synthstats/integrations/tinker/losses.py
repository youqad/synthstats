"""TB and SubTB loss functions for GFlowNet training.

Vanilla TB: (logZ + sum(log_pf) - log_R)^2.
Endpoint SubTB: local flow-matching at each prefix for variance reduction.
Combined: L = L_TB + alpha * L_SubTB, alpha << 1.
"""

from __future__ import annotations

from typing import Any

from synthstats.core.constants import SUBTB_LAMBDA_DEFAULT, TB_MAX_RESIDUAL_DEFAULT
from synthstats.core.math_utils import sanitize_finite
from synthstats.train.objectives.losses import subtb_loss


def compute_vanilla_tb_loss(
    log_pf: Any,
    log_reward: Any,
    logZ: Any,
    loss_mask: Any,
    *,
    max_residual: float = TB_MAX_RESIDUAL_DEFAULT,
    return_residual: bool = False,
) -> Any:
    """Vanilla TB loss with optional residual for Tinker's analytic logZ update."""
    import torch

    device = log_pf.device
    dtype = log_pf.dtype
    if isinstance(logZ, torch.Tensor):
        logZ_t = logZ.to(device=device, dtype=dtype)
    else:
        logZ_t = torch.tensor(logZ, device=device, dtype=dtype)

    loss = subtb_loss(
        log_probs=log_pf,
        loss_mask=loss_mask,
        log_rewards=log_reward,
        logZ=logZ_t,
        max_residual=max_residual,
    )

    if return_residual:
        mask_f = loss_mask.float()
        trajectory_logprob = (log_pf * mask_f).sum(dim=-1)
        safe_log_reward = sanitize_finite(log_reward, -max_residual)
        residual = logZ_t + trajectory_logprob - safe_log_reward
        residual_clamped = residual.clamp(-max_residual, max_residual)
        return loss, residual_clamped.mean().item()
    return loss


def compute_combined_tb_subtb_loss(
    log_pf: Any,
    log_reward: Any,
    logZ: Any,
    loss_mask: Any,
    *,
    eos_logprob: Any | None = None,
    eos_available: Any | None = None,
    log_sparse_reward: float | None = None,
    ab_subtb_alpha: float = 0.1,
    subtb_lambda: float = SUBTB_LAMBDA_DEFAULT,
    max_residual: float = TB_MAX_RESIDUAL_DEFAULT,
) -> tuple[Any, dict[str, float]]:
    """Combined TB + SubTB loss. SubTB only where EOS logprobs are available."""
    from synthstats.train.objectives.subtb_endpoint import (
        broadcast_terminal_reward,
        compute_endpoint_subtb_loss,
        create_eos_unavailable_mask,
    )

    tb_loss, tb_residual = compute_vanilla_tb_loss(
        log_pf=log_pf,
        log_reward=log_reward,
        logZ=logZ,
        loss_mask=loss_mask,
        max_residual=max_residual,
        return_residual=True,
    )

    metrics: dict[str, float] = {
        "loss/tb": tb_loss.item(),
        "logZ": logZ.item() if hasattr(logZ, "item") else float(logZ),
        "tb_residual": tb_residual,
    }

    B, T = log_pf.shape
    device = log_pf.device

    if (eos_logprob is None) != (eos_available is None):
        raise ValueError("eos_logprob and eos_available must be provided together or both be None")
    if eos_logprob is None and eos_available is None:
        # no EOS info, TB-only
        eos_logprob, eos_available = create_eos_unavailable_mask(B, T, device)
        eos_logprob = eos_logprob.to(device=device, dtype=log_pf.dtype)
    else:
        import torch

        from synthstats.integrations.tinker.eos_extraction import pad_eos_for_subtb

        if not torch.is_tensor(eos_logprob):
            eos_logprob = torch.as_tensor(eos_logprob)
        if not torch.is_tensor(eos_available):
            eos_available = torch.as_tensor(eos_available)

        eos_logprob = eos_logprob.to(device=device, dtype=log_pf.dtype)
        eos_available = eos_available.to(device=device, dtype=torch.bool)

        if eos_logprob.shape != eos_available.shape:
            raise ValueError(
                "eos_logprob and eos_available must have the same shape: "
                f"eos_logprob={tuple(eos_logprob.shape)} "
                f"eos_available={tuple(eos_available.shape)}"
            )

        if eos_logprob.dim() != 2:
            raise ValueError(
                f"eos_logprob must be 2D [B, T] or [B, T+1], got shape={tuple(eos_logprob.shape)}"
            )

        if eos_logprob.shape[0] != B:
            raise ValueError(
                "eos_logprob batch size mismatch: "
                f"eos_logprob.shape[0]={eos_logprob.shape[0]} log_pf.shape[0]={B}"
            )

        if eos_logprob.shape[1] == T:
            eos_logprob, eos_available = pad_eos_for_subtb(eos_logprob, eos_available)
        elif eos_logprob.shape[1] != T + 1:
            raise ValueError(
                "eos_logprob must have shape [B, T] or [B, T+1]: "
                f"got {tuple(eos_logprob.shape)} with T={T}"
            )

    per_prefix_log_reward = broadcast_terminal_reward(
        log_reward, T, log_sparse_reward=log_sparse_reward
    )

    # SubTB must not contribute logZ gradients (analytic update uses TB residual only)
    subtb_logZ = logZ.detach() if hasattr(logZ, "detach") else logZ

    subtb_loss, subtb_metrics = compute_endpoint_subtb_loss(
        log_pf=log_pf,
        log_reward=per_prefix_log_reward,
        eos_logprob=eos_logprob,
        eos_available=eos_available,
        loss_mask=loss_mask,
        logZ=subtb_logZ,
        subtb_lambda=subtb_lambda,
        max_residual=max_residual,
    )

    total_loss = tb_loss + ab_subtb_alpha * subtb_loss

    metrics["loss/subtb"] = subtb_metrics["subtb_loss"]
    metrics["loss/total"] = total_loss.item()
    metrics["coverage/eos_available"] = subtb_metrics["subtb_coverage"]
    metrics["subtb/valid_count"] = subtb_metrics["subtb_valid_count"]
    if subtb_metrics.get("subtb_warning", 0.0) > 0:
        metrics["subtb/warning"] = 1.0

    return total_loss, metrics
