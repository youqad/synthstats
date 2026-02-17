"""TB and SubTB losses for SkyRL integration.

TB:    L = (logZ + sum(log_pi) - log_R)^2
SubTB: L = Sigma_{len} lambda^{len-1} * (delta_cumsum[:, len:] - delta_cumsum[:, :-len])^2

tb_identity estimator passes log_rewards through as advantages.
logZ (and eos_logprobs for SubTB) injected via config by TBTrainer/SubTBTrainer.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from synthstats.core.constants import SUBTB_LAMBDA_DEFAULT, TB_MAX_RESIDUAL_DEFAULT
from synthstats.core.math_utils import sanitize_finite
from synthstats.train.objectives.losses import compute_modified_subtb_core

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def _config_get(config: DictConfig | dict[str, Any], key: str, default: Any) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def compute_tb_identity_advantage(
    token_level_rewards: Tensor,
    response_mask: Tensor,
    **kwargs: Any,
) -> tuple[Tensor, Tensor]:
    """Pass log_rewards through as 'advantages' for TB training."""
    with torch.no_grad():
        trajectory_rewards = (token_level_rewards * response_mask).sum(dim=-1, keepdim=True)
        log_rewards = trajectory_rewards.expand_as(token_level_rewards) * response_mask

    return log_rewards, log_rewards  # (advantages, returns)


def compute_trajectory_balance_loss(
    log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    config: DictConfig | dict[str, Any],
    loss_mask: Tensor | None = None,
    rollout_log_probs: Tensor | None = None,
) -> tuple[Tensor, float]:
    """TB loss: L = (logZ + sum(log_pi) - log_R)^2. Returns (loss, clip_ratio=0.0)."""
    # tensor preserves gradient for logZ learning
    logZ_tensor = getattr(config, "_logZ_tensor", None)
    logZ_val = _config_get(config, "logZ", None)
    max_residual = _config_get(config, "tb_max_residual", TB_MAX_RESIDUAL_DEFAULT)

    if logZ_tensor is None and logZ_val is None:
        raise RuntimeError(
            "TB logZ not found in config. "
            "Ensure TBTrainer sets config.logZ before loss computation."
        )

    if logZ_tensor is not None:
        logZ = logZ_tensor.to(dtype=log_probs.dtype, device=log_probs.device)
    else:
        logZ = torch.tensor(logZ_val, dtype=log_probs.dtype, device=log_probs.device)

    log_rewards = advantages

    if loss_mask is not None:
        mask_f = loss_mask.float()
        masked_logprobs = (log_probs * mask_f).sum(dim=-1)  # [B]
        token_counts = mask_f.sum(dim=-1)  # [B]
        log_rewards_seq = torch.where(
            token_counts > 0,
            (log_rewards * mask_f).sum(dim=-1) / token_counts,
            torch.zeros_like(token_counts),
        )
        valid_traj = token_counts > 0  # [B]
    else:
        masked_logprobs = log_probs.sum(dim=-1)  # [B]
        log_rewards_seq = log_rewards.mean(dim=-1)  # [B]
        valid_traj = torch.ones(log_probs.shape[0], dtype=torch.bool, device=log_probs.device)

    safe_log_rewards = sanitize_finite(log_rewards_seq, -max_residual)

    residual = logZ + masked_logprobs - safe_log_rewards
    residual = residual.clamp(-max_residual, max_residual)

    # empty-mask trajectories would give wrong gradient
    if valid_traj.all():
        loss = (residual**2).mean()
    else:
        valid_count = valid_traj.sum()
        if valid_count > 0:
            loss = (residual[valid_traj] ** 2).sum() / valid_count
        else:
            # All trajectories empty. Returning a plain `torch.tensor(0.0)` here
            # produces a non-differentiable constant and can break callers that
            # unconditionally call `loss.backward()`.
            #
            # Return a zero value that stays connected to the computation graph
            # (even though gradients will be zero in this degenerate case).
            logger.warning(
                "TB loss got an all-empty loss_mask (no valid tokens). "
                "Returning a zero loss connected to the graph; this batch will not learn."
            )
            loss = (masked_logprobs.sum() + logZ) * 0.0
            if not loss.requires_grad:
                # If neither log_probs nor logZ require grad (unexpected in training),
                # ensure backward() is still callable without error.
                loss = loss.requires_grad_()

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"TB loss: {loss.item():.4f}, logZ: {logZ.item():.4f}, "
            f"mean_log_pi: {masked_logprobs.mean().item():.4f}, "
            f"mean_log_R: {safe_log_rewards.mean().item():.4f}"
        )

    return loss, 0.0  # no clip ratio for TB


def compute_modified_subtb_loss(
    log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    config: DictConfig | dict[str, Any],
    loss_mask: Tensor | None = None,
    rollout_log_probs: Tensor | None = None,
) -> tuple[Tensor, float]:
    """Modified SubTB loss from gfn-lm-tuning. Falls back to vanilla TB if no EOS logprobs.

    Does NOT incorporate logZ (EOS-potential regularizer only). For proper TB+SubTB
    with logZ, use compute_endpoint_subtb_loss.
    """
    eos_logprobs = getattr(config, "_eos_logprobs", None)

    if eos_logprobs is None:
        logger.debug("No _eos_logprobs in config, falling back to vanilla TB")
        return compute_trajectory_balance_loss(
            log_probs, old_log_probs, advantages, config, loss_mask, rollout_log_probs
        )

    subtb_lambda = _config_get(config, "subtb_lambda", SUBTB_LAMBDA_DEFAULT)
    max_residual = _config_get(config, "tb_max_residual", TB_MAX_RESIDUAL_DEFAULT)

    B, T = log_probs.shape
    device = log_probs.device
    dtype = log_probs.dtype

    # ensure eos_logprobs is on the right device/dtype
    if isinstance(eos_logprobs, Tensor):
        eos_logprobs = eos_logprobs.to(dtype=dtype, device=device)
    else:
        eos_logprobs = torch.tensor(eos_logprobs, dtype=dtype, device=device)

    # handle mask
    if loss_mask is not None:
        mask = loss_mask.float()
    else:
        mask = torch.ones(B, T, device=device, dtype=dtype)

    # advantages is actually log_rewards (from tb_identity estimator)
    log_rewards = advantages

    loss = compute_modified_subtb_core(
        log_probs=log_probs,
        eos_logprobs=eos_logprobs,
        log_rewards=log_rewards,
        mask=mask,
        subtb_lambda=subtb_lambda,
        max_residual=max_residual,
    )

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"SubTB loss: {loss.item():.4f}, lambda: {subtb_lambda}, seq_len: {T}, batch_size: {B}"
        )

    return loss, 0.0  # no clip ratio for SubTB


def estimate_log_partition_vargrad(
    log_probs: Tensor,
    log_rewards: Tensor,
    response_mask: Tensor,
    ref_log_probs: Tensor | None = None,
    reward_temp: float = 1.0,
) -> Tensor:
    """VarGrad partition function estimator (TBA arXiv:2503.18929).

    Estimates logZ per prompt from a batch of K responses, avoiding the need
    for a learned logZ parameter. Works best with K >= 4 responses per prompt.

    Formula (log-space for numerical stability):
        logZ(x) = logsumexp_k(r(y_k)/beta - log pi_theta(y_k|x) + log pi_ref(y_k|x)) - log(K)
    """
    if reward_temp <= 0:
        raise ValueError(f"reward_temp must be positive, got {reward_temp}")

    traj_log_prob = (log_probs * response_mask.float()).sum(dim=-1)
    score = log_rewards / reward_temp - traj_log_prob

    if ref_log_probs is not None:
        ref_traj_log_prob = (ref_log_probs * response_mask.float()).sum(dim=-1)
        score = score + ref_traj_log_prob

    K = score.shape[0]
    logZ = torch.logsumexp(score, dim=0) - torch.log(
        torch.tensor(K, dtype=score.dtype, device=score.device)
    )
    return logZ


def compute_tb_loss_with_kl(
    log_probs: Tensor,
    log_rewards: Tensor,
    response_mask: Tensor,
    logZ: Tensor,
    ref_log_probs: Tensor | None = None,
    length_normalize: bool = True,
    reward_temp: float = 1.0,
    max_residual: float = 100.0,
) -> tuple[Tensor, dict[str, float]]:
    """TB loss with optional KL regularization (FlowRL/TBA combined)."""
    if reward_temp <= 0:
        raise ValueError(f"reward_temp must be positive, got {reward_temp}")
    B, T = log_probs.shape
    mask_f = response_mask.float()
    response_lengths = mask_f.sum(dim=-1).clamp(min=1.0)

    traj_log_prob = (log_probs * mask_f).sum(dim=-1)
    if length_normalize:
        traj_log_prob = traj_log_prob / response_lengths

    ref_contribution = torch.zeros_like(traj_log_prob)
    if ref_log_probs is not None:
        ref_traj_log_prob = (ref_log_probs * mask_f).sum(dim=-1)
        if length_normalize:
            ref_traj_log_prob = ref_traj_log_prob / response_lengths
        ref_contribution = ref_traj_log_prob

    scaled_rewards = log_rewards / reward_temp if reward_temp != 1.0 else log_rewards
    residual = logZ + traj_log_prob - scaled_rewards - ref_contribution
    residual = residual.clamp(-max_residual, max_residual)
    loss = (residual**2).mean()

    metrics = {
        "tb_loss": loss.item(),
        "mean_residual": residual.mean().item(),
        "std_residual": residual.std().item(),
        "mean_traj_log_prob": traj_log_prob.mean().item(),
        "mean_response_length": response_lengths.mean().item(),
    }
    if ref_log_probs is not None:
        metrics["mean_ref_log_prob"] = ref_contribution.mean().item()

    return loss, metrics


def compute_flowrl_loss(
    log_probs: Tensor,
    log_rewards: Tensor,
    response_mask: Tensor,
    logZ: Tensor,
    ref_log_probs: Tensor | None = None,
    old_log_probs: Tensor | None = None,
    length_normalize: bool = True,
    reward_temp: float = 15.0,
    max_residual: float = 100.0,
    use_importance_weights: bool = False,
) -> tuple[Tensor, dict[str, float]]:
    """Full FlowRL loss (arXiv:2509.15207) with importance weights."""
    if reward_temp <= 0:
        raise ValueError(f"reward_temp must be positive, got {reward_temp}")
    B, T = log_probs.shape
    mask_f = response_mask.float()
    response_lengths = mask_f.sum(dim=-1).clamp(min=1.0)

    traj_log_prob = (log_probs * mask_f).sum(dim=-1)
    if length_normalize:
        traj_log_prob = traj_log_prob / response_lengths

    ref_contribution = torch.zeros_like(traj_log_prob)
    if ref_log_probs is not None:
        ref_traj = (ref_log_probs * mask_f).sum(dim=-1)
        if length_normalize:
            ref_traj = ref_traj / response_lengths
        ref_contribution = ref_traj

    scaled_rewards = log_rewards / reward_temp if reward_temp != 1.0 else log_rewards
    residual = logZ + traj_log_prob - scaled_rewards - ref_contribution
    residual = residual.clamp(-max_residual, max_residual)
    squared_residual = residual**2

    if use_importance_weights and old_log_probs is not None:
        old_traj_log_prob = (old_log_probs * mask_f).sum(dim=-1)
        log_ratio = (log_probs * mask_f).sum(dim=-1) - old_traj_log_prob
        log_ratio = log_ratio.clamp(-5.0, 5.0)
        weights = torch.exp(log_ratio).detach()
    else:
        if use_importance_weights and old_log_probs is None:
            logger.warning("use_importance_weights=True but old_log_probs not provided")
        weights = torch.ones(B, device=log_probs.device, dtype=log_probs.dtype)

    loss = (weights * squared_residual).mean()

    metrics = {
        "flowrl_loss": loss.item(),
        "mean_residual": residual.mean().item(),
        "mean_is_weight": weights.mean().item(),
        "reward_temp": reward_temp,
    }
    return loss, metrics


def compute_vargrad_tb_loss(
    log_probs: Tensor,
    log_rewards: Tensor,
    response_mask: Tensor,
    ref_log_probs: Tensor | None = None,
    length_normalize: bool = True,
    reward_temp: float = 1.0,
    max_residual: float = 100.0,
    prompt_ids: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """VarGrad TB loss: estimates logZ from batch (no learned parameter)."""
    if reward_temp <= 0:
        raise ValueError(f"reward_temp must be positive, got {reward_temp}")

    B, T = log_probs.shape
    mask_f = response_mask.float()
    response_lengths = mask_f.sum(dim=-1).clamp(min=1.0)

    traj_log_prob = (log_probs * mask_f).sum(dim=-1)
    traj_log_prob_normed = traj_log_prob / response_lengths if length_normalize else traj_log_prob

    ref_contribution = torch.zeros_like(traj_log_prob)
    ref_contribution_normed = torch.zeros_like(traj_log_prob)
    if ref_log_probs is not None:
        ref_traj = (ref_log_probs * mask_f).sum(dim=-1)
        ref_contribution = ref_traj
        ref_contribution_normed = ref_traj / response_lengths if length_normalize else ref_traj

    if prompt_ids is not None:
        unique_prompts = prompt_ids.unique()
        logZ_per_sample = torch.zeros(B, device=log_probs.device, dtype=log_probs.dtype)
        for pid in unique_prompts:
            group_mask = prompt_ids == pid
            group_scores = (
                log_rewards[group_mask] / reward_temp
                - traj_log_prob[group_mask]
                + ref_contribution[group_mask]
            )
            K = group_scores.shape[0]
            group_logZ = torch.logsumexp(group_scores, dim=0) - torch.log(
                torch.tensor(K, dtype=group_scores.dtype, device=group_scores.device)
            )
            logZ_per_sample[group_mask] = group_logZ
        logZ = logZ_per_sample
    else:
        scores = log_rewards / reward_temp - traj_log_prob + ref_contribution
        K = B
        logZ = torch.logsumexp(scores, dim=0) - torch.log(
            torch.tensor(K, dtype=scores.dtype, device=scores.device)
        )
        logZ = logZ.expand(B)

    logZ_detached = logZ.detach()
    scaled_rewards = log_rewards / reward_temp if reward_temp != 1.0 else log_rewards
    residual = logZ_detached + traj_log_prob_normed - scaled_rewards - ref_contribution_normed
    residual = residual.clamp(-max_residual, max_residual)
    loss = (residual**2).mean()

    metrics = {
        "vargrad_loss": loss.item(),
        "estimated_logZ": logZ_detached.mean().item(),
        "mean_residual": residual.mean().item(),
        "std_residual": residual.std().item(),
    }
    return loss, metrics


def _register_with_skyrl() -> bool:
    """Register TB/SubTB components with SkyRL if available."""
    try:
        from skyrl_train.utils.ppo_utils import (
            register_advantage_estimator,
            register_policy_loss,
        )

        register_advantage_estimator("tb_identity")(compute_tb_identity_advantage)
        logger.info("Registered 'tb_identity' advantage estimator with SkyRL")

        register_policy_loss("trajectory_balance")(compute_trajectory_balance_loss)
        logger.info("Registered 'trajectory_balance' policy loss with SkyRL")

        register_policy_loss("modified_subtb")(compute_modified_subtb_loss)
        logger.info("Registered 'modified_subtb' policy loss with SkyRL")

        return True

    except ImportError:
        logger.warning(
            "SkyRL not available. TB/SubTB losses and estimator not registered. "
            "Install skyrl-train to use native SkyRL integration."
        )
        return False


# auto-register on module import
SKYRL_REGISTERED = _register_with_skyrl()
