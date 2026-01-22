"""Enhanced Trajectory Balance losses with FlowRL/TBA features.

Extends trajectory_balance.py with:
- VarGrad logZ estimation (TBA arXiv:2503.18929): No learned logZ parameter,
  estimate partition function from K samples per prompt instead.
- Reference model KL regularization (FlowRL arXiv:2509.15207, TBA):
  Adds log π_ref(y|x) term to prevent policy collapse.
- Importance-weighted TB (FlowRL): PPO-style importance weights for off-policy.
- Combined FlowRL loss integrating all features.

These can be used as drop-in replacements for the base TB/SubTB losses
in GFlowNetTrainer, controlled via GFNConfig.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


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
        logZ(x) = logsumexp_k(r(y_k)/β - log π_θ(y_k|x) + log π_ref(y_k|x)) - log(K)

    When ref_log_probs is None, simplifies to:
        logZ(x) = logsumexp_k(r(y_k)/β - log π_θ(y_k|x)) - log(K)

    Args:
        log_probs: Policy log probs [B, T], already masked by response_mask
        log_rewards: Trajectory log rewards [B]
        response_mask: [B, T] mask for response positions
        ref_log_probs: Reference policy log probs [B, T] or None
        reward_temp: β, reward temperature for scaling (FlowRL uses β=15)

    Returns:
        logZ estimate [B] (one per sample; average over responses for same prompt)
    """
    # trajectory-level log prob: sum over response positions
    traj_log_prob = (log_probs * response_mask.float()).sum(dim=-1)  # [B]

    # score = r(y)/β - log π_θ(y|x)
    score = log_rewards / reward_temp - traj_log_prob

    # add reference model correction if available
    if ref_log_probs is not None:
        ref_traj_log_prob = (ref_log_probs * response_mask.float()).sum(dim=-1)
        score = score + ref_traj_log_prob

    # logZ = logsumexp(scores) - log(K) where K = batch size
    # for per-prompt estimation, caller should group by prompt first
    K = score.shape[0]
    logZ = torch.logsumexp(score, dim=0) - torch.log(torch.tensor(K, dtype=score.dtype, device=score.device))

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
    """TB loss with optional KL regularization (FlowRL/TBA combined).

    Formula (FlowRL arXiv:2509.15207):
        L = (logZ + (1/|y|)·log π_θ(y|x) - β·r(x,y) - (1/|y|)·log π_ref(y|x))²

    Without reference model:
        L = (logZ + (1/|y|)·log π_θ(y|x) - β·r(x,y))²

    Args:
        log_probs: Policy log probs [B, T], pre-masked by response_mask
        log_rewards: Trajectory log rewards [B]
        response_mask: [B, T] response position mask
        logZ: Partition function estimate (scalar or [B])
        ref_log_probs: Reference policy log probs [B, T] or None
        length_normalize: Divide by response length (FlowRL uses this)
        reward_temp: β scaling for rewards
        max_residual: Clamp residual magnitude

    Returns:
        (loss, metrics_dict) with training diagnostics
    """
    B, T = log_probs.shape
    mask_f = response_mask.float()

    # response lengths for normalization
    response_lengths = mask_f.sum(dim=-1).clamp(min=1.0)  # [B]

    # trajectory log prob
    traj_log_prob = (log_probs * mask_f).sum(dim=-1)  # [B]
    if length_normalize:
        traj_log_prob = traj_log_prob / response_lengths

    # reference model contribution
    ref_contribution = torch.zeros_like(traj_log_prob)
    if ref_log_probs is not None:
        ref_traj_log_prob = (ref_log_probs * mask_f).sum(dim=-1)
        if length_normalize:
            ref_traj_log_prob = ref_traj_log_prob / response_lengths
        ref_contribution = ref_traj_log_prob

    # scaled rewards
    scaled_rewards = log_rewards * reward_temp if reward_temp != 1.0 else log_rewards

    # TB residual: logZ + log π_θ - r - log π_ref
    residual = logZ + traj_log_prob - scaled_rewards - ref_contribution
    residual = residual.clamp(-max_residual, max_residual)

    loss = (residual ** 2).mean()

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
    """Full FlowRL loss (arXiv:2509.15207) with importance weights.

    Integrates:
    - Length normalization (1/|y|)
    - Reference model KL regularization
    - Importance sampling weights (PPO-style, optional)
    - Configurable reward temperature β

    Formula:
        L = w · (logZ + (1/|y|)·log π_θ - β·r - (1/|y|)·log π_ref)²

    where w = exp(Σ(log π_θ - log π_old)) is the importance weight.

    Args:
        log_probs: Current policy [B, T]
        log_rewards: Log rewards [B]
        response_mask: [B, T] response mask
        logZ: Partition function (scalar or [B])
        ref_log_probs: Reference policy [B, T] or None
        old_log_probs: Old policy for IS weights [B, T] or None
        length_normalize: Divide by response length
        reward_temp: β for reward scaling (FlowRL default: 15)
        max_residual: Residual clamp
        use_importance_weights: Apply PPO-style IS correction

    Returns:
        (loss, metrics_dict)
    """
    B, T = log_probs.shape
    mask_f = response_mask.float()
    response_lengths = mask_f.sum(dim=-1).clamp(min=1.0)

    # trajectory log prob
    traj_log_prob = (log_probs * mask_f).sum(dim=-1)
    if length_normalize:
        traj_log_prob = traj_log_prob / response_lengths

    # reference model
    ref_contribution = torch.zeros_like(traj_log_prob)
    if ref_log_probs is not None:
        ref_traj = (ref_log_probs * mask_f).sum(dim=-1)
        if length_normalize:
            ref_traj = ref_traj / response_lengths
        ref_contribution = ref_traj

    # scaled rewards
    scaled_rewards = log_rewards * reward_temp if reward_temp != 1.0 else log_rewards

    # residual
    residual = logZ + traj_log_prob - scaled_rewards - ref_contribution
    residual = residual.clamp(-max_residual, max_residual)

    squared_residual = residual ** 2

    # importance weights (PPO-style off-policy correction)
    if use_importance_weights and old_log_probs is not None:
        old_traj_log_prob = (old_log_probs * mask_f).sum(dim=-1)
        # w = exp(log π_θ - log π_old) = π_θ / π_old
        log_ratio = (log_probs * mask_f).sum(dim=-1) - old_traj_log_prob
        # clamp for stability (like PPO clip)
        log_ratio = log_ratio.clamp(-5.0, 5.0)
        weights = torch.exp(log_ratio).detach()  # stop gradient on weights
    else:
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
    """VarGrad TB loss: estimates logZ from batch (no learned parameter).

    Combines TBA's VarGrad estimator with TB loss. Treats the batch as K
    responses to estimate the partition function, then computes TB loss.

    For multi-prompt batches, provide prompt_ids to estimate logZ per prompt.
    For single-prompt batches (all responses to same prompt), omit prompt_ids.

    Args:
        log_probs: Policy log probs [B, T]
        log_rewards: Log rewards [B]
        response_mask: [B, T]
        ref_log_probs: Reference log probs [B, T] or None
        length_normalize: Divide by response length
        reward_temp: β for reward scaling
        max_residual: Residual clamp
        prompt_ids: [B] integer IDs grouping responses by prompt (optional)

    Returns:
        (loss, metrics_dict)
    """
    B, T = log_probs.shape
    mask_f = response_mask.float()
    response_lengths = mask_f.sum(dim=-1).clamp(min=1.0)

    # trajectory log prob (unnormalized — used for both logZ estimation and loss)
    traj_log_prob = (log_probs * mask_f).sum(dim=-1)  # [B]

    # optionally apply length normalization for the loss
    if length_normalize:
        traj_log_prob_normed = traj_log_prob / response_lengths
    else:
        traj_log_prob_normed = traj_log_prob

    # reference model
    ref_contribution = torch.zeros_like(traj_log_prob)
    ref_contribution_normed = torch.zeros_like(traj_log_prob)
    if ref_log_probs is not None:
        ref_traj = (ref_log_probs * mask_f).sum(dim=-1)
        ref_contribution = ref_traj
        ref_contribution_normed = ref_traj / response_lengths if length_normalize else ref_traj

    # estimate logZ per prompt (or globally)
    if prompt_ids is not None:
        # group by prompt and estimate logZ per group
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
        # single prompt: estimate from full batch
        scores = log_rewards / reward_temp - traj_log_prob + ref_contribution
        K = B
        logZ = torch.logsumexp(scores, dim=0) - torch.log(
            torch.tensor(K, dtype=scores.dtype, device=scores.device)
        )
        # broadcast to batch
        logZ = logZ.expand(B)

    # detach logZ to prevent gradient through the estimator
    # (VarGrad: gradient comes from the TB loss, not from logZ estimation)
    logZ_detached = logZ.detach()

    # scaled rewards
    scaled_rewards = log_rewards * reward_temp if reward_temp != 1.0 else log_rewards

    # TB residual
    residual = logZ_detached + traj_log_prob_normed - scaled_rewards - ref_contribution_normed
    residual = residual.clamp(-max_residual, max_residual)

    loss = (residual ** 2).mean()

    metrics = {
        "vargrad_loss": loss.item(),
        "estimated_logZ": logZ_detached.mean().item(),
        "mean_residual": residual.mean().item(),
        "std_residual": residual.std().item(),
    }

    return loss, metrics
