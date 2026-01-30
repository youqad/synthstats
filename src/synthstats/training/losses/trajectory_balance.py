"""Trajectory Balance (TB) and Sub-Trajectory Balance (SubTB) losses for SkyRL.

Registers:
- `trajectory_balance` policy loss via @register_policy_loss (vanilla TB)
- `modified_subtb` policy loss via @register_policy_loss (SubTB from gfn-lm-tuning)
- `tb_identity` advantage estimator via @register_advantage_estimator

Loss equations:
- TB:    L = (logZ + sum(log_pi) - log_R)^2
- SubTB: L = Σ_{len} λ^{len-1} * (delta_cumsum[:, len:] - delta_cumsum[:, :-len])^2
         where delta[t] = log_pf[t] - eos_logprob[t] + eos_logprob[t+1]

Architecture:
1. tb_identity estimator passes log_rewards through as "advantages"
2. TB/SubTB loss receives log_rewards via the advantages parameter
3. logZ is injected via config by TBTrainer/SubTBTrainer before each step
4. SubTB additionally requires _eos_logprobs tensor injected into config

This design works correctly with Ray distributed workers because:
- No global state needed
- Config serializes properly across processes
- Tensor injection bypasses OmegaConf serialization for gradient flow
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# TB Identity Advantage Estimator
# -----------------------------------------------------------------------------


def compute_tb_identity_advantage(
    token_level_rewards: Tensor,
    response_mask: Tensor,
    **kwargs: Any,
) -> tuple[Tensor, Tensor]:
    """Identity estimator that passes log_rewards through as 'advantages'.

    This estimator is designed for Trajectory Balance training where we need
    the raw log_rewards, not PPO-style advantages.

    SkyRL's architecture computes advantages from rewards, then passes them
    to policy losses. By registering this identity estimator, we ensure the
    TB loss receives log_rewards in the advantages parameter.

    Args:
        token_level_rewards: Raw rewards from environment [B, T].
            For GFlowNets, this should be log(R) where R is the trajectory reward.
        response_mask: Mask for valid response tokens [B, T]
        **kwargs: Ignored (GAE params like gamma, lambd, values)

    Returns:
        (advantages, returns) - both are the same: trajectory log_rewards
        broadcast to sequence shape for compatibility with SkyRL's masking.
    """
    with torch.no_grad():
        # token_level_rewards typically has reward only at final token
        # sum to get trajectory-level reward
        trajectory_rewards = (token_level_rewards * response_mask).sum(dim=-1, keepdim=True)

        # broadcast back to sequence shape (SkyRL expects [B, T] advantages)
        # the mask ensures only valid tokens contribute to the loss
        log_rewards = trajectory_rewards.expand_as(token_level_rewards) * response_mask

    return log_rewards, log_rewards  # (advantages, returns)


# -----------------------------------------------------------------------------
# Trajectory Balance Policy Loss
# -----------------------------------------------------------------------------


def compute_trajectory_balance_loss(
    log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    config: DictConfig | dict[str, Any],
    loss_mask: Tensor | None = None,
    rollout_log_probs: Tensor | None = None,
) -> tuple[Tensor, float]:
    """Trajectory Balance loss for GFlowNets.

    This is the core GFlowNet training objective that learns to sample
    trajectories proportionally to their reward.

    TB loss: L = (logZ + sum(log_pi) - log_R)^2

    where:
    - logZ: learned partition function (injected via config by TBTrainer)
    - log_pi: sum of log probabilities under the policy
    - log_R: log reward for the trajectory

    Args:
        log_probs: Current policy log probs [B, T]
        old_log_probs: Old policy log probs (unused for TB)
        advantages: Log rewards from tb_identity estimator [B, T]
        config: Training config containing:
            - logZ: current partition function estimate (required)
            - tb_max_residual: clamp residual to [-max, max] (default: 100.0)
        loss_mask: Boolean mask for valid tokens [B, T]
        rollout_log_probs: Rollout log probs (unused)

    Returns:
        (loss, clip_ratio) - clip_ratio is always 0.0 for TB (no clipping)
    """
    # get logZ from config (injected by TBTrainer)
    # first check for tensor (preserves gradient for logZ learning)
    logZ_tensor = getattr(config, "_logZ_tensor", None)

    if hasattr(config, "get"):
        logZ_val = config.get("logZ", None)
        max_residual = config.get("tb_max_residual", 100.0)
    elif isinstance(config, dict):
        logZ_val = config.get("logZ", None)
        max_residual = config.get("tb_max_residual", 100.0)
    else:
        logZ_val = getattr(config, "logZ", None)
        max_residual = getattr(config, "tb_max_residual", 100.0)

    if logZ_tensor is None and logZ_val is None:
        raise RuntimeError(
            "TB logZ not found in config. "
            "Ensure TBTrainer sets config.logZ before loss computation."
        )

    # use tensor if available (preserves gradient), otherwise create from float
    if logZ_tensor is not None:
        logZ = logZ_tensor.to(dtype=log_probs.dtype, device=log_probs.device)
    else:
        logZ = torch.tensor(logZ_val, dtype=log_probs.dtype, device=log_probs.device)

    # advantages is actually log_rewards (from tb_identity estimator)
    log_rewards = advantages

    # sum log_probs per trajectory (masked)
    if loss_mask is not None:
        mask_f = loss_mask.float()
        # sum log probs for the trajectory
        masked_logprobs = (log_probs * mask_f).sum(dim=-1)  # [B]
        # log_rewards is broadcast, so sum and normalize to get trajectory value
        # since it's broadcast from a single value, just take the sum divided by count
        token_counts = mask_f.sum(dim=-1).clamp(min=1)
        log_rewards_seq = (log_rewards * mask_f).sum(dim=-1) / token_counts  # [B]
    else:
        masked_logprobs = log_probs.sum(dim=-1)  # [B]
        log_rewards_seq = log_rewards.mean(dim=-1)  # [B]

    # handle NaN/inf in log_rewards
    safe_log_rewards = torch.where(
        torch.isfinite(log_rewards_seq),
        log_rewards_seq,
        torch.full_like(log_rewards_seq, -max_residual),
    )

    # TB loss: (logZ + log_pi - log_R)^2
    residual = logZ + masked_logprobs - safe_log_rewards
    residual = residual.clamp(-max_residual, max_residual)

    loss = (residual**2).mean()

    # debug logging
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"TB loss: {loss.item():.4f}, logZ: {logZ.item():.4f}, "
            f"mean_log_pi: {masked_logprobs.mean().item():.4f}, "
            f"mean_log_R: {safe_log_rewards.mean().item():.4f}"
        )

    return loss, 0.0  # no clip ratio for TB


# -----------------------------------------------------------------------------
# Sub-Trajectory Balance (SubTB) Policy Loss
# -----------------------------------------------------------------------------


def compute_modified_subtb_loss(
    log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    config: DictConfig | dict[str, Any],
    loss_mask: Tensor | None = None,
    rollout_log_probs: Tensor | None = None,
) -> tuple[Tensor, float]:
    """Modified Sub-Trajectory Balance loss from gfn-lm-tuning.

    SubTB computes flow matching residuals for ALL sub-trajectory lengths,
    weighted by lambda^(len-1). This provides stronger gradient signal than
    vanilla TB which only matches full trajectories.

    The key insight: instead of only enforcing flow conservation at the final
    state (TB), SubTB enforces it at every sub-trajectory endpoint. This gives
    denser reward signal during training.

    Formula:
        delta[t] = log_pf[t] - eos_logprob[t] + eos_logprob[t+1]
        (when log_R is broadcast/constant, the log_R terms cancel)

        For each sub-trajectory length len:
            residual = delta_cumsum[:, len:] - delta_cumsum[:, :-len]
            loss += lambda^(len-1) * residual^2

    Args:
        log_probs: Current policy log probs [B, T]
        old_log_probs: Old policy log probs (unused for SubTB)
        advantages: Log rewards from tb_identity estimator [B, T]
        config: Training config containing:
            - logZ: current partition function estimate (required)
            - _eos_logprobs: EOS log probs tensor [B, T] (injected by SubTBTrainer)
            - subtb_lambda: decay factor for sub-trajectory lengths (default: 0.9)
            - tb_max_residual: clamp residual magnitude (default: 100.0)
        loss_mask: Boolean mask for valid tokens [B, T]
        rollout_log_probs: Rollout log probs (unused)

    Returns:
        (loss, clip_ratio) - clip_ratio is always 0.0 for SubTB (no clipping)

    Note:
        Falls back to vanilla TB if _eos_logprobs is not available in config.
    """
    # get eos_logprobs from config (injected by SubTBTrainer)
    eos_logprobs = getattr(config, "_eos_logprobs", None)

    # fallback to vanilla TB if no EOS logprobs available
    if eos_logprobs is None:
        logger.debug("No _eos_logprobs in config, falling back to vanilla TB")
        return compute_trajectory_balance_loss(
            log_probs, old_log_probs, advantages, config, loss_mask, rollout_log_probs
        )

    # get config values
    if hasattr(config, "get"):
        subtb_lambda = config.get("subtb_lambda", 0.9)
        max_residual = config.get("tb_max_residual", 100.0)
    elif isinstance(config, dict):
        subtb_lambda = config.get("subtb_lambda", 0.9)
        max_residual = config.get("tb_max_residual", 100.0)
    else:
        subtb_lambda = getattr(config, "subtb_lambda", 0.9)
        max_residual = getattr(config, "tb_max_residual", 100.0)

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
    # we need this for the final step to anchor flow to reward
    log_rewards = advantages

    # compute per-step delta (flow matching residual)
    # For INTERNAL steps (t = 0 to T-2):
    #   delta[t] = log_pf[t] - eos_logprob[t] + eos_logprob[t+1]
    # For FINAL step (t = T-1, termination):
    #   delta[T-1] = log_pf[T-1] - eos_logprob[T-1] + log_R
    # This anchors the flow to the trajectory reward.

    # sanitize log_rewards (handle NaN/inf like vanilla TB does)
    safe_log_rewards = torch.where(
        torch.isfinite(log_rewards),
        log_rewards,
        torch.full_like(log_rewards, -max_residual),
    )

    # For variable-length sequences with padding, the reward must anchor at each
    # sample's LAST VALID position, not the global last column.
    # We compute delta in two parts:
    #   1. Internal delta at ALL positions: log_pf[t] - eos[t] + eos[t+1]
    #   2. At the last valid position, REPLACE with: log_pf[t] - eos[t] + log_R
    #
    # The difference is: at position t, we add (log_R - eos[t+1]) to convert
    # internal delta to final delta.

    if T == 1:
        # single-token: only the final (reward-anchored) delta
        # take trajectory reward from any position (all same due to broadcast)
        traj_reward = safe_log_rewards[:, 0:1]  # [B, 1]
        delta = log_probs - eos_logprobs + traj_reward  # [B, 1]
    else:
        # compute internal deltas for ALL positions first
        # internal: delta[t] = log_pf[t] - eos[t] + eos[t+1]
        # for position T-1, we use eos[T-1] as placeholder (will be corrected below)
        eos_next = torch.cat([eos_logprobs[:, 1:], eos_logprobs[:, -1:]], dim=1)  # [B, T]
        delta = log_probs - eos_logprobs + eos_next  # [B, T]

        # find each sample's last valid position
        # handles non-contiguous masks (e.g., prompt masking, gaps)
        # last_valid_idx[b] = max index where mask[b, idx] > 0
        valid = mask > 0
        idxs = torch.arange(T, device=device, dtype=torch.long)
        # multiply valid by indices, take max to get last valid position
        # if all masked, falls back to 0
        last_valid_idx = torch.where(
            valid.any(dim=1),
            (valid.float() * idxs).long().max(dim=1).values,
            torch.zeros(B, device=device, dtype=torch.long),
        )  # [B]

        # at the last valid position, we need to replace eos[t+1] with log_R
        # correction = log_R - eos[t+1]
        # use gather to get reward at each sample's last valid position (handles non-broadcast)
        traj_reward = safe_log_rewards.gather(1, last_valid_idx.unsqueeze(1)).squeeze(1)  # [B]
        eos_next_at_last = eos_next.gather(1, last_valid_idx.unsqueeze(1)).squeeze(1)  # [B]
        correction = traj_reward - eos_next_at_last  # [B]

        # apply correction only at the last valid position using scatter_add
        correction_tensor = torch.zeros_like(delta)  # [B, T]
        correction_tensor.scatter_(1, last_valid_idx.unsqueeze(1), correction.unsqueeze(1))
        delta = delta + correction_tensor

    # CRITICAL: zero out masked positions BEFORE cumsum to prevent NaN propagation
    # if delta[i] is NaN/inf at a masked position, cumsum would propagate it forward
    # this ensures garbage values at masked positions don't corrupt the loss
    delta = torch.where(mask > 0, delta, torch.zeros_like(delta))

    # cumulative sum for efficient sub-trajectory computation
    # prepend zero so cumsum[len] - cumsum[0] = sum(delta[0:len])
    zeros = torch.zeros(B, 1, device=device, dtype=dtype)
    delta_cumsum = torch.cat([zeros, delta], dim=1).cumsum(dim=1)  # [B, T+1]

    # precompute mask cumsum for sub-trajectory validity checking
    # a sub-trajectory [t, t+len) is valid only if ALL positions in range are valid
    # mask_cumsum[t+len] - mask_cumsum[t] == len means no gaps in mask
    mask_cumsum = torch.cat([zeros, mask], dim=1).cumsum(dim=1)  # [B, T+1]

    # track samples with at least one valid position (exclude fully-masked samples)
    has_valid = (mask.sum(dim=1) > 0).float().unsqueeze(1)  # [B, 1]

    # sum over all sub-trajectory lengths (including full trajectory T)
    batch_loss = torch.tensor(0.0, device=device, dtype=dtype)
    total_weight = torch.tensor(0.0, device=device, dtype=dtype)

    for subtraj_len in range(1, T + 1):
        # residual for sub-trajectories of this length
        # residual[t] = sum(delta[t:t+len]) = cumsum[t+len] - cumsum[t]
        residual = delta_cumsum[:, subtraj_len:] - delta_cumsum[:, :-subtraj_len]  # [B, T+1-len]
        residual = residual.clamp(-max_residual, max_residual)

        # sub-trajectory [t, t+len) is valid only if ALL positions are valid
        # mask_sum >= subtraj_len - 0.5 handles floating-point precision issues
        # (bfloat16 only represents integers exactly up to 256; cumsum loses precision beyond that)
        mask_sum = mask_cumsum[:, subtraj_len:] - mask_cumsum[:, :-subtraj_len]  # [B, T+1-len]
        valid_mask = (mask_sum >= subtraj_len - 0.5).float() * has_valid  # [B, T+1-len]

        # zero out invalid positions to prevent NaN propagation
        # (NaN * 0 = NaN, but torch.where avoids this)
        residual = torch.where(valid_mask > 0, residual, torch.zeros_like(residual))

        # lambda-weighted squared residual
        weight = subtb_lambda ** (subtraj_len - 1)
        batch_loss = batch_loss + weight * (residual**2).sum()
        total_weight = total_weight + weight * valid_mask.sum()

    # normalize by total weight
    loss = batch_loss / total_weight.clamp(min=1.0)

    # debug logging
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"SubTB loss: {loss.item():.4f}, lambda: {subtb_lambda}, seq_len: {T}, batch_size: {B}"
        )

    return loss, 0.0  # no clip ratio for SubTB


# -----------------------------------------------------------------------------
# VarGrad / FlowRL / KL-regularized TB losses
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# SkyRL Registration
# -----------------------------------------------------------------------------


def _register_with_skyrl() -> bool:
    """Register TB/SubTB components with SkyRL if available.

    Registers:
        - tb_identity: Advantage estimator that passes log_rewards through
        - trajectory_balance: Vanilla TB loss (whole-trajectory matching)
        - modified_subtb: SubTB loss (sub-trajectory flow matching)

    Returns:
        True if registration succeeded, False otherwise.
    """
    try:
        from skyrl_train.utils.ppo_utils import (
            register_advantage_estimator,
            register_policy_loss,
        )

        # register advantage estimator (shared by TB and SubTB)
        register_advantage_estimator("tb_identity")(compute_tb_identity_advantage)
        logger.info("Registered 'tb_identity' advantage estimator with SkyRL")

        # register vanilla TB policy loss
        register_policy_loss("trajectory_balance")(compute_trajectory_balance_loss)
        logger.info("Registered 'trajectory_balance' policy loss with SkyRL")

        # register SubTB policy loss (from gfn-lm-tuning)
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
