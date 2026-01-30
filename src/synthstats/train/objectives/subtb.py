"""SubTB (Sub-Trajectory Balance) objective for GFlowNet training.

Pure loss computation as nn.Module:
- Contains logZ as learnable Parameter
- forward() computes loss and metrics
- No optimizer, no stepping (that's the Learner's job)

Supports:
- Vanilla TB: L = (logZ + sum(log_pi) - log_R)^2
- Modified SubTB: Lambda-weighted sub-trajectory flow matching
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class SubTBConfig:
    """Configuration for SubTB objective."""

    beta: float = 1.0
    loss_type: str = "tb"  # "tb" or "modified_subtb"
    subtb_lambda: float = 0.9
    tb_max_residual: float = 100.0
    logZ_init: float = 0.0

    # reference policy (KL regularization)
    use_ref_policy: bool = False
    ref_weight: float = 1.0
    normalize_by_length: bool = False

    # entropy bonus
    entropy_coef: float = 0.01


class SubTBObjective(nn.Module):
    """SubTB loss computation with learnable logZ.

    This is the core GFlowNet objective that learns to sample trajectories
    proportionally to their reward.

    Args:
        config: SubTBConfig with hyperparameters
        device: Device for logZ parameter

    Example:
        >>> objective = SubTBObjective(SubTBConfig(loss_type="tb"))
        >>> loss, metrics = objective(batch)
        >>> # caller handles backward() and optimizer step
    """

    def __init__(
        self,
        config: SubTBConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.config = config or SubTBConfig()
        self._device = torch.device(device)

        # learnable log partition function
        self.logZ = nn.Parameter(torch.tensor(self.config.logZ_init, device=self._device))

    def forward(
        self,
        batch: dict[str, Any],
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute SubTB loss and metrics.

        Args:
            batch: Dict with keys:
                - log_probs: [B, T] or [B] (if pre-summed)
                - log_reward: [B]
                - loss_mask: optional [B, T] bool/int/float
                - entropy: optional [B] or [B, T]
                - ref_log_probs: optional [B, T] or [B]
                - eos_logprobs: optional [B, T] for modified SubTB

        Returns:
            (loss_tensor, metrics_dict) where metrics contains:
                - loss: total loss value
                - tb_loss: TB component before entropy bonus
                - entropy: mean entropy
                - logZ: current logZ value
        """
        # move tensors to device
        log_probs: Tensor = batch["log_probs"].to(self._device)
        log_reward: Tensor = batch["log_reward"].to(self._device)
        mask: Tensor | None = batch.get("loss_mask")
        entropy: Tensor | None = batch.get("entropy")
        ref_log_probs: Tensor | None = batch.get("ref_log_probs")
        eos_logprobs: Tensor | None = batch.get("eos_logprobs")

        if mask is not None:
            mask = mask.to(self._device)
        if entropy is not None:
            entropy = entropy.to(self._device)
        if ref_log_probs is not None:
            ref_log_probs = ref_log_probs.to(self._device)
        if eos_logprobs is not None:
            eos_logprobs = eos_logprobs.to(self._device)

        # validate log_probs shape and prepare for loss (expects 2D)
        if log_probs.dim() == 1:
            log_probs_2d = log_probs.unsqueeze(1)
            mask_2d = torch.ones_like(log_probs_2d, dtype=torch.bool)
        elif log_probs.dim() == 2:
            log_probs_2d = log_probs
            mask_2d = mask if mask is not None else torch.ones_like(log_probs_2d, dtype=torch.bool)
        else:
            raise ValueError(f"log_probs must be 1D [B] or 2D [B, T], got shape {log_probs.shape}")

        # validate ref_log_probs consistency
        if ref_log_probs is not None and not self.config.use_ref_policy:
            raise ValueError("ref_log_probs provided but use_ref_policy=False in SubTBConfig")
        if self.config.use_ref_policy and ref_log_probs is None:
            raise ValueError("use_ref_policy=True requires ref_log_probs in batch")

        use_modified_subtb = self.config.loss_type == "modified_subtb" and eos_logprobs is not None
        if self.config.loss_type == "modified_subtb" and eos_logprobs is None:
            warnings.warn(
                "modified_subtb requires eos_logprobs; falling back to vanilla TB",
                UserWarning,
                stacklevel=2,
            )

        if use_modified_subtb:
            assert eos_logprobs is not None  # guaranteed by use_modified_subtb check
            loss_tensor = self._compute_modified_subtb(
                log_probs_2d, mask_2d, log_reward, eos_logprobs
            )
        else:
            loss_tensor = self._compute_vanilla_tb(log_probs_2d, mask_2d, log_reward, ref_log_probs)

        raw_tb_loss = loss_tensor.item()

        entropy_term = self._compute_entropy_term(entropy, mask_2d)
        if self.config.entropy_coef > 0 and entropy_term is not None:
            loss_tensor = loss_tensor - self.config.entropy_coef * entropy_term

        metrics = {
            "loss": loss_tensor.item(),
            "tb_loss": raw_tb_loss,
            "entropy": entropy_term.item() if entropy_term is not None else 0.0,
            "logZ": self.logZ.item(),
        }

        return loss_tensor, metrics

    def _compute_vanilla_tb(
        self,
        log_probs: Tensor,
        mask: Tensor,
        log_reward: Tensor,
        ref_log_probs: Tensor | None,
    ) -> Tensor:
        """Vanilla Trajectory Balance loss."""
        mask_f = mask.float()
        masked_logprobs = (log_probs * mask_f).sum(dim=-1)  # [B]

        # reference policy correction
        if ref_log_probs is not None:
            if ref_log_probs.dim() == 1:
                masked_ref = ref_log_probs
            else:
                masked_ref = (ref_log_probs * mask_f).sum(dim=-1)
            if self.config.ref_weight != 1.0:
                masked_ref = masked_ref * self.config.ref_weight
            masked_logprobs = masked_logprobs - masked_ref

        # length normalization
        if self.config.normalize_by_length:
            lengths = mask_f.sum(dim=-1).clamp_min(1.0)
            masked_logprobs = masked_logprobs / lengths

        # handle NaN/inf in log_reward
        safe_log_reward = torch.where(
            torch.isfinite(log_reward),
            log_reward,
            torch.full_like(log_reward, -self.config.tb_max_residual),
        )

        # TB: (logZ + log_pi - log_R)^2
        residual = self.logZ + masked_logprobs - safe_log_reward
        residual = residual.clamp(-self.config.tb_max_residual, self.config.tb_max_residual)

        return (residual**2).mean()

    def _compute_modified_subtb(
        self,
        log_probs: Tensor,
        mask: Tensor,
        log_reward: Tensor,
        eos_logprobs: Tensor,
    ) -> Tensor:
        """Modified Sub-Trajectory Balance loss (from gfn-lm-tuning)."""
        B, T = log_probs.shape
        device = log_probs.device
        dtype = log_probs.dtype
        subtb_lambda = self.config.subtb_lambda
        max_residual = self.config.tb_max_residual

        mask_f = mask.float()

        # broadcast log_reward to [B, T] for safe operations
        if log_reward.dim() == 1:
            log_reward_2d = log_reward.unsqueeze(1).expand(B, T)
        else:
            log_reward_2d = log_reward

        safe_log_rewards = torch.where(
            torch.isfinite(log_reward_2d),
            log_reward_2d,
            torch.full_like(log_reward_2d, -max_residual),
        )

        if T == 1:
            traj_reward = safe_log_rewards[:, 0:1]
            delta = log_probs - eos_logprobs + traj_reward
        else:
            eos_next = torch.cat([eos_logprobs[:, 1:], eos_logprobs[:, -1:]], dim=1)
            delta = log_probs - eos_logprobs + eos_next

            # find last valid position per sample
            valid = mask_f > 0
            idxs = torch.arange(T, device=device, dtype=torch.long)
            last_valid_idx = torch.where(
                valid.any(dim=1),
                (valid.float() * idxs).long().max(dim=1).values,
                torch.zeros(B, device=device, dtype=torch.long),
            )

            # anchor final position to reward
            traj_reward = safe_log_rewards.gather(1, last_valid_idx.unsqueeze(1)).squeeze(1)
            eos_next_at_last = eos_next.gather(1, last_valid_idx.unsqueeze(1)).squeeze(1)
            correction = traj_reward - eos_next_at_last

            correction_tensor = torch.zeros_like(delta)
            correction_tensor.scatter_(1, last_valid_idx.unsqueeze(1), correction.unsqueeze(1))
            delta = delta + correction_tensor

        # zero masked positions
        delta = torch.where(mask_f > 0, delta, torch.zeros_like(delta))

        # cumulative sum
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

            residual = torch.where(valid_mask > 0, residual, torch.zeros_like(residual))

            weight = subtb_lambda ** (subtraj_len - 1)
            batch_loss = batch_loss + weight * (residual**2).sum()
            total_weight = total_weight + weight * valid_mask.sum()

        return batch_loss / total_weight.clamp(min=1.0)

    def _compute_entropy_term(
        self,
        entropy: Tensor | None,
        mask: Tensor,
    ) -> Tensor | None:
        """Compute mean entropy for bonus."""
        if entropy is None or entropy.numel() == 0:
            return None

        # sanitize NaN
        if torch.isnan(entropy).any():
            entropy = torch.nan_to_num(entropy, nan=0.0)

        if entropy.dim() == 1:
            return entropy.mean()

        mask_bool = mask.bool() if mask.dtype != torch.bool else mask
        ent_masked = entropy.masked_fill(~mask_bool, 0.0)
        denom = mask_bool.sum(dim=1).clamp_min(1).float()
        ent_per_traj = ent_masked.sum(dim=1) / denom
        return ent_per_traj.mean()

    def state_dict(self) -> dict[str, Any]:  # type: ignore[override]
        """Serialize objective state."""
        return {
            "logZ": self.logZ.item(),
            "config": {
                "beta": self.config.beta,
                "loss_type": self.config.loss_type,
                "subtb_lambda": self.config.subtb_lambda,
                "tb_max_residual": self.config.tb_max_residual,
                "logZ_init": self.config.logZ_init,
                "use_ref_policy": self.config.use_ref_policy,
                "ref_weight": self.config.ref_weight,
                "normalize_by_length": self.config.normalize_by_length,
                "entropy_coef": self.config.entropy_coef,
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:  # type: ignore[override]
        """Restore objective state."""
        with torch.no_grad():
            self.logZ.fill_(state["logZ"])
