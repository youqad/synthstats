"""SubTB (Sub-Trajectory Balance) objective for GFlowNet training.

Pure loss computation as nn.Module:
- Contains logZ as learnable Parameter
- forward() computes loss and metrics
- No optimizer, no stepping (that's the Learner's job)

Supports:
- Vanilla TB: L = (logZ + sum(log_pi) - log_R)^2
- Modified SubTB: Lambda-weighted sub-trajectory flow matching
- Action-boundary SubTB: TB anchor + action-level SubTB regularizer
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from synthstats.core.constants import SUBTB_LAMBDA_DEFAULT, TB_MAX_RESIDUAL_DEFAULT
from synthstats.core.math_utils import sanitize_finite
from synthstats.train.objectives.losses import (
    compute_action_subtb_core,
    compute_modified_subtb_core,
    subtb_loss,
)


@dataclass
class SubTBConfig:
    """Configuration for SubTB objective."""

    loss_type: str = "tb"  # "tb" | "modified_subtb" | "ab_subtb"
    subtb_lambda: float = SUBTB_LAMBDA_DEFAULT
    tb_max_residual: float = TB_MAX_RESIDUAL_DEFAULT
    logZ_init: float = 0.0

    # reference policy (KL regularization)
    use_ref_policy: bool = False
    ref_weight: float = 1.0
    normalize_by_length: bool = False

    # entropy bonus
    entropy_coef: float = 0.01

    # action-boundary SubTB (AB-SubTB)
    ab_subtb_alpha: float = 0.1
    use_boundary_critic: bool = False
    boundary_critic_hidden_dim: int = 32
    boundary_critic_loss_coef: float = 1.0


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

        self._boundary_feature_dim = 4
        if self.config.use_boundary_critic:
            hidden = max(4, int(self.config.boundary_critic_hidden_dim))
            self.boundary_critic: nn.Module | None = nn.Sequential(
                nn.Linear(self._boundary_feature_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 1),
            ).to(self._device)
        else:
            self.boundary_critic = None

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

        # ensure 2D for loss functions
        if log_probs.dim() == 1:
            log_probs_2d = log_probs.unsqueeze(1)
            mask_2d = torch.ones_like(log_probs_2d, dtype=torch.bool)
        elif log_probs.dim() == 2:
            log_probs_2d = log_probs
            mask_2d = mask if mask is not None else torch.ones_like(log_probs_2d, dtype=torch.bool)
        else:
            raise ValueError(f"log_probs must be 1D [B] or 2D [B, T], got shape {log_probs.shape}")

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

        ab_subtb_loss: Tensor | None = None
        ab_critic_loss: Tensor | None = None

        if self.config.loss_type == "ab_subtb":
            tb_tensor = self._compute_vanilla_tb(log_probs_2d, mask_2d, log_reward, ref_log_probs)
            ab_subtb_loss, ab_critic_loss = self._compute_action_boundary_subtb(
                log_probs=log_probs_2d,
                mask=mask_2d,
                log_reward=log_reward,
            )
            loss_tensor = tb_tensor + self.config.ab_subtb_alpha * ab_subtb_loss
            if ab_critic_loss is not None:
                loss_tensor = loss_tensor + self.config.boundary_critic_loss_coef * ab_critic_loss
            raw_tb_loss = tb_tensor.item()
        elif use_modified_subtb:
            assert eos_logprobs is not None
            loss_tensor = self._compute_modified_subtb(
                log_probs_2d, mask_2d, log_reward, eos_logprobs
            )
            raw_tb_loss = loss_tensor.item()
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
        if ab_subtb_loss is not None:
            metrics["ab_subtb_loss"] = ab_subtb_loss.item()
        if ab_critic_loss is not None:
            metrics["ab_critic_loss"] = ab_critic_loss.item()

        return loss_tensor, metrics

    def _compute_vanilla_tb(
        self,
        log_probs: Tensor,
        mask: Tensor,
        log_reward: Tensor,
        ref_log_probs: Tensor | None,
    ) -> Tensor:
        """Vanilla Trajectory Balance loss. Delegates to canonical subtb_loss()."""
        return subtb_loss(
            log_probs=log_probs,
            loss_mask=mask,
            log_rewards=log_reward,
            logZ=self.logZ,
            max_residual=self.config.tb_max_residual,
            ref_log_probs=ref_log_probs,
            ref_weight=self.config.ref_weight,
            normalize_by_length=self.config.normalize_by_length,
        )

    def _compute_modified_subtb(
        self,
        log_probs: Tensor,
        mask: Tensor,
        log_reward: Tensor,
        eos_logprobs: Tensor,
    ) -> Tensor:
        """Modified SubTB loss. Delegates to canonical compute_modified_subtb_core()."""
        return compute_modified_subtb_core(
            log_probs=log_probs,
            eos_logprobs=eos_logprobs,
            log_rewards=log_reward,
            mask=mask,
            subtb_lambda=self.config.subtb_lambda,
            max_residual=self.config.tb_max_residual,
        )

    def _compute_action_boundary_subtb(
        self,
        *,
        log_probs: Tensor,
        mask: Tensor,
        log_reward: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        """Compute action-boundary SubTB loss and optional critic loss."""
        action_mask = mask.bool()
        B, T = log_probs.shape
        safe_log_reward = sanitize_finite(log_reward, -self.config.tb_max_residual)

        start_u = self.logZ.detach().expand(B)
        terminal_u = safe_log_reward

        if T == 0:
            zero = (log_probs.sum() + start_u.sum() + terminal_u.sum()) * 0.0
            if not zero.requires_grad:
                zero = zero.requires_grad_()
            return zero, None

        critic_loss: Tensor | None = None
        if T <= 1:
            u_policy = torch.stack([start_u, terminal_u], dim=1)
        elif self.boundary_critic is not None:
            features, boundary_valid = self._build_boundary_features(log_probs, action_mask)
            pred = self.boundary_critic(features.reshape(-1, features.shape[-1])).reshape(B, T - 1)
            pred = torch.where(boundary_valid, pred, torch.zeros_like(pred))

            # actor loss uses detached potentials
            u_policy = torch.cat(
                [start_u.unsqueeze(1), pred.detach(), terminal_u.unsqueeze(1)],
                dim=1,
            )

            # critic loss uses detached policy logprobs
            u_critic = torch.cat(
                [
                    start_u.unsqueeze(1),
                    pred,
                    terminal_u.unsqueeze(1),
                ],
                dim=1,
            )
            critic_loss = compute_action_subtb_core(
                log_pf=log_probs.detach(),
                u=u_critic,
                mask=action_mask,
                subtb_lambda=self.config.subtb_lambda,
                max_residual=self.config.tb_max_residual,
            )
        else:
            u_policy = self._uniform_boundary_potentials(
                start_u=start_u,
                terminal_u=terminal_u,
                action_mask=action_mask,
            )

        policy_loss = compute_action_subtb_core(
            log_pf=log_probs,
            u=u_policy,
            mask=action_mask,
            subtb_lambda=self.config.subtb_lambda,
            max_residual=self.config.tb_max_residual,
        )
        return policy_loss, critic_loss

    def _uniform_boundary_potentials(
        self,
        *,
        start_u: Tensor,
        terminal_u: Tensor,
        action_mask: Tensor,
    ) -> Tensor:
        """Simple boundary potential baseline: linear interpolation by action index."""
        B, T = action_mask.shape
        if T <= 1:
            return torch.stack([start_u, terminal_u], dim=1)

        lengths = action_mask.float().sum(dim=1)  # [B]
        boundary_idx = torch.arange(1, T, device=start_u.device, dtype=start_u.dtype).unsqueeze(0)
        denom = lengths.clamp_min(1.0).unsqueeze(1)
        frac = boundary_idx / denom
        interior = start_u.unsqueeze(1) + frac * (terminal_u.unsqueeze(1) - start_u.unsqueeze(1))

        valid = boundary_idx < lengths.unsqueeze(1)
        interior = torch.where(valid, interior, torch.zeros_like(interior))

        return torch.cat([start_u.unsqueeze(1), interior, terminal_u.unsqueeze(1)], dim=1)

    def _build_boundary_features(
        self,
        log_probs: Tensor,
        action_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Build lightweight numeric features for interior boundary potentials."""
        B, T = log_probs.shape
        if T <= 1:
            empty = torch.zeros(B, 0, self._boundary_feature_dim, device=log_probs.device)
            empty_mask = torch.zeros(B, 0, dtype=torch.bool, device=log_probs.device)
            return empty, empty_mask

        mask_f = action_mask.float()
        safe_log_probs = sanitize_finite(log_probs, -self.config.tb_max_residual)
        cumsum = (safe_log_probs * mask_f).cumsum(dim=1)

        # interior boundaries are after action i for i in [0, T-2]
        prefix_sum = cumsum[:, :-1]
        prev_log_pf = safe_log_probs[:, :-1]
        next_log_pf = safe_log_probs[:, 1:]

        step_idx = torch.arange(1, T, device=log_probs.device, dtype=log_probs.dtype).unsqueeze(0)
        step_frac = step_idx / max(T, 1)

        features = torch.stack(
            [prefix_sum, prev_log_pf, next_log_pf, step_frac.expand(B, -1)],
            dim=-1,
        )

        lengths = action_mask.float().sum(dim=1)  # [B]
        boundary_valid = step_idx < lengths.unsqueeze(1)
        return features, boundary_valid

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
        state: dict[str, Any] = {
            "logZ": self.logZ.item(),
            "config": {
                "loss_type": self.config.loss_type,
                "subtb_lambda": self.config.subtb_lambda,
                "tb_max_residual": self.config.tb_max_residual,
                "logZ_init": self.config.logZ_init,
                "use_ref_policy": self.config.use_ref_policy,
                "ref_weight": self.config.ref_weight,
                "normalize_by_length": self.config.normalize_by_length,
                "entropy_coef": self.config.entropy_coef,
                "ab_subtb_alpha": self.config.ab_subtb_alpha,
                "use_boundary_critic": self.config.use_boundary_critic,
                "boundary_critic_hidden_dim": self.config.boundary_critic_hidden_dim,
                "boundary_critic_loss_coef": self.config.boundary_critic_loss_coef,
            },
        }
        if self.boundary_critic is not None:
            state["boundary_critic"] = self.boundary_critic.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:  # type: ignore[override]
        """Restore objective state."""
        with torch.no_grad():
            self.logZ.fill_(state["logZ"])
        if self.boundary_critic is not None and "boundary_critic" in state:
            self.boundary_critic.load_state_dict(state["boundary_critic"])
