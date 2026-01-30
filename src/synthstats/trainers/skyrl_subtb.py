"""SubTB (trajectory balance) trainer for GFlowNet training.

SkyRL-compatible but standalone - can be used without SkyRL installed.
Loss functions are registered with SkyRL for future BasePPOExp integration.

Expects batches with keys:
- log_probs: Tensor [B, T] or [B] (if pre-summed)
- log_reward: Tensor [B]
- loss_mask: optional Tensor [B, T] (bool/int/float), True/1 for valid steps
- entropy: optional Tensor [B] or [B, T]
- eos_logprobs: optional Tensor [B, T] for modified SubTB loss

Design decisions:
1. Optimizer is caller-managed: train_step() can optionally backward+step
2. log_reward is detached: no gradient flow to reward function
3. Uses subtb_loss from training/losses (which registers with SkyRL if available)
4. Supports both vanilla TB and modified SubTB (with EOS logprobs)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn

from synthstats.training.losses import subtb_loss
from synthstats.training.losses.trajectory_balance import compute_modified_subtb_loss


@dataclass
class SubTBConfig:
    """Configuration for SubTB/TB training."""

    beta: float = 0.1
    entropy_coef: float = 0.01
    loss_type: str = "tb"  # "tb" or "modified_subtb"
    subtb_lambda: float = 0.9
    tb_max_residual: float = 100.0
    # Optional TB variants
    use_ref_policy: bool = False
    ref_weight: float = 1.0
    normalize_by_length: bool = False
    allow_mismatched_tokenizer: bool = False
    # LogZ learning parameters
    logZ_init: float = 0.0
    lr_logZ: float = 0.001  # 10x base LR recommended (if base lr is 0.0001)
    # Behavior policy temperatures
    pf_temp_low: float = 0.5
    pf_temp_high: float = 2.0
    # Reward temperature schedule
    reward_temp_start: float = 1.0
    reward_temp_end: float = 0.7
    reward_temp_horizon: int = 200
    # Gradient clipping
    max_grad_norm: float | None = None  # e.g., 1.0 for clipping


class SkyRLSubTBTrainer:
    """SkyRL-compatible trainer for SubTB loss.

    This trainer wraps the core subtb_loss function to provide a SkyRL-compatible
    interface with optional optimizer management.

    Args:
        config: SubTBConfig with hyperparameters
        device: Device string ("cpu", "cuda", etc.)
        optimizer: Optional optimizer. If provided, train_step() will call
                   backward() and step(). If None, caller manages backward.

    Example (with optimizer):
        >>> trainer = SkyRLSubTBTrainer(device="cuda")
        >>> optimizer = torch.optim.AdamW([
        ...     {'params': model.parameters(), 'lr': 0.0001},
        ...     {'params': [trainer.logZ], 'lr': 0.001}
        ... ])
        >>> trainer.optimizer = optimizer
        >>> result = trainer.train_step(batch)  # backward+step called internally

    Example (caller manages backward):
        >>> trainer = SkyRLSubTBTrainer()
        >>> result = trainer.train_step(batch)
        >>> # result["loss_tensor"].backward()  # if loss_tensor returned
        >>> optimizer.step()
    """

    def __init__(
        self,
        config: SubTBConfig | None = None,
        device: str = "cpu",
        optimizer: torch.optim.Optimizer | None = None,
    ):
        self.config = config or SubTBConfig()
        self.device = device
        self.optimizer = optimizer

        # learnable log partition function
        self.logZ = nn.Parameter(torch.tensor(self.config.logZ_init, device=device))

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Compute SubTB loss and optionally update parameters.

        Expected batch keys:
          - log_probs: Tensor [B, T] or [B]
          - log_reward: Tensor [B], log-transformed rewards
          - loss_mask: optional Tensor [B, T] (bool/int/float), True/1 for valid steps
          - entropy: optional Tensor [B] or [B, T]
          - ref_log_probs: optional Tensor [B, T] or [B]
          - eos_logprobs: optional Tensor [B, T], EOS log probs for modified SubTB

        Returns:
            dict with keys:
              - loss: float, total loss for logging
              - tb_loss: float, trajectory balance component
              - entropy: float, mean entropy (0.0 if not provided)
              - logZ: float, current logZ value

        Note: If optimizer is set, this calls backward() and step().
              Otherwise, caller is responsible for calling backward().
        """
        # move tensors to device
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)

        log_probs: torch.Tensor = batch["log_probs"]
        log_reward: torch.Tensor = batch["log_reward"]
        mask: torch.Tensor | None = batch.get("loss_mask")
        entropy: torch.Tensor | None = batch.get("entropy")
        ref_log_probs: torch.Tensor | None = batch.get("ref_log_probs")
        eos_logprobs: torch.Tensor | None = batch.get("eos_logprobs")

        # ensure logZ is on correct device (compare by type to avoid cuda:0 != cuda)
        if self.logZ.device.type != torch.device(self.device).type:
            self.logZ = nn.Parameter(self.logZ.to(self.device))

        # validate log_probs shape and prepare for subtb_loss (expects 2D)
        if log_probs.dim() == 1:
            # pre-summed: reshape to [B, 1] for subtb_loss
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

        # choose loss function based on config
        use_modified_subtb = self.config.loss_type == "modified_subtb" and eos_logprobs is not None

        if self.config.loss_type == "modified_subtb" and eos_logprobs is None:
            import warnings

            warnings.warn(
                "modified_subtb requires eos_logprobs; falling back to vanilla TB",
                UserWarning,
                stacklevel=2,
            )

        if use_modified_subtb:
            assert eos_logprobs is not None  # guaranteed by use_modified_subtb check
            loss_config = {
                "logZ": self.logZ,
                "subtb_lambda": self.config.subtb_lambda,
                "tb_max_residual": self.config.tb_max_residual,
            }

            class ConfigProxy:  # noqa: B903
                def __init__(self, base_config: dict, eos: torch.Tensor) -> None:
                    self._base = base_config
                    self._eos_logprobs = eos

                def get(self, key: str, default: Any = None) -> Any:
                    return self._base.get(key, default)

                def __getattr__(self, name: str) -> Any:
                    if name.startswith("_"):
                        return object.__getattribute__(self, name)
                    return self._base.get(name)

            config_proxy = ConfigProxy(loss_config, eos_logprobs)

            B, T = log_probs_2d.shape
            log_reward_broadcast = log_reward.unsqueeze(1).expand(B, T)

            loss_tensor, _ = compute_modified_subtb_loss(
                log_probs_2d,
                log_probs_2d.detach(),
                log_reward_broadcast,
                config_proxy,
                loss_mask=mask_2d,
            )
        else:
            loss_tensor = subtb_loss(
                log_probs_2d,
                mask_2d,
                log_reward,
                self.logZ,
                ref_log_probs=ref_log_probs,
                ref_weight=self.config.ref_weight,
                normalize_by_length=self.config.normalize_by_length,
                max_residual=self.config.tb_max_residual,
            )

        # compute raw TB loss for logging (before entropy bonus)
        raw_tb_loss = loss_tensor.item()

        # compute entropy term if provided (keep as tensor to preserve gradients)
        entropy_term = None
        if entropy is not None:
            # guard against empty entropy tensor
            if entropy.numel() == 0:
                entropy_term = torch.tensor(0.0, device=self.device)
            else:
                # sanitize any NaN values that slipped through
                if torch.isnan(entropy).any():
                    entropy = torch.nan_to_num(entropy, nan=0.0)

                if entropy.dim() == 1:
                    entropy_term = entropy.mean()
                elif entropy.dim() == 2:
                    if mask is not None:
                        mask_bool = mask.bool() if mask.dtype != torch.bool else mask
                        ent_masked = entropy.masked_fill(~mask_bool, 0.0)
                        denom = mask_bool.sum(dim=1).clamp_min(1).float()
                        ent_per_traj = ent_masked.sum(dim=1) / denom
                        entropy_term = ent_per_traj.mean()
                    else:
                        entropy_term = entropy.mean()

        # apply entropy bonus (subtract to encourage exploration)
        if self.config.entropy_coef > 0 and entropy_term is not None:
            loss_tensor = loss_tensor - self.config.entropy_coef * entropy_term

        # compute total loss for logging (after entropy bonus)
        total_loss_val = loss_tensor.item()

        # do backward and step if optimizer is set
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
            loss_tensor.backward()
            # gradient clipping if configured
            if self.config.max_grad_norm is not None:
                params = [
                    p
                    for group in self.optimizer.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
                if params:
                    torch.nn.utils.clip_grad_norm_(params, self.config.max_grad_norm)
            self.optimizer.step()

        result = {
            "loss": total_loss_val,
            "tb_loss": raw_tb_loss,
            "entropy": float(entropy_term.detach().item()) if entropy_term is not None else 0.0,
            "logZ": float(self.logZ.item()),
        }

        # return loss_tensor for caller-managed backward when optimizer is None
        if self.optimizer is None:
            result["loss_tensor"] = loss_tensor

        return result

    def state_dict(self) -> dict[str, Any]:
        """Serialize trainer state for checkpointing."""
        return {
            "logZ": self.logZ.detach().cpu().item(),
            "config": asdict(self.config),
            "device": self.device,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore trainer state from checkpoint (logZ only)."""
        with torch.no_grad():
            self.logZ.fill_(state["logZ"])
