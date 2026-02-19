"""SkyRL-compatible SubTB trainer.

Standalone (works without SkyRL). Delegates loss to SubTBObjective.
Optimizer is caller-managed unless explicitly provided.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from synthstats.core.constants import (
    LOGZ_LR_DEFAULT,
    SUBTB_LAMBDA_DEFAULT,
    TB_MAX_RESIDUAL_DEFAULT,
)
from synthstats.train.objectives.subtb import SubTBConfig as _ObjConfig
from synthstats.train.objectives.subtb import SubTBObjective


@dataclass
class SkyRLSubTBConfig:
    """SubTB/TB hyperparameters."""

    beta: float = 0.1
    entropy_coef: float = 0.01
    loss_type: str = "tb"  # "tb" | "modified_subtb" | "ab_subtb" | "agentic_subtb"
    subtb_lambda: float = SUBTB_LAMBDA_DEFAULT
    tb_max_residual: float = TB_MAX_RESIDUAL_DEFAULT
    ab_subtb_alpha: float = 0.1
    use_boundary_critic: bool = False
    boundary_critic_hidden_dim: int = 32
    boundary_critic_loss_coef: float = 1.0
    # Optional TB variants
    use_ref_policy: bool = False
    ref_weight: float = 1.0
    normalize_by_length: bool = False
    allow_mismatched_tokenizer: bool = False
    # LogZ learning parameters
    logZ_init: float = 0.0
    logZ_lr: float = LOGZ_LR_DEFAULT
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
    """SkyRL-compatible SubTB trainer.

    If optimizer is provided, train_step() calls backward+step internally.
    Otherwise caller manages the backward pass.
    """

    def __init__(
        self,
        config: SkyRLSubTBConfig | None = None,
        device: str = "cpu",
        optimizer: torch.optim.Optimizer | None = None,
    ):
        self.config = config or SkyRLSubTBConfig()
        self.device = device
        self.optimizer = optimizer

        obj_config = _ObjConfig(
            loss_type=self.config.loss_type,
            subtb_lambda=self.config.subtb_lambda,
            tb_max_residual=self.config.tb_max_residual,
            logZ_init=self.config.logZ_init,
            ab_subtb_alpha=self.config.ab_subtb_alpha,
            use_boundary_critic=self.config.use_boundary_critic,
            boundary_critic_hidden_dim=self.config.boundary_critic_hidden_dim,
            boundary_critic_loss_coef=self.config.boundary_critic_loss_coef,
            use_ref_policy=self.config.use_ref_policy,
            ref_weight=self.config.ref_weight,
            normalize_by_length=self.config.normalize_by_length,
            entropy_coef=self.config.entropy_coef,
        )
        self._objective = SubTBObjective(config=obj_config, device=device)

        self.logZ = self._objective.logZ

    def train_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Compute SubTB loss and optionally update parameters."""
        loss_tensor, metrics = self._objective(batch)

        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
            loss_tensor.backward()
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

        result: dict[str, Any] = dict(metrics)

        if self.optimizer is None:
            result["loss_tensor"] = loss_tensor

        return result

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "objective": self._objective.state_dict(),
            "logZ": self.logZ.detach().cpu().item(),
            "config": asdict(self.config),
            "device": self.device,
        }
        if self.optimizer is not None:
            state["optimizer"] = self.optimizer.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        objective_state = state.get("objective")
        if objective_state is not None:
            self._objective.load_state_dict(objective_state)
        elif "logZ" in state:
            # backward-compatible fallback for older checkpoints
            with torch.no_grad():
                self.logZ.fill_(state["logZ"])

        optimizer_state = state.get("optimizer")
        if self.optimizer is not None and optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
