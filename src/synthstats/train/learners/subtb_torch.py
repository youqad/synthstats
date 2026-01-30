"""PyTorch-based SubTB learner.

Owns optimizer and handles gradient updates for policy + logZ parameters.
Wraps SubTBObjective and implements the Learner protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from synthstats.train.objectives.subtb import SubTBConfig, SubTBObjective


@dataclass
class SubTBTorchConfig:
    """Configuration for SubTBTorchLearner."""

    # optimizer
    policy_lr: float = 1e-5
    logZ_lr: float = 1e-1
    weight_decay: float = 0.0
    max_grad_norm: float | None = 1.0

    # precision
    amp_enabled: bool = False


class SubTBTorchLearner:
    """PyTorch-based SubTB learner.

    Owns the optimizer and handles gradient updates. Wraps SubTBObjective
    for loss computation.

    Args:
        objective: SubTBObjective for loss computation
        policy: Policy module with trainable parameters (optional)
        config: SubTBTorchConfig with optimizer settings
        device: Device for training

    Example:
        >>> objective = SubTBObjective(SubTBConfig())
        >>> learner = SubTBTorchLearner(objective, policy, config)
        >>> metrics = learner.update(batch)
    """

    def __init__(
        self,
        objective: SubTBObjective,
        policy: nn.Module | None = None,
        config: SubTBTorchConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.objective = objective
        self.policy = policy
        self.config = config or SubTBTorchConfig()
        self._device = torch.device(device)

        # build optimizer
        self.optimizer = self._build_optimizer()

        # AMP scaler
        self._scaler = torch.amp.GradScaler("cuda") if self.config.amp_enabled else None

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer with separate parameter groups."""
        param_groups = []

        # logZ parameters (from objective)
        param_groups.append(
            {
                "params": [self.objective.logZ],
                "lr": self.config.logZ_lr,
                "weight_decay": 0.0,  # no weight decay on logZ
            }
        )

        # policy parameters (if trainable)
        if self.policy is not None:
            policy_params = [p for p in self.policy.parameters() if p.requires_grad]
            if policy_params:
                param_groups.append(
                    {
                        "params": policy_params,
                        "lr": self.config.policy_lr,
                        "weight_decay": self.config.weight_decay,
                    }
                )

        return torch.optim.AdamW(param_groups)

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        """Update parameters from batch.

        Args:
            batch: Dict with keys log_probs, log_reward, mask, entropy, etc.

        Returns:
            Dict with metrics (loss, tb_loss, entropy, logZ)
        """
        self.optimizer.zero_grad(set_to_none=True)

        if self.config.amp_enabled and self._scaler is not None:
            with torch.amp.autocast("cuda"):
                loss, metrics = self.objective(batch)
            self._scaler.scale(loss).backward()
            if self.config.max_grad_norm is not None:
                self._scaler.unscale_(self.optimizer)
                self._clip_gradients()
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            loss, metrics = self.objective(batch)
            loss.backward()
            if self.config.max_grad_norm is not None:
                self._clip_gradients()
            self.optimizer.step()

        return metrics

    def _clip_gradients(self) -> None:
        """Clip gradients to max_grad_norm."""
        params = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    params.append(p)
        if params and self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(params, self.config.max_grad_norm)

    @property
    def logZ(self) -> float:
        """Current logZ value."""
        return self.objective.logZ.item()

    def state_dict(self) -> dict[str, Any]:
        """Serialize learner state for checkpointing."""
        state = {
            "objective": self.objective.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self._scaler is not None:
            state["scaler"] = self._scaler.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore learner state from checkpoint."""
        self.objective.load_state_dict(state["objective"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self._scaler is not None and "scaler" in state:
            self._scaler.load_state_dict(state["scaler"])


def create_subtb_learner(
    policy: nn.Module | None = None,
    device: str | torch.device = "cpu",
    objective_config: SubTBConfig | None = None,
    learner_config: SubTBTorchConfig | None = None,
) -> SubTBTorchLearner:
    """Factory function to create SubTBTorchLearner with objective.

    Args:
        policy: Optional policy module with trainable parameters
        device: Device for training
        objective_config: Config for SubTBObjective
        learner_config: Config for SubTBTorchLearner

    Returns:
        Configured SubTBTorchLearner
    """
    objective = SubTBObjective(
        config=objective_config or SubTBConfig(),
        device=device,
    )
    return SubTBTorchLearner(
        objective=objective,
        policy=policy,
        config=learner_config or SubTBTorchConfig(),
        device=device,
    )
