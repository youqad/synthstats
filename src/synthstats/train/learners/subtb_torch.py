"""PyTorch-based SubTB learner. Owns optimizer, wraps SubTBObjective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from synthstats.core.constants import LOGZ_LR_DEFAULT
from synthstats.train.objectives.subtb import SubTBConfig, SubTBObjective


@dataclass
class SubTBTorchConfig:
    policy_lr: float = 1e-5
    logZ_lr: float = LOGZ_LR_DEFAULT
    weight_decay: float = 0.0
    max_grad_norm: float | None = 1.0

    amp_enabled: bool = False


class SubTBTorchLearner:
    """Owns optimizer and handles gradient updates via SubTBObjective."""

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

        self.optimizer = self._build_optimizer()
        self._scaler = torch.amp.GradScaler("cuda") if self.config.amp_enabled else None

    def _build_optimizer(self) -> torch.optim.Optimizer:
        param_groups = []

        param_groups.append(
            {
                "params": [self.objective.logZ],
                "lr": self.config.logZ_lr,
                "weight_decay": 0.0,
            }
        )

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
        params = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    params.append(p)
        if params and self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(params, self.config.max_grad_norm)

    @property
    def logZ(self) -> float:
        return self.objective.logZ.item()

    def state_dict(self) -> dict[str, Any]:
        state = {
            "objective": self.objective.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self._scaler is not None:
            state["scaler"] = self._scaler.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
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
