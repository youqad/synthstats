"""Temperature and learning rate schedulers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class RewardTemperatureSchedule:
    """Anneals reward temperature from start to end over horizon steps."""

    start: float = 1.0
    end: float = 0.1
    horizon: int = 1000
    mode: Literal["linear", "cosine"] = "linear"

    def get(self, step: int) -> float:
        if self.horizon <= 0:
            return self.end

        t = min(1.0, step / self.horizon)

        if self.mode == "cosine":
            import math

            t = 0.5 * (1 - math.cos(math.pi * t))

        return self.start + (self.end - self.start) * t

    def scale_reward(self, log_reward: float, step: int) -> float:
        temp = self.get(step)
        return log_reward / temp if temp > 0 else log_reward


@dataclass
class ExplorationTemperatureConfig:
    """Per-trajectory temperature perturbation for diversity."""

    pf_temp_low: float = 0.5
    pf_temp_high: float = 2.0
    perturb_prob: float = 0.5

    def sample_temperature(self) -> float:
        """Return 1.0 or a perturbed value with probability perturb_prob."""
        if random.random() < self.perturb_prob:
            return random.uniform(self.pf_temp_low, self.pf_temp_high)
        return 1.0


@dataclass
class BehaviorMixConfig:
    """Mixing weights for on-policy, replay, and perturbed trajectories."""

    on_policy: float = 0.25
    replay: float = 0.25
    perturbed: float = 0.50

    def __post_init__(self) -> None:
        total = self.on_policy + self.replay + self.perturbed
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Behavior mix must sum to 1.0, got {total}")

    def sample_source(self) -> Literal["on_policy", "replay", "perturbed"]:
        r = random.random()
        if r < self.on_policy:
            return "on_policy"
        if r < self.on_policy + self.replay:
            return "replay"
        return "perturbed"


@dataclass
class LogZLearningRateConfig:
    """LogZ learning rate, typically 100x model LR per gfn-lm-tuning."""

    multiplier: float = 100.0
    absolute: float | None = None

    def get(self, model_lr: float) -> float:
        if self.absolute is not None:
            return self.absolute
        return model_lr * self.multiplier


def create_warmup_scheduler(
    optimizer: Any,  # torch.optim.Optimizer
    warmup_steps: int,
    total_steps: int,
) -> Any:  # torch.optim.lr_scheduler.LambdaLR
    """Linear warmup then linear decay to zero."""
    import torch.optim.lr_scheduler as lr_scheduler

    def get_lr_mult_at_step(step: int) -> float:
        if step < warmup_steps:
            return min(step / warmup_steps, 1.0) if warmup_steps > 0 else 1.0
        if total_steps <= warmup_steps:
            return 1.0
        return max((total_steps - step) / (total_steps - warmup_steps), 0.0)

    return lr_scheduler.LambdaLR(optimizer, get_lr_mult_at_step)
