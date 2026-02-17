"""Step logger with optional WandB integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepLogger:

    wandb_module: Any = None
    prefix: str = "train/"
    history: list[dict[str, Any]] = field(default_factory=list)

    def log_step(
        self,
        step_idx: int,
        loss: float,
        logZ: float,
        avg_reward: float,
        **extra: Any,
    ) -> None:
        metrics = {
            "loss": loss,
            "logZ": logZ,
            "avg_reward": avg_reward,
            **extra,
        }

        self.history.append(metrics)

        if self.wandb_module is not None:
            wandb_metrics = {f"{self.prefix}{k}": v for k, v in metrics.items()}
            self.wandb_module.log(wandb_metrics, step=step_idx)

    def log_evaluation(
        self,
        step_idx: int,
        eval_reward: float,
        success_rate: float,
        **extra: Any,
    ) -> None:
        metrics = {
            "eval_reward": eval_reward,
            "success_rate": success_rate,
            **extra,
        }

        if self.wandb_module is not None:
            wandb_metrics = {f"eval/{k}": v for k, v in metrics.items()}
            self.wandb_module.log(wandb_metrics, step=step_idx)

    def log_episode(
        self,
        episode_idx: int,
        reward: float,
        length: int,
        success: bool,
        **extra: Any,
    ) -> None:
        metrics = {
            "reward": reward,
            "length": length,
            "success": int(success),
            **extra,
        }

        if self.wandb_module is not None:
            wandb_metrics = {f"episode/{k}": v for k, v in metrics.items()}
            self.wandb_module.log(wandb_metrics, step=episode_idx)

    def get_summary(self) -> dict[str, float]:
        if not self.history:
            return {}

        n = len(self.history)

        summary = {}

        losses = [h.get("loss", 0) for h in self.history if "loss" in h]
        if losses:
            summary["avg_loss"] = sum(losses) / len(losses)

        rewards = [h.get("avg_reward", 0) for h in self.history if "avg_reward" in h]
        if rewards:
            summary["avg_reward"] = sum(rewards) / len(rewards)

        logzs = [h.get("logZ", 0) for h in self.history if "logZ" in h]
        if logzs:
            summary["final_logZ"] = logzs[-1]

        summary["num_steps"] = n

        return summary
