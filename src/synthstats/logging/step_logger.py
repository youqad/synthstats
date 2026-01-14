"""Step logger for training metrics.

Provides structured logging with optional WandB integration.
Works without WandB installed for local development.

Usage:
    # without wandb
    logger = StepLogger()
    logger.log_step(step_idx=1, loss=0.5, logZ=0.1, avg_reward=1.0)

    # with wandb
    import wandb
    wandb.init(project="synthstats")
    logger = StepLogger(wandb_module=wandb)
    logger.log_step(step_idx=1, loss=0.5, logZ=0.1, avg_reward=1.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepLogger:
    """Structured logger for training steps.

    Tracks metrics history and optionally logs to WandB.

    Args:
        wandb_module: Optional wandb module for remote logging
        prefix: Prefix for wandb metric names (e.g., "train/")
    """

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
        """Log a training step.

        Args:
            step_idx: Current training step
            loss: Training loss
            logZ: Current logZ value
            avg_reward: Average reward for this step
            **extra: Additional metrics to log
        """
        metrics = {
            "loss": loss,
            "logZ": logZ,
            "avg_reward": avg_reward,
            **extra,
        }

        # store in history
        self.history.append(metrics)

        # log to wandb if available
        if self.wandb_module is not None:
            wandb_metrics = {
                f"{self.prefix}{k}": v
                for k, v in metrics.items()
            }
            self.wandb_module.log(wandb_metrics, step=step_idx)

    def log_evaluation(
        self,
        step_idx: int,
        eval_reward: float,
        success_rate: float,
        **extra: Any,
    ) -> None:
        """Log evaluation metrics.

        Args:
            step_idx: Current training step
            eval_reward: Average evaluation reward
            success_rate: Success rate (0-1)
            **extra: Additional metrics
        """
        metrics = {
            "eval_reward": eval_reward,
            "success_rate": success_rate,
            **extra,
        }

        if self.wandb_module is not None:
            wandb_metrics = {
                f"eval/{k}": v
                for k, v in metrics.items()
            }
            self.wandb_module.log(wandb_metrics, step=step_idx)

    def log_episode(
        self,
        episode_idx: int,
        reward: float,
        length: int,
        success: bool,
        **extra: Any,
    ) -> None:
        """Log per-episode metrics.

        Args:
            episode_idx: Episode index
            reward: Episode reward
            length: Episode length (number of steps)
            success: Whether episode was successful
            **extra: Additional metrics
        """
        metrics = {
            "reward": reward,
            "length": length,
            "success": int(success),
            **extra,
        }

        if self.wandb_module is not None:
            wandb_metrics = {
                f"episode/{k}": v
                for k, v in metrics.items()
            }
            self.wandb_module.log(wandb_metrics, step=episode_idx)

    def get_summary(self) -> dict[str, float]:
        """Get summary statistics from history.

        Returns:
            Dict with avg_loss, avg_reward, etc.
        """
        if not self.history:
            return {}

        n = len(self.history)

        # compute averages for common metrics
        summary = {}

        # loss
        losses = [h.get("loss", 0) for h in self.history if "loss" in h]
        if losses:
            summary["avg_loss"] = sum(losses) / len(losses)

        # reward
        rewards = [h.get("avg_reward", 0) for h in self.history if "avg_reward" in h]
        if rewards:
            summary["avg_reward"] = sum(rewards) / len(rewards)

        # logZ
        logzs = [h.get("logZ", 0) for h in self.history if "logZ" in h]
        if logzs:
            summary["final_logZ"] = logzs[-1]

        summary["num_steps"] = n

        return summary
