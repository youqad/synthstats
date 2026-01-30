"""Tinker API-based SubTB learner.

Wraps Tinker's training API for SubTB training. Converts trajectories
to Tinker's batch format and calls their training endpoint.

Note: logZ is owned exclusively by TinkerTrainer. This learner accesses
it via trainer.logZ property - no separate logZ here to avoid drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from synthstats.integrations.tinker.adapter import TinkerTrainer


@dataclass
class SubTBTinkerConfig:
    """Configuration for SubTBTinkerLearner."""

    api_key: str | None = None
    model: str = "Qwen/Qwen3-4B"
    lora_rank: int = 32
    learning_rate: float = 1e-5
    logZ_init: float = 0.0


class SubTBTinkerLearner:
    """Wraps TinkerTrainer for API-based SubTB training.

    logZ is owned by the trainer; this learner delegates to trainer.train_step().
    """

    def __init__(
        self,
        config: SubTBTinkerConfig | None = None,
    ) -> None:
        self.config = config or SubTBTinkerConfig()
        self._trainer: TinkerTrainer | None = None

    def _get_trainer(self) -> TinkerTrainer:
        """Get or create TinkerTrainer."""
        if self._trainer is None:
            from synthstats.integrations.tinker.adapter import TinkerConfig, TinkerTrainer

            tinker_config = TinkerConfig(
                model=self.config.model,
                api_key=self.config.api_key,
                lora_rank=self.config.lora_rank,
                learning_rate=self.config.learning_rate,
            )
            self._trainer = TinkerTrainer(
                config=tinker_config,
                logZ_init=self.config.logZ_init,
            )
        return self._trainer

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        """Update parameters from batch.

        For Tinker, the batch should contain prompts/completions rather
        than tokenized log_probs.

        Args:
            batch: Dict with prompts, completions, log_reward

        Returns:
            Dict with metrics (loss, logZ)
        """
        trainer = self._get_trainer()

        # Tinker train step (policy + logZ update via API)
        tinker_metrics = trainer.train_step(batch)

        # return trainer's logZ (single source of truth)
        metrics = {
            "loss": tinker_metrics.get("loss", 0.0),
            "logZ": trainer.logZ.item(),
            "tinker_loss": tinker_metrics.get("loss", 0.0),
        }

        return metrics

    @property
    def logZ(self) -> float:
        """Current logZ value (from trainer)."""
        return self._get_trainer().logZ.item()

    def state_dict(self) -> dict[str, Any]:
        """Serialize learner state."""
        trainer = self._get_trainer()
        return {
            "logZ": trainer.logZ.item(),
            "config": {
                "model": self.config.model,
                "lora_rank": self.config.lora_rank,
                "learning_rate": self.config.learning_rate,
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore learner state."""
        import torch

        trainer = self._get_trainer()
        with torch.no_grad():
            trainer.logZ.fill_(state["logZ"])
