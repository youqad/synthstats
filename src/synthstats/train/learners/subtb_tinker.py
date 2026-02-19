"""Tinker API-based SubTB learner.

logZ is owned by TinkerTrainer; accessed via trainer.logZ to avoid drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from synthstats.integrations.tinker.adapter import TinkerTrainer


@dataclass
class SubTBTinkerConfig:
    api_key: str | None = None
    base_url: str | None = None
    model: str = "Qwen/Qwen3-4B"
    lora_rank: int = 32
    learning_rate: float = 1e-5
    logZ_init: float = 0.0


class SubTBTinkerLearner:
    """Wraps TinkerTrainer for API-based SubTB training."""

    def __init__(
        self,
        config: SubTBTinkerConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config or SubTBTinkerConfig()
        self.device = device  # unused for API training, accepted for interface parity
        self._trainer: TinkerTrainer | None = None

    def _get_trainer(self) -> TinkerTrainer:
        if self._trainer is None:
            from synthstats.integrations.tinker.adapter import TinkerConfig, TinkerTrainer

            tinker_config = TinkerConfig(
                model=self.config.model,
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                lora_rank=self.config.lora_rank,
                learning_rate=self.config.learning_rate,
            )
            self._trainer = TinkerTrainer(
                config=tinker_config,
                logZ_init=self.config.logZ_init,
            )
        return self._trainer

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        trainer = self._get_trainer()

        tinker_metrics = trainer.train_step(batch)

        metrics = {
            "loss": tinker_metrics.get("loss", 0.0),
            "logZ": trainer.logZ.item(),
            "tinker_loss": tinker_metrics.get("loss", 0.0),
        }

        return metrics

    @property
    def logZ(self) -> float:
        return self._get_trainer().logZ.item()

    def state_dict(self) -> dict[str, Any]:
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
        import torch

        trainer = self._get_trainer()
        with torch.no_grad():
            trainer.logZ.fill_(state["logZ"])
