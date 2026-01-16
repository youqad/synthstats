"""Training components for SynthStats."""

from synthstats.training.losses import subtb_loss
from synthstats.training.tb_trainer import (
    SKYRL_AVAILABLE,
    LogZModule,
    TBTrainer,
    TBTrainerMixin,
)
from synthstats.training.trainer import Trainer, TrainerConfig, TrainMetrics

__all__ = [
    # core trainer
    "Trainer",
    "TrainerConfig",
    "TrainMetrics",
    # losses
    "subtb_loss",
    # TB/SkyRL integration
    "TBTrainer",
    "TBTrainerMixin",
    "LogZModule",
    "SKYRL_AVAILABLE",
]
