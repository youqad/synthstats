"""SkyRL-compatible trainers for SynthStats.

The trainers module now provides the native SkyRL TBTrainer.
For the archived SkyRLSubTBTrainer, see: _archive/skyrl_integration_2026-01/
"""

from synthstats.training.tb_trainer import (
    SKYRL_AVAILABLE,
    LogZModule,
    TBTrainer,
    TBTrainerMixin,
)

__all__ = [
    "TBTrainer",
    "TBTrainerMixin",
    "LogZModule",
    "SKYRL_AVAILABLE",
]
