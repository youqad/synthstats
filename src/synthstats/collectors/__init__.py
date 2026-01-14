"""Trajectory collectors for SynthStats.

DEPRECATED: The collectors module has been archived as part of the native SkyRL
integration refactor (January 2026). SkyRL provides native collector functionality.

For custom collection needs, use:
- SkyRL's native generators and collectors
- synthstats.training.TBTrainer for TB-specific training
- synthstats.envs.BoxingEnv for environment interaction

Archived code is available at: _archive/skyrl_integration_2026-01/src/collectors/
"""

import warnings

warnings.warn(
    "synthstats.collectors is deprecated. Use SkyRL's native collectors or "
    "synthstats.training.TBTrainer for GFlowNet training.",
    DeprecationWarning,
    stacklevel=2,
)

__all__: list[str] = []
