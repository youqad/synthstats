"""Backward-compatible re-export from the old module path.

The Tinker adapter moved to synthstats.integrations.tinker.adapter.
This shim keeps Hydra configs (configs/trainer/tinker.yaml, etc.)
that reference the old path working.
"""

from synthstats.integrations.tinker.adapter import (  # noqa: F401
    MockTinkerClient,
    MockTinkerTrainingClient,
    MockTokenizer,
    TinkerConfig,
    TinkerEnvProtocol,
    TinkerOptionalDependencyError,
    TinkerPolicy,
    TinkerTrainer,
    TurnBoundary,
    _build_turn_mask,
    is_tinker_available,
    require_tinker,
    trajectories_to_tinker_batch,
)
