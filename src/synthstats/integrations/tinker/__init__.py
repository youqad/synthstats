"""Tinker API integration.

Provides adapters for using Tinker's training API with SynthStats.

Key components:
- TinkerPolicy: Wraps Tinker.sample() to match HFPolicy interface
- TinkerTrainer: Uses forward_backward_custom() with TB loss
- TinkerConfig: Configuration dataclass
- trajectories_to_tinker_batch: Convert Trajectory objects to Tinker format
"""

from synthstats.integrations.tinker.adapter import (
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

__all__ = [
    "MockTinkerClient",
    "MockTinkerTrainingClient",
    "MockTokenizer",
    "TinkerConfig",
    "TinkerEnvProtocol",
    "TinkerOptionalDependencyError",
    "TinkerPolicy",
    "TinkerTrainer",
    "TurnBoundary",
    "_build_turn_mask",
    "is_tinker_available",
    "require_tinker",
    "trajectories_to_tinker_batch",
]
