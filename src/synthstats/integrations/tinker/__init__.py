"""Tinker API adapters."""

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
from synthstats.integrations.tinker.eos_extraction import (
    extract_eos_from_tinker_result,
    extract_eos_from_topk,
    get_default_eos_token_ids,
)
from synthstats.integrations.tinker.losses import (
    compute_combined_tb_subtb_loss,
    compute_vanilla_tb_loss,
)

__all__ = [
    # adapter
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
    "compute_combined_tb_subtb_loss",
    "compute_vanilla_tb_loss",
    "is_tinker_available",
    "require_tinker",
    "trajectories_to_tinker_batch",
    # EOS extraction
    "extract_eos_from_tinker_result",
    "extract_eos_from_topk",
    "get_default_eos_token_ids",
]
