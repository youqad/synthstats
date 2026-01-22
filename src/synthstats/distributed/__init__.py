"""Distributed training infrastructure for SkyRL integration.

This package provides GFlowNet-specific distributed training components
that integrate with SkyRL's Ray-based infrastructure.

Key components:
- DriverGFNReplayBuffer: Replay buffer managed on driver with on-sample re-scoring
- GFlowNetTrainer: RayPPOTrainer subclass for SubTB loss
- GFlowNetExp: BasePPOExp subclass for GFlowNet experiments
- GFNConfig: Configuration dataclass for GFlowNet training
- GFNBatch: Batch structure for GFlowNet training (SubTB fields)
- Scoring utilities: Actor scoring with multi-EOS extraction

Architecture:
    GFlowNetExp (BasePPOExp)
    └── GFlowNetTrainer (RayPPOTrainer)
        ├── DriverGFNReplayBuffer (driver-side)
        ├── vLLM rollout (generation only)
        └── FSDP actor scoring (log_probs + eos_logprobs)

Dispatch modes:
- Distributed scoring via policy_actor_group when SkyRL is available
- Falls back to local model scoring for standalone development

Import-safe: Works without SkyRL installed for local development.

Usage:
    # Standalone mode (no SkyRL)
    from synthstats.distributed import GFlowNetTrainer, GFNConfig

    trainer = GFlowNetTrainer(cfg, gfn_config=GFNConfig())
    trainer.train()

    # SkyRL mode
    from synthstats.distributed import GFlowNetExp

    exp = GFlowNetExp(cfg)
    exp.trainer.train()
"""

from __future__ import annotations

from synthstats.distributed.driver_replay_buffer import (
    BufferEntry,
    DriverGFNReplayBuffer,
)
from synthstats.distributed.scoring import (
    build_response_mask,
    compute_log_probs_with_eos,
    get_stop_token_ids,
    STOP_TOKEN_IDS,
)

__all__ = [
    # replay buffer
    "BufferEntry",
    "DriverGFNReplayBuffer",
    # scoring utilities
    "build_response_mask",
    "compute_log_probs_with_eos",
    "get_stop_token_ids",
    "STOP_TOKEN_IDS",
]

# export trainer components (work without SkyRL)
try:
    from synthstats.distributed.gfn_trainer import (
        GFlowNetTrainer,
        GFNConfig,
        GFNBatch,
    )

    __all__.extend(["GFlowNetTrainer", "GFNConfig", "GFNBatch"])
except ImportError:
    # minimal dependencies not available
    pass

# export experiment class (requires SkyRL)
try:
    from synthstats.distributed.gfn_exp import GFlowNetExp

    __all__.append("GFlowNetExp")
except ImportError:
    # SkyRL not available
    pass
