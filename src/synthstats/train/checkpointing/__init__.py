"""Checkpointing - state persistence.

- CheckpointManager: Protocol for checkpoint management
- FullStateCheckpoint: Complete state (policy, optimizer, RNG, replay, step)
- MinimalCheckpoint: Minimal state for backends that own their state
"""

from synthstats.train.checkpointing.base import CheckpointManager
from synthstats.train.checkpointing.minimal import MinimalCheckpoint
from synthstats.train.checkpointing.torch_full import FullStateCheckpoint

__all__ = [
    "CheckpointManager",
    "FullStateCheckpoint",
    "MinimalCheckpoint",
]
