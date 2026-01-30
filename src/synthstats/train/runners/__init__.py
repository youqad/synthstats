"""Training runners - execution backends.

Runners own the training execution:
- LocalRunner: Pure PyTorch single-node training
- SkyRLRayRunner: Distributed training via Ray + SkyRL
- TinkerRunner: API-based training via Tinker
"""

from synthstats.train.runners.base import Runner, RunResult

__all__ = [
    "Runner",
    "RunResult",
]
