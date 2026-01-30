"""Training system for SynthStats.

The training system is organized as:
- runners/: Execution backends (local PyTorch, Ray+SkyRL, Tinker API)
- loop/: Generic training loop components (collectors, batching, replay)
- learners/: Parameter update implementations
- objectives/: Loss computation (SubTB, etc.)
- checkpointing/: State persistence
- logging/: Metric logging sinks
- utils/: Seeding, device resolution, etc.
"""

from synthstats.train.learners.subtb_torch import SubTBTorchLearner
from synthstats.train.loop.loop_runner import LoopRunner
from synthstats.train.objectives.subtb import SubTBObjective

__all__ = [
    "SubTBObjective",
    "SubTBTorchLearner",
    "LoopRunner",
]
