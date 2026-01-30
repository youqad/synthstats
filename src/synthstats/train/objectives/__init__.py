"""Objectives - loss computation.

Objectives compute loss and metrics from batches:
- SubTBObjective: Trajectory balance loss with learnable logZ
"""

from synthstats.train.objectives.subtb import SubTBObjective

__all__ = [
    "SubTBObjective",
]
