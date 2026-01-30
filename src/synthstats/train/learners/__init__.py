"""Learners - parameter update implementations.

Learners take trajectories and update model parameters:
- SubTBTorchLearner: PyTorch-based SubTB training
- SubTBTinkerLearner: Tinker API-based SubTB training
"""

from synthstats.train.learners.base import Learner
from synthstats.train.learners.subtb_torch import SubTBTorchLearner

__all__ = [
    "Learner",
    "SubTBTorchLearner",
]
