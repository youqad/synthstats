"""Training loop components.

- LoopRunner: Generic collect → learn → log → checkpoint loop
- collectors: Trajectory collection implementations
- batching: Batch building utilities
- replay: Replay buffer implementations
"""

from synthstats.train.loop.batching import build_subtb_batch, build_tinker_batch
from synthstats.train.loop.collectors import CollectedTrajectory, TrajectoryCollector
from synthstats.train.loop.loop_runner import LoopRunner
from synthstats.train.loop.replay import GFNReplayBuffer, ReplayBuffer

__all__ = [
    "LoopRunner",
    "TrajectoryCollector",
    "CollectedTrajectory",
    "build_subtb_batch",
    "build_tinker_batch",
    "ReplayBuffer",
    "GFNReplayBuffer",
]
