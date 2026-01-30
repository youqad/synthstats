"""Generic training loop: collect → learn → log → checkpoint.

LoopRunner is the core training loop used by LocalRunner and TinkerRunner.
It orchestrates trajectory collection, batch building, learning, logging,
and checkpointing.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from synthstats.train.checkpointing.base import CheckpointManager
    from synthstats.train.logging.base import LoggerSink

logger = logging.getLogger(__name__)


@runtime_checkable
class Collector(Protocol):
    """Protocol for trajectory collectors."""

    def collect(
        self,
        episodes: int,
        temperature: float,
        compute_ref_log_probs: bool = False,
    ) -> list[Any]:
        """Collect trajectories."""
        ...


@runtime_checkable
class Learner(Protocol):
    """Protocol for learners."""

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        """Update from batch, return metrics."""
        ...

    @property
    def logZ(self) -> float:
        """Current logZ value."""
        ...


@dataclass
class LoopConfig:
    """Configuration for LoopRunner."""

    # training
    num_steps: int = 1000
    batch_size: int = 4
    temperature: float = 0.7

    # evaluation
    eval_interval: int = 100
    eval_episodes: int = 8

    # replay buffer
    replay_buffer_size: int = 0
    replay_ratio: float = 0.5
    replay_prioritized: bool = False
    replay_alpha: float = 1.0
    use_gfn_replay: bool = True

    # misc
    reward_floor: float = 1e-4
    log_interval: int = 10
    device: str = "cpu"


class LoopRunner:
    """Generic training loop: collect → learn → log → checkpoint.

    Orchestrates the training process without knowing about specific
    backends (PyTorch vs Tinker vs Ray).

    Args:
        collector: Trajectory collector
        learner: Parameter updater
        logger_sink: Metric logger
        checkpoint_manager: Checkpoint handler
        config: Loop configuration

    Example:
        >>> loop = LoopRunner(collector, learner, logger, checkpoint_mgr, config)
        >>> loop.run(steps=100)
    """

    def __init__(
        self,
        collector: Collector,
        learner: Learner,
        logger_sink: LoggerSink | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        config: LoopConfig | None = None,
        batch_builder: Any | None = None,
    ) -> None:
        self.collector = collector
        self.learner = learner
        self.logger_sink = logger_sink
        self.checkpoint_manager = checkpoint_manager
        self.config = config or LoopConfig()
        self.batch_builder = batch_builder

        self.step_count = 0
        self._all_metrics: list[dict[str, float]] = []

        # replay buffers (lazy init)
        self._replay_buffer: Any | None = None
        self._gfn_replay_buffer: Any | None = None

        if self.config.replay_buffer_size > 0:
            self._init_replay_buffer()

    def _init_replay_buffer(self) -> None:
        """Initialize appropriate replay buffer."""
        if self.config.use_gfn_replay:
            from synthstats.train.loop.replay import GFNReplayBuffer

            self._gfn_replay_buffer = GFNReplayBuffer(
                capacity=self.config.replay_buffer_size,
                prioritized=self.config.replay_prioritized,
                alpha=self.config.replay_alpha,
            )
        else:
            from synthstats.train.loop.replay import ReplayBuffer

            self._replay_buffer = ReplayBuffer(
                capacity=self.config.replay_buffer_size,
                prioritized=self.config.replay_prioritized,
                alpha=self.config.replay_alpha,
            )

    def train_step(self) -> dict[str, float]:
        """Run a single training step.

        1. Collect trajectories (fresh + replay if enabled)
        2. Build batch
        3. Update parameters
        4. Log metrics

        Returns:
            Metrics dict
        """
        # determine fresh vs replay split
        has_gfn_buffer = (
            self._gfn_replay_buffer is not None
            and len(self._gfn_replay_buffer) >= self.config.batch_size
        )
        has_simple_buffer = (
            self._replay_buffer is not None and len(self._replay_buffer) >= self.config.batch_size
        )

        if has_gfn_buffer or has_simple_buffer:
            num_replay = int(self.config.batch_size * self.config.replay_ratio)
            num_fresh = self.config.batch_size - num_replay
        else:
            num_fresh = self.config.batch_size
            num_replay = 0

        # collect fresh trajectories
        fresh_trajectories = self.collector.collect(
            episodes=num_fresh,
            temperature=self.config.temperature,
        )

        # add to buffer
        if self._gfn_replay_buffer is not None:
            for traj in fresh_trajectories:
                reward = getattr(traj, "reward", 0.0)
                if hasattr(reward, "total"):
                    reward = reward.total
                log_reward = math.log(max(float(reward), self.config.reward_floor))
                self._gfn_replay_buffer.add_from_trajectory(traj, log_reward=log_reward)
        elif self._replay_buffer is not None:
            for traj in fresh_trajectories:
                if hasattr(traj, "detach"):
                    self._replay_buffer.add(traj.detach())
                else:
                    self._replay_buffer.add(traj)

        # sample from replay
        replay_trajectories = []
        if num_replay > 0:
            if has_gfn_buffer:
                replay_trajectories = self._gfn_replay_buffer.sample(
                    batch_size=num_replay,
                    collector=self.collector,
                    temperature=self.config.temperature,
                )
            elif has_simple_buffer:
                replay_trajectories = self._replay_buffer.sample(num_replay)

        trajectories = fresh_trajectories + replay_trajectories

        # build batch
        if self.batch_builder is not None:
            batch = self.batch_builder(
                trajectories,
                reward_floor=self.config.reward_floor,
                device=self.config.device,
            )
        else:
            # default: use build_subtb_batch
            from synthstats.train.loop.batching import build_subtb_batch

            batch = build_subtb_batch(
                trajectories,
                reward_floor=self.config.reward_floor,
                device=self.config.device,
            )

        # update parameters
        metrics = self.learner.update(batch)

        # add trajectory metrics
        def _get_reward(r: Any) -> float:
            if hasattr(r, "total"):
                return float(r.total)
            return float(r)

        avg_reward = sum(_get_reward(t.reward) for t in trajectories) / len(trajectories)
        metrics["avg_reward"] = avg_reward
        metrics["num_episodes"] = len(trajectories)
        metrics["replay_ratio"] = num_replay / len(trajectories) if trajectories else 0.0

        # buffer stats
        if self._gfn_replay_buffer is not None:
            self._gfn_replay_buffer.increment_policy_version()
            staleness = self._gfn_replay_buffer.get_staleness_stats()
            metrics["buffer_mean_staleness"] = staleness["mean_staleness"]
            metrics["buffer_max_staleness"] = staleness["max_staleness"]
            metrics["buffer_size"] = len(self._gfn_replay_buffer)

        self.step_count += 1
        self._all_metrics.append(metrics)

        # log
        if self.logger_sink is not None and self.step_count % self.config.log_interval == 0:
            self.logger_sink.log(self.step_count, metrics)

        return metrics

    def run(self, steps: int | None = None) -> list[dict[str, float]]:
        """Run training for n steps.

        Args:
            steps: Number of steps (defaults to config.num_steps)

        Returns:
            List of metrics from each step
        """
        n = steps if steps is not None else self.config.num_steps
        metrics_list = []

        for _ in range(n):
            metrics = self.train_step()
            metrics_list.append(metrics)

            # checkpoint
            if self.checkpoint_manager is not None:
                self.checkpoint_manager.maybe_save(
                    step=self.step_count,
                    learner=self.learner,
                )

        return metrics_list

    def evaluate(self, episodes: int | None = None) -> dict[str, float]:
        """Evaluate current policy without training.

        Args:
            episodes: Number of episodes (defaults to config)

        Returns:
            Evaluation metrics
        """
        n = episodes or self.config.eval_episodes
        temp = max(self.config.temperature, 1e-3)  # avoid zero

        trajectories = self.collector.collect(episodes=n, temperature=temp)

        def _get_reward(t: Any) -> float:
            r = getattr(t, "reward", 0.0)
            return float(r.total) if hasattr(r, "total") else float(r)

        rewards = [_get_reward(t) for t in trajectories]
        avg = sum(rewards) / len(rewards)
        mx = max(rewards)
        mn = min(rewards)
        success = sum(1 for r in rewards if r > 0) / len(rewards)

        return {
            "eval_avg_reward": avg,
            "eval_max_reward": mx,
            "eval_min_reward": mn,
            "eval_success_rate": success,
            "eval_episodes": n,
            "logZ": self.learner.logZ,
        }

    @property
    def metrics_history(self) -> list[dict[str, float]]:
        """All metrics collected during training."""
        return self._all_metrics
