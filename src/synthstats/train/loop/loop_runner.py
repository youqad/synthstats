"""Generic training loop: collect, learn, log, checkpoint."""

from __future__ import annotations

import dataclasses
import logging
import math
from collections import deque
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from synthstats.core.constants import REWARD_FLOOR_DEFAULT
from synthstats.train.data.collate import extract_reward
from synthstats.train.data.metrics import summarize_eval_metrics
from synthstats.train.learners.base import Learner

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
    ) -> list[Any]: ...


@dataclass
class LoopConfig:
    """LoopRunner configuration."""

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
    reward_floor: float = REWARD_FLOOR_DEFAULT
    log_interval: int = 10
    device: str = "cpu"


class LoopRunner:
    """Generic training loop: collect, learn, log, checkpoint.

    Backend-agnostic; works with PyTorch, Tinker, or Ray.
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
        self._all_metrics: deque[dict[str, float]] = deque(maxlen=10_000)
        self._last_eos_downgraded = False

        # replay buffers (lazy init)
        self._replay_buffer: Any | None = None
        self._gfn_replay_buffer: Any | None = None

        if self.config.replay_buffer_size > 0:
            self._init_replay_buffer()

    def _init_replay_buffer(self) -> None:
        if self.config.use_gfn_replay:
            from synthstats.train.data.replay import GFNReplayBuffer

            self._gfn_replay_buffer = GFNReplayBuffer(
                capacity=self.config.replay_buffer_size,
                prioritized=self.config.replay_prioritized,
                alpha=self.config.replay_alpha,
            )
        else:
            from synthstats.train.data.replay import ReplayBuffer

            self._replay_buffer = ReplayBuffer(
                capacity=self.config.replay_buffer_size,
                prioritized=self.config.replay_prioritized,
                alpha=self.config.replay_alpha,
            )

    @property
    def gfn_replay_buffer(self) -> Any:
        """GFN replay buffer, or None."""
        return self._gfn_replay_buffer

    def _replay_availability(self) -> tuple[bool, bool]:
        has_gfn_buffer = (
            self._gfn_replay_buffer is not None
            and len(self._gfn_replay_buffer) >= self.config.batch_size
        )
        has_simple_buffer = (
            self._replay_buffer is not None and len(self._replay_buffer) >= self.config.batch_size
        )
        return has_gfn_buffer, has_simple_buffer

    def _compute_batch_split(
        self,
        has_gfn_buffer: bool,
        has_simple_buffer: bool,
    ) -> tuple[int, int]:
        if has_gfn_buffer or has_simple_buffer:
            num_replay = int(self.config.batch_size * self.config.replay_ratio)
            num_fresh = self.config.batch_size - num_replay
        else:
            num_fresh = self.config.batch_size
            num_replay = 0
        return num_fresh, num_replay

    def _collect_fresh_trajectories(self, num_fresh: int) -> list[Any]:
        return self.collector.collect(
            episodes=num_fresh,
            temperature=self.config.temperature,
        )

    def _add_fresh_to_replay(self, fresh_trajectories: list[Any]) -> None:
        if self._gfn_replay_buffer is not None:
            for traj in fresh_trajectories:
                log_reward = math.log(max(extract_reward(traj), self.config.reward_floor))
                self._gfn_replay_buffer.add_from_trajectory(traj, log_reward=log_reward)
            return

        if self._replay_buffer is not None:
            for traj in fresh_trajectories:
                if hasattr(traj, "detach"):
                    self._replay_buffer.add(traj.detach())
                else:
                    self._replay_buffer.add(traj)

    def _sample_replay_trajectories(
        self,
        *,
        num_replay: int,
        has_gfn_buffer: bool,
        has_simple_buffer: bool,
    ) -> list[Any]:
        if num_replay <= 0:
            return []

        if has_gfn_buffer:
            assert self._gfn_replay_buffer is not None
            return self._gfn_replay_buffer.sample(
                batch_size=num_replay,
                collector=self.collector,
                temperature=self.config.temperature,
            )

        if has_simple_buffer:
            assert self._replay_buffer is not None
            return self._replay_buffer.sample(num_replay)

        return []

    def _combine_trajectories(
        self,
        fresh_trajectories: list[Any],
        replay_trajectories: list[Any],
    ) -> list[Any]:
        trajectories = fresh_trajectories + replay_trajectories
        self._last_eos_downgraded = False

        # guard: strip EOS when mixing fresh (has EOS) + replay (no EOS)
        if replay_trajectories and fresh_trajectories:
            has_eos = [
                hasattr(t, "eos_logprobs") and t.eos_logprobs is not None for t in trajectories
            ]
            if any(has_eos) and not all(has_eos):
                logger.warning(
                    "stripping eos_logprobs: replay lacks EOS, falling back to vanilla TB"
                )
                self._last_eos_downgraded = True
                return [
                    replace(t, eos_logprobs=None) if dataclasses.is_dataclass(t) else t
                    for t in trajectories
                ]

        return trajectories

    def _build_batch(self, trajectories: list[Any]) -> Any:
        if self.batch_builder is not None:
            return self.batch_builder(
                trajectories,
                reward_floor=self.config.reward_floor,
                device=self.config.device,
            )

        from synthstats.train.data.collate import build_subtb_batch

        return build_subtb_batch(
            trajectories,
            reward_floor=self.config.reward_floor,
            device=self.config.device,
        )

    def _update_buffer_metrics(self, metrics: dict[str, float]) -> None:
        if self._gfn_replay_buffer is None:
            return

        self._gfn_replay_buffer.increment_policy_version()
        staleness = self._gfn_replay_buffer.get_staleness_stats()
        metrics["buffer_mean_staleness"] = staleness["mean_staleness"]
        metrics["buffer_max_staleness"] = staleness["max_staleness"]
        metrics["buffer_size"] = len(self._gfn_replay_buffer)

    def _finalize_step_metrics(
        self,
        metrics: dict[str, float],
        trajectories: list[Any],
        *,
        num_replay: int,
    ) -> dict[str, float]:
        avg_reward = sum(extract_reward(t) for t in trajectories) / len(trajectories)
        metrics["avg_reward"] = avg_reward
        metrics["num_episodes"] = len(trajectories)
        metrics["replay_ratio"] = num_replay / len(trajectories) if trajectories else 0.0
        metrics["eos_downgraded"] = 1.0 if self._last_eos_downgraded else 0.0
        self._update_buffer_metrics(metrics)
        return metrics

    def _record_step(self, metrics: dict[str, float]) -> None:
        self.step_count += 1
        self._all_metrics.append(metrics)

        if self.logger_sink is not None and self.step_count % self.config.log_interval == 0:
            self.logger_sink.log(self.step_count, metrics)

    def train_step(self) -> dict[str, float]:
        """Collect, build batch, learn, log. Returns metrics dict."""
        has_gfn_buffer, has_simple_buffer = self._replay_availability()
        num_fresh, num_replay = self._compute_batch_split(has_gfn_buffer, has_simple_buffer)

        fresh_trajectories = self._collect_fresh_trajectories(num_fresh)
        self._add_fresh_to_replay(fresh_trajectories)

        replay_trajectories = self._sample_replay_trajectories(
            num_replay=num_replay,
            has_gfn_buffer=has_gfn_buffer,
            has_simple_buffer=has_simple_buffer,
        )
        trajectories = self._combine_trajectories(fresh_trajectories, replay_trajectories)
        batch = self._build_batch(trajectories)
        metrics = self.learner.update(batch)
        metrics = self._finalize_step_metrics(metrics, trajectories, num_replay=num_replay)
        self._record_step(metrics)
        return metrics

    def run(self, steps: int | None = None) -> list[dict[str, float]]:
        """Run training for n steps (defaults to config.num_steps)."""
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
        """Evaluate current policy without training."""
        import torch

        n = episodes or self.config.eval_episodes
        temp = max(self.config.temperature, 1e-3)  # avoid zero

        with torch.no_grad():
            trajectories = self.collector.collect(episodes=n, temperature=temp)

        rewards = [extract_reward(t) for t in trajectories]
        return summarize_eval_metrics(
            rewards,
            episodes=n,
            logZ=self.learner.logZ,
        )

    @property
    def metrics_history(self) -> list[dict[str, float]]:
        """All metrics from training."""
        return list(self._all_metrics)
