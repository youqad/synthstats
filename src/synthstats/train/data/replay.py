"""Replay buffers for off-policy GFlowNet training."""

from __future__ import annotations

import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from synthstats.train.data.collate import extract_reward

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@runtime_checkable
class ReplayCollector(Protocol):
    def replay_entry(self, entry: BufferEntry, temperature: float = 1.0) -> Any | None: ...


@dataclass
class BufferEntry:
    """Stores action sequence only (no tensors); re-scored on sample.

    log_reward is from collection time and won't be re-scaled under reward
    temperature annealing. Keep replay fraction modest (<=25%) if annealing.
    """

    actions: list[dict[str, Any]]
    log_reward: float
    observations: list[str]
    policy_version: int = 0
    temperature: float = 1.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": self.actions,
            "log_reward": self.log_reward,
            "observations": self.observations,
            "policy_version": self.policy_version,
            "temperature": self.temperature,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BufferEntry:
        return cls(
            actions=data["actions"],
            log_reward=data["log_reward"],
            observations=data["observations"],
            policy_version=data.get("policy_version", 0),
            temperature=data.get("temperature", 1.0),
            timestamp=data.get("timestamp", 0.0),
        )


class GFNReplayBuffer:
    """Replay buffer that re-scores entries with the current policy on sample,
    eliminating off-policy bias. See BufferEntry for reward annealing caveat.
    """

    def __init__(
        self,
        capacity: int,
        prioritized: bool = False,
        alpha: float = 1.0,
        max_retry_factor: float = 2.0,
    ) -> None:
        self._capacity = capacity
        self._prioritized = prioritized
        self._alpha = alpha
        self._max_retry_factor = max_retry_factor
        self._buffer: deque[BufferEntry] = deque(maxlen=capacity)
        self._policy_version = 0

    def add(self, entry: BufferEntry) -> None:
        self._buffer.append(entry)

    def pre_populate(
        self,
        entries: list[BufferEntry],
        *,
        dedupe: bool = True,
        source_label: str = "sft",
    ) -> int:
        """Seed buffer with entries (e.g., SFT data). Returns count added."""
        added = 0
        skipped_dupe = 0

        seen_signatures: set[str] = set()
        if dedupe:
            for existing in self._buffer:
                sig = self._action_signature(existing)
                seen_signatures.add(sig)

        for entry in entries:
            if dedupe:
                sig = self._action_signature(entry)
                if sig in seen_signatures:
                    skipped_dupe += 1
                    continue
                seen_signatures.add(sig)

            self._buffer.append(entry)
            added += 1

        if skipped_dupe > 0:
            logger.info(
                f"Pre-populated {added} {source_label} entries, skipped {skipped_dupe} duplicates"
            )
        else:
            logger.info(f"Pre-populated {added} {source_label} entries")

        return added

    @staticmethod
    def _action_signature(entry: BufferEntry) -> str:
        import json

        return json.dumps(entry.actions, sort_keys=True)

    def add_from_trajectory(self, traj: Any, log_reward: float) -> None:
        temperature = getattr(traj, "temperature", 1.0)
        entry = BufferEntry(
            actions=list(traj.actions),
            log_reward=log_reward,
            observations=list(traj.observations),
            policy_version=self._policy_version,
            temperature=temperature,
        )
        self.add(entry)

    def sample(
        self,
        batch_size: int,
        collector: ReplayCollector,
        temperature: float = 1.0,
    ) -> list[Any]:
        """Sample entries and re-score with current policy via the collector."""
        if batch_size == 0:
            return []
        if len(self._buffer) == 0:
            raise ValueError("Cannot sample from empty buffer")

        results: list[Any] = []
        max_attempts = int(batch_size * self._max_retry_factor)
        attempts = 0

        while len(results) < batch_size and attempts < max_attempts:
            entry = self._select_entries(1)[0]
            result = collector.replay_entry(entry, temperature=temperature)
            if result is not None:
                results.append(result)
            attempts += 1

        if len(results) < batch_size:
            logger.warning(f"Only got {len(results)}/{batch_size} valid samples")

        return results

    def _select_entries(self, n: int) -> list[BufferEntry]:
        import math

        if not self._prioritized or self._alpha == 0:
            return random.choices(list(self._buffer), k=n)

        entries = list(self._buffer)
        log_rewards = [e.log_reward for e in entries]
        max_lr = max(log_rewards)
        eps = 1e-8
        weights = [math.exp(self._alpha * (lr - max_lr)) + eps for lr in log_rewards]
        return random.choices(entries, weights=weights, k=n)

    def increment_policy_version(self) -> None:
        self._policy_version += 1

    def get_staleness_stats(self) -> dict[str, float]:
        if len(self._buffer) == 0:
            return {"mean_staleness": 0.0, "max_staleness": 0}
        staleness = [self._policy_version - e.policy_version for e in self._buffer]
        return {
            "mean_staleness": sum(staleness) / len(staleness),
            "max_staleness": max(staleness),
        }

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self):
        return iter(self._buffer)

    def state_dict(self) -> dict[str, Any]:
        return {
            "entries": [e.to_dict() for e in self._buffer],
            "policy_version": self._policy_version,
            "capacity": self._capacity,
            "prioritized": self._prioritized,
            "alpha": self._alpha,
            "max_retry_factor": self._max_retry_factor,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._policy_version = state["policy_version"]
        self._capacity = state["capacity"]
        self._prioritized = state["prioritized"]
        self._alpha = state["alpha"]
        self._max_retry_factor = state.get("max_retry_factor", 2.0)
        self._buffer = deque(maxlen=self._capacity)
        for entry_dict in state["entries"]:
            self._buffer.append(BufferEntry.from_dict(entry_dict))


class ReplayBuffer:
    """FIFO replay buffer with optional reward-proportional sampling."""

    def __init__(
        self,
        capacity: int,
        prioritized: bool = False,
        alpha: float = 1.0,
    ) -> None:
        self._capacity = capacity
        self._prioritized = prioritized
        self._alpha = alpha
        self._buffer: deque[Any] = deque(maxlen=capacity)

    def add(self, trajectory: Any) -> None:
        self._buffer.append(trajectory)

    def sample(self, batch_size: int) -> list[Any]:
        if batch_size == 0:
            return []
        if len(self._buffer) == 0:
            raise ValueError("Cannot sample from empty buffer")

        if self._prioritized:
            return self._prioritized_sample(batch_size)
        return random.choices(list(self._buffer), k=batch_size)

    def _prioritized_sample(self, batch_size: int) -> list[Any]:
        trajectories = list(self._buffer)
        rewards = [extract_reward(t) for t in trajectories]
        min_reward = min(rewards)
        shifted = [r - min_reward for r in rewards]
        eps = 1e-6
        weights = [(r + eps) ** self._alpha for r in shifted]
        return random.choices(trajectories, weights=weights, k=batch_size)

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self):
        return iter(self._buffer)

    def state_dict(self) -> dict[str, Any]:
        entries = []
        for t in self._buffer:
            if hasattr(t, "to_dict"):
                entries.append(t.to_dict())
            else:
                entries.append({"reward": extract_reward(t)})
        return {
            "trajectories": entries,
            "capacity": self._capacity,
            "prioritized": self._prioritized,
            "alpha": self._alpha,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        from synthstats.core.types import Trajectory

        self._capacity = state["capacity"]
        self._prioritized = state["prioritized"]
        self._alpha = state["alpha"]
        self._buffer = deque(maxlen=self._capacity)
        for traj_dict in state["trajectories"]:
            if "messages" in traj_dict:
                self._buffer.append(Trajectory.from_dict(traj_dict))
