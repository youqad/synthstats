"""Driver-side GFlowNet replay buffer for Ray distributed training.

Lives on the driver process and stores token IDs (not action dicts) for
efficient serialization when dispatching to workers for re-scoring.

Differs from training/buffers/gfn_replay.py which uses a collector for
re-scoring. This one dispatches to Ray workers instead.

FlowRL/TBA Enhancements (arXiv:2509.15207, arXiv:2503.18929):
- Deduplication: Hash token sequences to prevent overfitting
- Hybrid sampling: Mix recency-prioritized (on-policy) with reward-prioritized
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BufferEntry:
    """Replay entry with token IDs for Ray serialization."""

    action_token_ids: list[int]
    prompt_token_ids: list[int]
    log_reward: float
    policy_version: int = 0
    temperature: float = 1.0
    timestamp: float = field(default_factory=time.time)
    trajectory_id: str | None = None
    terminated: bool = True

    def total_tokens(self) -> int:
        """Total token count (prompt + response)."""
        return len(self.prompt_token_ids) + len(self.action_token_ids)

    def response_length(self) -> int:
        """Response (action) token count."""
        return len(self.action_token_ids)

    def content_hash(self) -> str:
        """Hash of prompt+action tokens for deduplication."""
        content = tuple(self.prompt_token_ids) + tuple(self.action_token_ids)
        return hashlib.md5(str(content).encode()).hexdigest()


class DriverGFNReplayBuffer:
    """GFlowNet replay buffer for Ray driver.

    Supports prioritized sampling, deduplication, and hybrid recency+reward
    sampling (FlowRL arXiv:2509.15207, TBA arXiv:2503.18929).
    """

    def __init__(
        self,
        capacity: int,
        prioritized: bool = True,
        alpha: float = 1.0,
        min_entries_before_sample: int = 1,
        deduplicate: bool = True,
        recency_ratio: float = 0.5,
    ) -> None:
        self._capacity = capacity
        self._prioritized = prioritized
        self._alpha = alpha
        self._min_entries = min_entries_before_sample
        self._deduplicate = deduplicate
        self._recency_ratio = recency_ratio
        self._buffer: deque[BufferEntry] = deque(maxlen=capacity)
        self._seen_hashes: set[str] = set()  # for deduplication
        self._policy_version = 0
        self._version_lock = threading.Lock()  # thread-safe version access
        self._duplicates_rejected = 0  # stats tracking

    def add(self, entry: BufferEntry) -> bool:
        """Add entry, evicting oldest if full. Returns False if duplicate."""
        if self._deduplicate:
            entry_hash = entry.content_hash()
            if entry_hash in self._seen_hashes:
                self._duplicates_rejected += 1
                return False

            # check if we're evicting an old entry
            if len(self._buffer) == self._capacity:
                old_entry = self._buffer[0]
                old_hash = old_entry.content_hash()
                self._seen_hashes.discard(old_hash)

            self._seen_hashes.add(entry_hash)

        self._buffer.append(entry)
        return True

    def add_from_batch(
        self,
        input_ids: list[list[int]],
        prompt_lengths: list[int],
        log_rewards: list[float],
        terminated: list[bool] | None = None,
        temperatures: list[float] | None = None,
        trajectory_ids: list[str] | None = None,
    ) -> int:
        """Add entries from training batch. Returns count added."""
        batch_size = len(input_ids)

        if terminated is None:
            terminated = [True] * batch_size
        if temperatures is None:
            temperatures = [1.0] * batch_size
        if trajectory_ids is None:
            trajectory_ids = [None] * batch_size  # type: ignore[list-item]

        # capture version atomically for all entries in batch
        current_version = self.policy_version

        added = 0
        for i in range(batch_size):
            prompt_len = prompt_lengths[i]
            entry = BufferEntry(
                action_token_ids=input_ids[i][prompt_len:],
                prompt_token_ids=input_ids[i][:prompt_len],
                log_reward=log_rewards[i],
                policy_version=current_version,
                temperature=temperatures[i],
                trajectory_id=trajectory_ids[i],
                terminated=terminated[i],
            )
            if self.add(entry):
                added += 1

        return added

    def sample(self, batch_size: int) -> list[BufferEntry]:
        """Sample with hybrid recency+reward strategy (TBA arXiv:2503.18929).

        Raises ValueError if buffer below min threshold.
        """
        if len(self._buffer) < self._min_entries:
            raise ValueError(
                f"Buffer has {len(self._buffer)} entries, "
                f"need at least {self._min_entries} before sampling"
            )

        if batch_size == 0:
            return []

        batch_size = min(batch_size, len(self._buffer))
        entries = list(self._buffer)
        n = len(entries)

        if not self._prioritized or self._alpha == 0:
            return random.choices(entries, k=batch_size)

        recency_count = int(batch_size * self._recency_ratio)
        reward_count = batch_size - recency_count

        samples: list[BufferEntry] = []

        # exponential recency weights (newest ~7x oldest)
        if recency_count > 0:
            recency_weights = [math.exp(2.0 * i / n) for i in range(n)]
            samples.extend(random.choices(entries, weights=recency_weights, k=recency_count))

        if reward_count > 0:
            log_rewards = [e.log_reward for e in entries]
            max_lr = max(log_rewards)
            reward_weights = [math.exp(self._alpha * (lr - max_lr)) for lr in log_rewards]
            samples.extend(random.choices(entries, weights=reward_weights, k=reward_count))

        random.shuffle(samples)
        return samples

    def increment_policy_version(self) -> None:
        """Bump version after policy update (tracks staleness). Thread-safe."""
        with self._version_lock:
            self._policy_version += 1

    @property
    def policy_version(self) -> int:
        """Current policy version. Thread-safe."""
        with self._version_lock:
            return self._policy_version

    def get_staleness_stats(self) -> dict[str, float | int]:
        """Returns mean_staleness and max_staleness (version gap)."""
        if len(self._buffer) == 0:
            return {"mean_staleness": 0.0, "max_staleness": 0}

        current_version = self.policy_version  # thread-safe access
        staleness = [current_version - e.policy_version for e in self._buffer]
        return {
            "mean_staleness": sum(staleness) / len(staleness),
            "max_staleness": max(staleness),
        }

    def get_stats(self) -> dict[str, Any]:
        """Buffer statistics including FlowRL/TBA metrics."""
        if len(self._buffer) == 0:
            return {
                "size": 0,
                "capacity": self._capacity,
                "fill_ratio": 0.0,
                "mean_staleness": 0.0,
                "max_staleness": 0,
                "mean_log_reward": 0.0,
                "mean_tokens": 0.0,
                "mean_response_length": 0.0,
                "duplicates_rejected": self._duplicates_rejected,
                "unique_hashes": len(self._seen_hashes),
            }

        staleness_stats = self.get_staleness_stats()
        log_rewards = [e.log_reward for e in self._buffer]
        token_counts = [e.total_tokens() for e in self._buffer]
        response_lengths = [e.response_length() for e in self._buffer]

        return {
            "size": len(self._buffer),
            "capacity": self._capacity,
            "fill_ratio": len(self._buffer) / self._capacity,
            "mean_staleness": staleness_stats["mean_staleness"],
            "max_staleness": staleness_stats["max_staleness"],
            "mean_log_reward": sum(log_rewards) / len(log_rewards),
            "mean_tokens": sum(token_counts) / len(token_counts),
            "mean_response_length": sum(response_lengths) / len(response_lengths),
            "duplicates_rejected": self._duplicates_rejected,
            "unique_hashes": len(self._seen_hashes),
        }

    def clear(self) -> None:
        self._buffer.clear()
        self._seen_hashes.clear()
        self._duplicates_rejected = 0

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self):
        return iter(self._buffer)
