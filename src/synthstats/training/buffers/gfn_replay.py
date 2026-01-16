"""GFlowNet-specific replay buffer with on-sample re-scoring.

Unlike standard replay buffers that store full trajectories with log_probs,
this buffer stores minimal action sequences and re-scores them with the
current policy when sampled. This eliminates off-policy bias from stale
log_probs.

Key design:
- BufferEntry stores actions + observations (no tensors)
- GFNReplayBuffer.sample() takes a collector with replay_entry() method
- Policy version tracking for staleness diagnostics

This buffer is compatible with any collector that implements:
- replay_entry(entry: BufferEntry, temperature: float) -> trajectory or None

Note: Replay requires the collector to be able to reconstruct trajectories
from action sequences. The returned trajectories should have fresh log_probs
from the current policy.
"""

from __future__ import annotations

import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    pass  # no external type dependencies


@runtime_checkable
class ReplayCollector(Protocol):
    """Protocol for collectors that support replay."""

    def replay_entry(
        self, entry: BufferEntry, temperature: float = 1.0
    ) -> Any | None:
        """Replay a buffer entry and return fresh trajectory."""
        ...

logger = logging.getLogger(__name__)


@dataclass
class BufferEntry:
    """Minimal replay entry - stores action sequence only, no tensors.

    This is what gets stored in the GFNReplayBuffer. When sampled, these
    entries are replayed through the environment and re-scored with the
    current policy.
    """

    actions: list[dict[str, Any]]
    log_reward: float
    observations: list[str]
    policy_version: int = 0
    temperature: float = 1.0
    timestamp: float = field(default_factory=time.time)


class GFNReplayBuffer:
    """GFlowNet replay buffer with on-sample re-scoring.

    Stores action sequences (no tensors). When sampled, entries are replayed
    through the environment and re-scored with the current policy, ensuring
    log_probs are always from the current policy.

    Args:
        capacity: Maximum number of entries to store (FIFO eviction)
        prioritized: If True, sample proportional to exp(log_reward)
        alpha: Prioritization exponent (0=uniform, 1=fully prioritized)
        max_retry_factor: Max entries to try when sampling fails

    Example:
        >>> buffer = GFNReplayBuffer(capacity=100, prioritized=True)
        >>> buffer.add_from_trajectory(traj, log_reward=-0.5)
        >>> samples = buffer.sample(batch_size=4, collector=collector)
        >>> # samples have FRESH log_probs from current policy
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
        """Add a BufferEntry to the buffer.

        If buffer is at capacity, the oldest entry is evicted (FIFO).
        """
        self._buffer.append(entry)

    def add_from_trajectory(
        self,
        traj: Any,
        log_reward: float,
    ) -> None:
        """Convert a trajectory object to BufferEntry and add.

        Extracts only the action sequence and observations - no tensors
        are stored. The log_probs and entropy from the trajectory are
        discarded since they'll be re-computed on sample.

        The trajectory object must have:
        - actions: list of action dicts
        - observations: list of observation strings
        - temperature: float (optional, defaults to 1.0)

        Args:
            traj: Trajectory object with actions and observations
            log_reward: Log-transformed reward (pre-computed)
        """
        temperature = getattr(traj, "temperature", 1.0)
        entry = BufferEntry(
            actions=list(traj.actions),  # copy to avoid mutation
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
        """Sample entries and re-score with current policy.

        Each sampled entry is replayed through the environment using
        collector.replay_entry(), which re-computes log_probs with the
        current policy's score_action() method.

        Args:
            batch_size: Number of trajectories to sample
            collector: Object implementing ReplayCollector protocol
            temperature: Sampling temperature for re-scoring

        Returns:
            List of trajectory objects with fresh log_probs

        Raises:
            ValueError: If buffer is empty
        """
        if batch_size == 0:
            return []

        if len(self._buffer) == 0:
            raise ValueError("Cannot sample from empty buffer")

        # select entries (uniform or prioritized)
        entries = self._select_entries(batch_size)

        # replay each entry with current policy
        results: list[Any] = []
        max_attempts = int(batch_size * self._max_retry_factor)
        attempts = 0

        for entry in entries:
            if len(results) >= batch_size:
                break

            result = collector.replay_entry(entry, temperature=temperature)
            if result is not None:
                results.append(result)

            attempts += 1
            if attempts >= max_attempts and len(results) < batch_size:
                logger.warning(
                    f"GFNReplayBuffer.sample: only got {len(results)}/{batch_size} "
                    f"valid samples after {attempts} attempts"
                )
                break

        return results

    def _select_entries(self, n: int) -> list[BufferEntry]:
        """Select n entries from buffer (with replacement)."""
        if not self._prioritized or self._alpha == 0:
            return random.choices(list(self._buffer), k=n)

        # prioritized sampling by exp(log_reward)
        entries = list(self._buffer)
        log_rewards = [e.log_reward for e in entries]

        # shift to positive range and apply alpha
        min_lr = min(log_rewards)
        shifted = [lr - min_lr for lr in log_rewards]

        # add epsilon to avoid zero weights
        eps = 1e-6
        weights = [(s + eps) ** self._alpha for s in shifted]

        return random.choices(entries, weights=weights, k=n)

    def increment_policy_version(self) -> None:
        """Increment policy version counter.

        Call this after each policy update to track staleness of
        buffer entries.
        """
        self._policy_version += 1

    def get_staleness_stats(self) -> dict[str, float]:
        """Get statistics about entry staleness.

        Returns:
            Dict with mean_staleness and max_staleness (version difference)
        """
        if len(self._buffer) == 0:
            return {"mean_staleness": 0.0, "max_staleness": 0}

        staleness = [
            self._policy_version - e.policy_version for e in self._buffer
        ]
        return {
            "mean_staleness": sum(staleness) / len(staleness),
            "max_staleness": max(staleness),
        }

    def __len__(self) -> int:
        """Return number of entries in buffer."""
        return len(self._buffer)

    def __iter__(self):
        """Iterate over all entries in buffer."""
        return iter(self._buffer)
