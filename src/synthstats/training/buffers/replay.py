"""Trajectory replay buffer for experience storage and sampling."""

import random
from collections import deque
from typing import Any

from synthstats.core.types import Trajectory


class ReplayBuffer:
    """FIFO replay buffer for trajectories with optional prioritized sampling.

    Stores trajectories and samples them for training. When capacity is reached,
    oldest trajectories are evicted (FIFO).

    Args:
        capacity: Maximum number of trajectories to store.
        prioritized: If True, sample proportional to reward. Default False.
        alpha: Prioritization exponent. 0 = uniform, 1 = fully prioritized.
    """

    def __init__(
        self, capacity: int, prioritized: bool = False, alpha: float = 1.0
    ) -> None:
        self._capacity = capacity
        self._prioritized = prioritized
        self._alpha = alpha
        self._buffer: deque[Trajectory] = deque(maxlen=capacity)

    def add(self, trajectory: Trajectory) -> None:
        """Add a trajectory to the buffer.

        If buffer is at capacity, the oldest trajectory is evicted.
        """
        self._buffer.append(trajectory)

    def sample(self, batch_size: int) -> list[Trajectory]:
        """Sample a batch of trajectories.

        Args:
            batch_size: Number of trajectories to sample.

        Returns:
            List of sampled trajectories.

        Raises:
            ValueError: If buffer is empty.
        """
        if batch_size == 0:
            return []

        if len(self._buffer) == 0:
            raise ValueError("Cannot sample from empty buffer")

        if self._prioritized:
            return self._prioritized_sample(batch_size)

        return random.choices(list(self._buffer), k=batch_size)

    def _prioritized_sample(self, batch_size: int) -> list[Trajectory]:
        """Sample with probability proportional to reward."""
        trajectories = list(self._buffer)
        rewards = [self._get_reward(t) for t in trajectories]

        # shift rewards to be non-negative (handle negative rewards)
        min_reward = min(rewards)
        shifted = [r - min_reward for r in rewards]

        # add small epsilon to avoid zero weights
        eps = 1e-6
        weights = [(r + eps) ** self._alpha for r in shifted]

        return random.choices(trajectories, weights=weights, k=batch_size)

    @staticmethod
    def _get_reward(trajectory: Any) -> float:
        """Extract reward scalar from trajectory-like objects."""
        reward = getattr(trajectory, "reward", 0.0)
        if isinstance(reward, (float, int)):
            return float(reward)
        if hasattr(reward, "total"):
            return float(reward.total)
        return 0.0

    def __len__(self) -> int:
        """Return number of trajectories in buffer."""
        return len(self._buffer)

    def __iter__(self):
        """Iterate over all trajectories in buffer."""
        return iter(self._buffer)
