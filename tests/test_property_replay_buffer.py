"""Property-based tests for Replay Buffers using Hypothesis.

Tests invariants for both ReplayBuffer and GFNReplayBuffer.
"""

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from synthstats.core.types import Message, Reward, Trajectory
from synthstats.training.buffers.gfn_replay import BufferEntry, GFNReplayBuffer
from synthstats.training.buffers.replay import ReplayBuffer


def make_trajectory(reward_total: float = 0.5) -> Trajectory:
    """Helper to create a minimal trajectory for testing."""
    return Trajectory(
        messages=[Message(role="user", content="test")],
        token_ids=[[1, 2, 3]],
        token_logprobs=[[-0.5, -0.3, -0.2]],
        loss_mask=[[True, True, True]],
        reward=Reward(total=reward_total, components={}, info={}),
    )


def make_buffer_entry(log_reward: float = 0.0, policy_version: int = 0) -> BufferEntry:
    """Helper to create a minimal BufferEntry for testing."""
    return BufferEntry(
        actions=[{"type": "answer", "payload": "42"}],
        log_reward=log_reward,
        observations=["test observation"],
        policy_version=policy_version,
        temperature=1.0,
    )


class TestReplayBufferProperties:
    """Property-based tests for ReplayBuffer."""

    @given(
        capacity=st.integers(min_value=1, max_value=100),
        n_adds=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=50)
    def test_capacity_never_exceeded(self, capacity: int, n_adds: int):
        """Buffer size never exceeds capacity."""
        buffer = ReplayBuffer(capacity=capacity)

        for i in range(n_adds):
            buffer.add(make_trajectory(reward_total=float(i)))

        assert len(buffer) <= capacity, (
            f"Buffer size {len(buffer)} exceeds capacity {capacity}"
        )

    @given(
        capacity=st.integers(min_value=2, max_value=20),
        n_adds=st.integers(min_value=3, max_value=50),
    )
    @settings(max_examples=30)
    def test_fifo_eviction_order(self, capacity: int, n_adds: int):
        """Oldest entries should be evicted first (FIFO)."""
        assume(n_adds > capacity)  # only test when eviction happens

        buffer = ReplayBuffer(capacity=capacity)

        for i in range(n_adds):
            buffer.add(make_trajectory(reward_total=float(i)))

        # check that oldest entries are gone
        all_rewards = {t.reward.total for t in buffer}
        for i in range(n_adds - capacity):
            assert float(i) not in all_rewards, f"Entry {i} should have been evicted"

        # check that newest entries remain
        for i in range(n_adds - capacity, n_adds):
            assert float(i) in all_rewards, f"Entry {i} should be present"

    @given(
        capacity=st.integers(min_value=5, max_value=20),
        n_in_buffer=st.integers(min_value=1, max_value=20),
        sample_size=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=30)
    def test_sample_returns_correct_size(self, capacity: int, n_in_buffer: int, sample_size: int):
        """sample() returns exactly the requested number of items."""
        buffer = ReplayBuffer(capacity=capacity)

        for i in range(min(n_in_buffer, capacity)):
            buffer.add(make_trajectory(reward_total=float(i)))

        if len(buffer) > 0:
            samples = buffer.sample(sample_size)
            assert len(samples) == sample_size, (
                f"Expected {sample_size} samples, got {len(samples)}"
            )

    @given(
        capacity=st.integers(min_value=5, max_value=20),
        n_in_buffer=st.integers(min_value=2, max_value=20),
    )
    @settings(max_examples=30)
    def test_sample_from_buffer_contents(self, capacity: int, n_in_buffer: int):
        """All sampled items must exist in the buffer."""
        buffer = ReplayBuffer(capacity=capacity)

        for i in range(min(n_in_buffer, capacity)):
            buffer.add(make_trajectory(reward_total=float(i)))

        buffer_rewards = {t.reward.total for t in buffer}
        samples = buffer.sample(5)

        for sample in samples:
            assert sample.reward.total in buffer_rewards, (
                f"Sampled reward {sample.reward.total} not in buffer"
            )


class TestGFNReplayBufferProperties:
    """Property-based tests for GFNReplayBuffer."""

    @given(
        capacity=st.integers(min_value=1, max_value=100),
        n_adds=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=50)
    def test_capacity_never_exceeded(self, capacity: int, n_adds: int):
        """Buffer size never exceeds capacity."""
        buffer = GFNReplayBuffer(capacity=capacity)

        for i in range(n_adds):
            buffer.add(make_buffer_entry(log_reward=float(-i)))

        assert len(buffer) <= capacity, f"Buffer size {len(buffer)} exceeds capacity {capacity}"

    @given(
        n_increments=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=30)
    def test_policy_version_monotonic(self, n_increments: int):
        """Policy version only increases, never decreases."""
        buffer = GFNReplayBuffer(capacity=10)

        versions = [buffer._policy_version]
        for _ in range(n_increments):
            buffer.increment_policy_version()
            versions.append(buffer._policy_version)

        for i in range(1, len(versions)):
            assert versions[i] >= versions[i - 1], (
                f"Policy version decreased: {versions[i - 1]} -> {versions[i]}"
            )

    @given(
        capacity=st.integers(min_value=5, max_value=20),
        n_entries=st.integers(min_value=1, max_value=20),
        n_version_bumps=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=30)
    def test_staleness_stats_valid(self, capacity: int, n_entries: int, n_version_bumps: int):
        """Staleness stats: mean <= max, both >= 0."""
        buffer = GFNReplayBuffer(capacity=capacity)

        # add entries with some version bumps interspersed
        for i in range(min(n_entries, capacity)):
            if i > 0 and i % 3 == 0:
                buffer.increment_policy_version()
            buffer.add(make_buffer_entry(log_reward=float(-i)))

        # bump version more
        for _ in range(n_version_bumps):
            buffer.increment_policy_version()

        stats = buffer.get_staleness_stats()

        assert stats["mean_staleness"] >= 0, (
            f"mean_staleness should be >= 0, got {stats['mean_staleness']}"
        )
        assert stats["max_staleness"] >= 0, (
            f"max_staleness should be >= 0, got {stats['max_staleness']}"
        )
        assert stats["mean_staleness"] <= stats["max_staleness"], (
            f"mean_staleness ({stats['mean_staleness']}) should be <= "
            f"max_staleness ({stats['max_staleness']})"
        )

    @given(
        capacity=st.integers(min_value=2, max_value=20),
        n_adds=st.integers(min_value=3, max_value=50),
    )
    @settings(max_examples=30)
    def test_fifo_eviction_order(self, capacity: int, n_adds: int):
        """Oldest entries should be evicted first (FIFO)."""
        assume(n_adds > capacity)

        buffer = GFNReplayBuffer(capacity=capacity)

        for i in range(n_adds):
            buffer.add(make_buffer_entry(log_reward=float(-i)))

        # check that oldest entries are gone
        all_log_rewards = {e.log_reward for e in buffer}
        for i in range(n_adds - capacity):
            assert float(-i) not in all_log_rewards, f"Entry {i} should have been evicted"
