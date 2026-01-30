"""Tests for DriverGFNReplayBuffer."""

from __future__ import annotations

import time

import pytest

from synthstats.distributed.driver_replay_buffer import (
    BufferEntry,
    DriverGFNReplayBuffer,
)


class TestBufferEntry:
    """Tests for BufferEntry dataclass."""

    def test_entry_creation(self) -> None:
        """Fields stored correctly."""
        entry = BufferEntry(
            action_token_ids=[101, 102, 103],
            prompt_token_ids=[1, 2, 3],
            log_reward=-0.5,
            policy_version=0,
            temperature=0.7,
        )

        assert entry.action_token_ids == [101, 102, 103]
        assert entry.prompt_token_ids == [1, 2, 3]
        assert entry.log_reward == -0.5
        assert entry.policy_version == 0
        assert entry.temperature == 0.7
        assert entry.terminated is True
        assert entry.trajectory_id is None

    def test_total_tokens(self) -> None:
        """Sums prompt + action tokens."""
        entry = BufferEntry(
            action_token_ids=[101, 102, 103],
            prompt_token_ids=[1, 2, 3, 4, 5],
            log_reward=-0.5,
        )

        assert entry.total_tokens() == 8  # 5 + 3

    def test_timestamp_auto_generated(self) -> None:
        """Timestamp defaults to now."""
        before = time.time()
        entry = BufferEntry(
            action_token_ids=[1],
            prompt_token_ids=[2],
            log_reward=0.0,
        )
        after = time.time()

        assert before <= entry.timestamp <= after


class TestDriverGFNReplayBuffer:
    """Tests for DriverGFNReplayBuffer."""

    def test_add_and_len(self) -> None:
        """add() increments length."""
        buffer = DriverGFNReplayBuffer(capacity=100)

        assert len(buffer) == 0

        entry = BufferEntry(
            action_token_ids=[1, 2, 3],
            prompt_token_ids=[10, 20],
            log_reward=-0.5,
        )
        buffer.add(entry)

        assert len(buffer) == 1

    def test_capacity_eviction(self) -> None:
        """FIFO eviction at capacity."""
        buffer = DriverGFNReplayBuffer(capacity=3)

        for i in range(5):
            buffer.add(
                BufferEntry(
                    action_token_ids=[i],
                    prompt_token_ids=[0],
                    log_reward=float(i),
                )
            )

        assert len(buffer) == 3
        # oldest entries (0, 1) should be evicted
        rewards = [e.log_reward for e in buffer]
        assert rewards == [2.0, 3.0, 4.0]

    def test_add_from_batch(self) -> None:
        """Batch add splits sequences correctly."""
        buffer = DriverGFNReplayBuffer(capacity=100)

        input_ids = [
            [1, 2, 3, 4, 5],  # prompt=[1,2], action=[3,4,5]
            [10, 20, 30, 40],  # prompt=[10,20,30], action=[40]
        ]
        prompt_lengths = [2, 3]
        log_rewards = [-0.5, -0.3]

        added = buffer.add_from_batch(
            input_ids=input_ids,
            prompt_lengths=prompt_lengths,
            log_rewards=log_rewards,
        )

        assert added == 2
        assert len(buffer) == 2

        entries = list(buffer)
        assert entries[0].prompt_token_ids == [1, 2]
        assert entries[0].action_token_ids == [3, 4, 5]
        assert entries[0].log_reward == -0.5

        assert entries[1].prompt_token_ids == [10, 20, 30]
        assert entries[1].action_token_ids == [40]
        assert entries[1].log_reward == -0.3

    def test_sample_uniform(self) -> None:
        """Uniform sampling returns entries."""
        buffer = DriverGFNReplayBuffer(capacity=100, prioritized=False)

        for i in range(10):
            buffer.add(
                BufferEntry(
                    action_token_ids=[i],
                    prompt_token_ids=[0],
                    log_reward=float(i),
                )
            )

        samples = buffer.sample(batch_size=5)
        assert len(samples) == 5
        assert all(isinstance(s, BufferEntry) for s in samples)

    def test_sample_prioritized(self) -> None:
        """Prioritized sampling favors high reward."""
        buffer = DriverGFNReplayBuffer(capacity=100, prioritized=True, alpha=1.0)

        # add entries with varying rewards
        for i in range(100):
            buffer.add(
                BufferEntry(
                    action_token_ids=[i],
                    prompt_token_ids=[0],
                    log_reward=float(i),  # higher i = higher reward
                )
            )

        # sample many times and check distribution
        samples = buffer.sample(batch_size=1000)
        avg_reward = sum(s.log_reward for s in samples) / len(samples)

        # with prioritized sampling, average should be > 50 (uniform mean)
        assert avg_reward > 50

    def test_sample_empty_raises(self) -> None:
        """Empty buffer raises ValueError."""
        buffer = DriverGFNReplayBuffer(capacity=100, min_entries_before_sample=1)

        with pytest.raises(ValueError, match="need at least"):
            buffer.sample(batch_size=1)

    def test_sample_respects_min_entries(self) -> None:
        """Respects min_entries_before_sample."""
        buffer = DriverGFNReplayBuffer(capacity=100, min_entries_before_sample=10)

        for i in range(5):
            buffer.add(
                BufferEntry(
                    action_token_ids=[i],
                    prompt_token_ids=[0],
                    log_reward=0.0,
                )
            )

        with pytest.raises(ValueError, match="need at least 10"):
            buffer.sample(batch_size=1)

    def test_policy_version_tracking(self) -> None:
        """Tracks staleness across versions."""
        buffer = DriverGFNReplayBuffer(capacity=100)

        assert buffer.policy_version == 0

        # add entry at version 0
        buffer.add(
            BufferEntry(
                action_token_ids=[1],
                prompt_token_ids=[0],
                log_reward=0.0,
                policy_version=buffer.policy_version,
            )
        )

        # increment version
        buffer.increment_policy_version()
        assert buffer.policy_version == 1

        # add entry at version 1
        buffer.add(
            BufferEntry(
                action_token_ids=[2],
                prompt_token_ids=[0],
                log_reward=0.0,
                policy_version=buffer.policy_version,
            )
        )

        buffer.increment_policy_version()
        assert buffer.policy_version == 2

        # check staleness
        stats = buffer.get_staleness_stats()
        assert stats["mean_staleness"] == 1.5  # (2-0 + 2-1) / 2
        assert stats["max_staleness"] == 2

    def test_get_stats(self) -> None:
        """Stats cover size, fill, staleness, rewards."""
        buffer = DriverGFNReplayBuffer(capacity=100)

        for i in range(10):
            buffer.add(
                BufferEntry(
                    action_token_ids=[i] * (i + 1),  # varying lengths
                    prompt_token_ids=[0, 0, 0],
                    log_reward=float(i) - 5,  # -5 to 4
                )
            )

        stats = buffer.get_stats()

        assert stats["size"] == 10
        assert stats["capacity"] == 100
        assert stats["fill_ratio"] == 0.1
        assert stats["mean_staleness"] == 0.0  # no version increments
        assert stats["max_staleness"] == 0
        assert stats["mean_log_reward"] == pytest.approx(-0.5)  # mean(-5..4)

    def test_clear(self) -> None:
        """clear() empties buffer."""
        buffer = DriverGFNReplayBuffer(capacity=100)

        for i in range(10):
            buffer.add(
                BufferEntry(
                    action_token_ids=[i],
                    prompt_token_ids=[0],
                    log_reward=0.0,
                )
            )

        assert len(buffer) == 10

        buffer.clear()
        assert len(buffer) == 0

    def test_iteration(self) -> None:
        """Iteration preserves insertion order."""
        buffer = DriverGFNReplayBuffer(capacity=100)

        for i in range(5):
            buffer.add(
                BufferEntry(
                    action_token_ids=[i],
                    prompt_token_ids=[0],
                    log_reward=float(i),
                )
            )

        entries = list(buffer)
        assert len(entries) == 5
        assert [e.log_reward for e in entries] == [0.0, 1.0, 2.0, 3.0, 4.0]
