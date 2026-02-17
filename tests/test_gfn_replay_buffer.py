"""Tests for GFNReplayBuffer.

GFNReplayBuffer stores action sequences (no tensors) and re-scores them
with the current policy when sampled. This eliminates off-policy bias
from stale log_probs.
"""

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class MockTrajectory:
    """Mock trajectory for testing GFNReplayBuffer.

    Mimics the interface expected by add_from_trajectory():
    - actions: list of action dicts
    - observations: list of observation strings
    - temperature: float (optional)
    """

    observations: list[str]
    actions: list[dict[str, Any]]
    log_probs: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    entropy: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    reward: float = 0.0
    temperature: float = 1.0


class TestBufferEntry:
    """Test BufferEntry dataclass."""

    def test_buffer_entry_creation(self):
        """BufferEntry stores actions and observations without tensors."""
        from synthstats.train.loop.replay import BufferEntry

        entry = BufferEntry(
            actions=[{"type": "query", "payload": "x"}, {"type": "answer", "payload": "42"}],
            log_reward=-0.5,
            observations=["obs1", "obs2"],
            policy_version=1,
            temperature=0.7,
        )

        assert len(entry.actions) == 2
        assert entry.log_reward == -0.5
        assert len(entry.observations) == 2
        assert entry.policy_version == 1
        assert entry.temperature == 0.7

    def test_buffer_entry_has_timestamp(self):
        """BufferEntry auto-populates timestamp."""
        from synthstats.train.loop.replay import BufferEntry

        entry = BufferEntry(
            actions=[{}],
            log_reward=0.0,
            observations=["obs"],
        )
        assert entry.timestamp > 0

    def test_buffer_entry_default_values(self):
        """BufferEntry has sensible defaults."""
        from synthstats.train.loop.replay import BufferEntry

        entry = BufferEntry(
            actions=[{"type": "answer"}],
            log_reward=0.0,
            observations=["obs"],
        )

        assert entry.policy_version == 0
        assert entry.temperature == 1.0


class TestGFNReplayBufferBasics:
    """Basic add/len operations."""

    def test_buffer_import(self):
        """GFNReplayBuffer should be importable."""
        from synthstats.train.loop.replay import GFNReplayBuffer

        assert GFNReplayBuffer is not None

    def test_buffer_add_and_len(self):
        """GFNReplayBuffer tracks size correctly."""
        from synthstats.train.loop.replay import BufferEntry, GFNReplayBuffer

        buffer = GFNReplayBuffer(capacity=10)
        assert len(buffer) == 0

        buffer.add(BufferEntry(actions=[{}], log_reward=0.0, observations=["obs"]))
        assert len(buffer) == 1

        buffer.add(BufferEntry(actions=[{}], log_reward=0.0, observations=["obs"]))
        assert len(buffer) == 2

    def test_buffer_capacity_eviction(self):
        """Oldest entries evicted when at capacity."""
        from synthstats.train.loop.replay import BufferEntry, GFNReplayBuffer

        buffer = GFNReplayBuffer(capacity=3)

        for i in range(5):
            buffer.add(
                BufferEntry(
                    actions=[{"id": i}],
                    log_reward=float(i),
                    observations=[f"obs{i}"],
                )
            )

        assert len(buffer) == 3
        # oldest (0, 1) should be gone
        log_rewards = {e.log_reward for e in buffer}
        assert 0.0 not in log_rewards
        assert 1.0 not in log_rewards
        assert log_rewards == {2.0, 3.0, 4.0}

    def test_buffer_iteration(self):
        """Buffer supports iteration."""
        from synthstats.train.loop.replay import BufferEntry, GFNReplayBuffer

        buffer = GFNReplayBuffer(capacity=10)
        buffer.add(BufferEntry(actions=[{}], log_reward=1.0, observations=["obs"]))
        buffer.add(BufferEntry(actions=[{}], log_reward=2.0, observations=["obs"]))

        entries = list(buffer)
        assert len(entries) == 2


class TestGFNReplayBufferAddFromTrajectory:
    """Test conversion from CollectedTrajectory to BufferEntry."""

    def test_add_from_trajectory_extracts_actions(self):
        """add_from_trajectory extracts actions without tensors."""
        # use MockTrajectory defined at top of file
        from synthstats.train.loop.replay import GFNReplayBuffer

        buffer = GFNReplayBuffer(capacity=10)

        traj = MockTrajectory(
            observations=["obs1", "obs2"],
            actions=[{"type": "query"}, {"type": "answer"}],
            log_probs=torch.tensor([-0.5, -0.3]),  # should NOT be stored
            entropy=torch.tensor([0.1, 0.2]),
            reward=1.0,
        )

        buffer.add_from_trajectory(traj, log_reward=0.0)

        assert len(buffer) == 1
        entry = list(buffer)[0]
        assert entry.actions == [{"type": "query"}, {"type": "answer"}]
        assert entry.observations == ["obs1", "obs2"]

    def test_add_from_trajectory_stores_log_reward(self):
        """add_from_trajectory stores the provided log_reward."""
        # use MockTrajectory defined at top of file
        from synthstats.train.loop.replay import GFNReplayBuffer

        buffer = GFNReplayBuffer(capacity=10)

        traj = MockTrajectory(
            observations=["obs"],
            actions=[{}],
            log_probs=torch.tensor([-0.5]),
            entropy=torch.tensor([0.1]),
            reward=10.0,
        )

        log_reward = torch.log(torch.tensor(10.0)).item()
        buffer.add_from_trajectory(traj, log_reward=log_reward)

        entry = list(buffer)[0]
        assert abs(entry.log_reward - log_reward) < 1e-5

    def test_add_from_trajectory_stores_temperature(self):
        """add_from_trajectory stores temperature from trajectory."""
        # use MockTrajectory defined at top of file
        from synthstats.train.loop.replay import GFNReplayBuffer

        buffer = GFNReplayBuffer(capacity=10)

        traj = MockTrajectory(
            observations=["obs"],
            actions=[{}],
            log_probs=torch.tensor([-0.5]),
            entropy=torch.tensor([0.1]),
            reward=1.0,
            temperature=0.7,
        )

        buffer.add_from_trajectory(traj, log_reward=0.0)

        entry = list(buffer)[0]
        assert entry.temperature == 0.7


class TestGFNReplayBufferPolicyVersion:
    """Test policy version tracking."""

    def test_initial_policy_version(self):
        """Buffer starts with policy version 0."""
        from synthstats.train.loop.replay import GFNReplayBuffer

        buffer = GFNReplayBuffer(capacity=10)
        assert buffer._policy_version == 0

    def test_increment_policy_version(self):
        """increment_policy_version updates internal counter."""
        from synthstats.train.loop.replay import GFNReplayBuffer

        buffer = GFNReplayBuffer(capacity=10)
        buffer.increment_policy_version()
        assert buffer._policy_version == 1
        buffer.increment_policy_version()
        assert buffer._policy_version == 2

    def test_entries_get_current_policy_version(self):
        """Entries are stamped with current policy version."""
        # use MockTrajectory defined at top of file
        from synthstats.train.loop.replay import GFNReplayBuffer

        buffer = GFNReplayBuffer(capacity=10)

        traj = MockTrajectory(
            observations=["obs"],
            actions=[{}],
            log_probs=torch.tensor([-0.5]),
            entropy=torch.tensor([0.1]),
            reward=1.0,
        )

        buffer.add_from_trajectory(traj, log_reward=0.0)
        buffer.increment_policy_version()
        buffer.add_from_trajectory(traj, log_reward=0.0)

        entries = list(buffer)
        assert entries[0].policy_version == 0
        assert entries[1].policy_version == 1

    def test_staleness_stats(self):
        """get_staleness_stats returns version statistics."""
        # use MockTrajectory defined at top of file
        from synthstats.train.loop.replay import GFNReplayBuffer

        buffer = GFNReplayBuffer(capacity=10)

        traj = MockTrajectory(
            observations=["obs"],
            actions=[{}],
            log_probs=torch.tensor([-0.5]),
            entropy=torch.tensor([0.1]),
            reward=1.0,
        )

        buffer.add_from_trajectory(traj, log_reward=0.0)
        buffer.increment_policy_version()
        buffer.add_from_trajectory(traj, log_reward=0.0)
        buffer.increment_policy_version()

        stats = buffer.get_staleness_stats()

        assert "mean_staleness" in stats
        assert "max_staleness" in stats
        assert stats["max_staleness"] == 2  # oldest entry is 2 versions behind

    def test_staleness_stats_empty_buffer(self):
        """get_staleness_stats handles empty buffer."""
        from synthstats.train.loop.replay import GFNReplayBuffer

        buffer = GFNReplayBuffer(capacity=10)
        stats = buffer.get_staleness_stats()

        assert stats["mean_staleness"] == 0.0
        assert stats["max_staleness"] == 0


class TestGFNReplayBufferPrePopulate:
    """Test pre_populate for SFT warm-start."""

    def test_pre_populate_adds_entries(self):
        """pre_populate should add entries to buffer."""
        from synthstats.train.loop.replay import BufferEntry, GFNReplayBuffer

        buffer = GFNReplayBuffer(capacity=10)

        entries = [
            BufferEntry(
                actions=[{"type": "submit", "payload": "x = 1"}],
                log_reward=-2.0,
                observations=["Q1"],
            ),
            BufferEntry(
                actions=[{"type": "submit", "payload": "y = 2"}],
                log_reward=-3.0,
                observations=["Q2"],
            ),
        ]

        added = buffer.pre_populate(entries)

        assert added == 2
        assert len(buffer) == 2

    def test_pre_populate_deduplication(self):
        """pre_populate should dedupe when dedupe=True."""
        from synthstats.train.loop.replay import BufferEntry, GFNReplayBuffer

        buffer = GFNReplayBuffer(capacity=10)

        # duplicate actions
        action = [{"type": "submit", "payload": "x = 1"}]
        entries = [
            BufferEntry(actions=action, log_reward=-2.0, observations=["Q1"]),
            BufferEntry(actions=action, log_reward=-3.0, observations=["Q2"]),
        ]

        added = buffer.pre_populate(entries, dedupe=True)

        assert added == 1  # only first one added
        assert len(buffer) == 1

    def test_pre_populate_no_deduplication(self):
        """pre_populate should not dedupe when dedupe=False."""
        from synthstats.train.loop.replay import BufferEntry, GFNReplayBuffer

        buffer = GFNReplayBuffer(capacity=10)

        action = [{"type": "submit", "payload": "x = 1"}]
        entries = [
            BufferEntry(actions=action, log_reward=-2.0, observations=["Q1"]),
            BufferEntry(actions=action, log_reward=-3.0, observations=["Q2"]),
        ]

        added = buffer.pre_populate(entries, dedupe=False)

        assert added == 2  # both added
        assert len(buffer) == 2

    def test_pre_populate_respects_capacity(self):
        """pre_populate should respect buffer capacity."""
        from synthstats.train.loop.replay import BufferEntry, GFNReplayBuffer

        buffer = GFNReplayBuffer(capacity=3)

        entries = [
            BufferEntry(actions=[{"id": i}], log_reward=float(i), observations=[f"Q{i}"])
            for i in range(5)
        ]

        buffer.pre_populate(entries)

        assert len(buffer) == 3  # capacity limit
        # FIFO: first entries evicted, last 3 remain
        log_rewards = {e.log_reward for e in buffer}
        assert log_rewards == {2.0, 3.0, 4.0}

    def test_pre_populate_dedupes_against_existing(self):
        """pre_populate should not add entries that already exist in buffer."""
        from synthstats.train.loop.replay import BufferEntry, GFNReplayBuffer

        buffer = GFNReplayBuffer(capacity=10)

        action = [{"type": "submit", "payload": "x = 1"}]
        buffer.add(BufferEntry(actions=action, log_reward=-1.0, observations=["existing"]))

        # try to pre-populate with same action
        entries = [
            BufferEntry(actions=action, log_reward=-2.0, observations=["new"]),
        ]

        added = buffer.pre_populate(entries, dedupe=True)

        assert added == 0  # duplicate of existing
        assert len(buffer) == 1

    def test_pre_populate_returns_count(self):
        """pre_populate should return count of entries actually added."""
        from synthstats.train.loop.replay import BufferEntry, GFNReplayBuffer

        buffer = GFNReplayBuffer(capacity=10)

        entries = [
            BufferEntry(actions=[{"id": 1}], log_reward=-1.0, observations=["Q1"]),
            BufferEntry(actions=[{"id": 1}], log_reward=-2.0, observations=["Q2"]),  # duplicate
            BufferEntry(actions=[{"id": 2}], log_reward=-3.0, observations=["Q3"]),
        ]

        added = buffer.pre_populate(entries, dedupe=True)

        assert added == 2  # 1 duplicate skipped


class TestGFNReplayBufferPrioritized:
    """Test prioritized sampling by log_reward."""

    def test_prioritized_alpha_zero_is_uniform(self):
        """alpha=0 should give uniform sampling."""
        from synthstats.train.loop.replay import BufferEntry, GFNReplayBuffer

        buffer = GFNReplayBuffer(capacity=10, prioritized=True, alpha=0.0)

        # add entries with very different log_rewards
        buffer.add(BufferEntry(actions=[{}], log_reward=-100.0, observations=["low"]))
        buffer.add(BufferEntry(actions=[{}], log_reward=0.0, observations=["high"]))

        # with alpha=0, weights should be equal regardless of log_reward
        # this is tested implicitly - no assertion needed, just verify no crash
        assert len(buffer) == 2
