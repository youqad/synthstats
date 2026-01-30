"""Tests for Trajectory Replay Buffer - WRITTEN FIRST per TDD."""

import pytest

from synthstats.core.types import Message, Reward, Trajectory


def make_trajectory(reward_total: float = 0.5) -> Trajectory:
    """Helper to create a minimal trajectory for testing."""
    return Trajectory(
        messages=[Message(role="user", content="test")],
        token_ids=[[1, 2, 3]],
        token_logprobs=[[-0.5, -0.3, -0.2]],
        loss_mask=[[True, True, True]],
        reward=Reward(total=reward_total, components={}, info={}),
    )


class TestReplayBufferBasics:
    """Basic add/len/sample operations."""

    def test_buffer_add_and_len(self):
        from synthstats.training.buffers import ReplayBuffer

        buffer = ReplayBuffer(capacity=10)
        assert len(buffer) == 0

        buffer.add(make_trajectory())
        assert len(buffer) == 1

        buffer.add(make_trajectory())
        assert len(buffer) == 2

    def test_buffer_sample_returns_trajectories(self):
        from synthstats.training.buffers import ReplayBuffer

        buffer = ReplayBuffer(capacity=10)
        traj = make_trajectory()
        buffer.add(traj)

        batch = buffer.sample(batch_size=1)

        assert len(batch) == 1
        assert isinstance(batch[0], Trajectory)
        assert batch[0].reward.total == traj.reward.total

    def test_buffer_capacity_eviction(self):
        from synthstats.training.buffers import ReplayBuffer

        buffer = ReplayBuffer(capacity=3)

        # add 5 trajectories with different rewards to track eviction
        for i in range(5):
            buffer.add(make_trajectory(reward_total=float(i)))

        assert len(buffer) == 3

        # oldest (reward=0, 1) should be evicted, newest (2, 3, 4) remain
        # use iteration to check contents directly (sampling is probabilistic)
        all_rewards = {t.reward.total for t in buffer}
        assert 0.0 not in all_rewards
        assert 1.0 not in all_rewards
        assert all_rewards == {2.0, 3.0, 4.0}

    def test_buffer_sample_with_replacement(self):
        from synthstats.training.buffers import ReplayBuffer

        buffer = ReplayBuffer(capacity=10)
        buffer.add(make_trajectory())

        # sample more than buffer size - should work with replacement
        batch = buffer.sample(batch_size=5)
        assert len(batch) == 5
        assert all(isinstance(t, Trajectory) for t in batch)


class TestReplayBufferEdgeCases:
    """Edge cases and error handling."""

    def test_empty_buffer_sample_raises(self):
        from synthstats.training.buffers import ReplayBuffer

        buffer = ReplayBuffer(capacity=10)

        with pytest.raises(ValueError, match="empty"):
            buffer.sample(batch_size=1)

    def test_zero_batch_size(self):
        from synthstats.training.buffers import ReplayBuffer

        buffer = ReplayBuffer(capacity=10)
        buffer.add(make_trajectory())

        batch = buffer.sample(batch_size=0)
        assert batch == []

    def test_capacity_one(self):
        from synthstats.training.buffers import ReplayBuffer

        buffer = ReplayBuffer(capacity=1)
        buffer.add(make_trajectory(reward_total=1.0))
        buffer.add(make_trajectory(reward_total=2.0))

        assert len(buffer) == 1
        batch = buffer.sample(batch_size=1)
        assert batch[0].reward.total == 2.0


class TestPrioritizedBuffer:
    """Prioritized sampling weighted by reward."""

    def test_prioritized_buffer_weights_by_reward(self):
        from synthstats.training.buffers import ReplayBuffer

        buffer = ReplayBuffer(capacity=100, prioritized=True)

        # add one high-reward and many low-reward trajectories
        for _ in range(10):
            buffer.add(make_trajectory(reward_total=0.01))
        buffer.add(make_trajectory(reward_total=10.0))  # much higher

        # sample many times and count high-reward samples
        high_reward_count = 0
        n_samples = 500
        for _ in range(n_samples):
            batch = buffer.sample(batch_size=1)
            if batch[0].reward.total > 5.0:
                high_reward_count += 1

        # with prioritization, high-reward should be sampled more often
        # uniform would give ~1/11 = 9%, prioritized should be much higher
        proportion = high_reward_count / n_samples
        assert proportion > 0.3, f"Expected >30% high-reward samples, got {proportion:.1%}"

    def test_prioritized_with_zero_rewards(self):
        """Zero rewards should still be sampleable."""
        from synthstats.training.buffers import ReplayBuffer

        buffer = ReplayBuffer(capacity=10, prioritized=True)
        buffer.add(make_trajectory(reward_total=0.0))
        buffer.add(make_trajectory(reward_total=0.0))

        batch = buffer.sample(batch_size=2)
        assert len(batch) == 2

    def test_prioritized_with_negative_rewards(self):
        """Negative rewards should be handled (shifted to positive)."""
        from synthstats.training.buffers import ReplayBuffer

        buffer = ReplayBuffer(capacity=10, prioritized=True)
        buffer.add(make_trajectory(reward_total=-1.0))
        buffer.add(make_trajectory(reward_total=-0.5))
        buffer.add(make_trajectory(reward_total=0.5))

        batch = buffer.sample(batch_size=3)
        assert len(batch) == 3

    def test_prioritized_alpha_parameter(self):
        """Alpha controls prioritization strength."""
        from synthstats.training.buffers import ReplayBuffer

        # alpha=0 should be uniform
        buffer_uniform = ReplayBuffer(capacity=100, prioritized=True, alpha=0.0)
        for _ in range(10):
            buffer_uniform.add(make_trajectory(reward_total=0.01))
        buffer_uniform.add(make_trajectory(reward_total=10.0))

        high_count_uniform = sum(
            1 for _ in range(200) if buffer_uniform.sample(1)[0].reward.total > 5.0
        )

        # alpha=1 should be strongly prioritized
        buffer_strong = ReplayBuffer(capacity=100, prioritized=True, alpha=1.0)
        for _ in range(10):
            buffer_strong.add(make_trajectory(reward_total=0.01))
        buffer_strong.add(make_trajectory(reward_total=10.0))

        high_count_strong = sum(
            1 for _ in range(200) if buffer_strong.sample(1)[0].reward.total > 5.0
        )

        # strong prioritization should yield more high-reward samples
        assert high_count_strong > high_count_uniform


class TestReplayBufferIteration:
    """Test iteration and access patterns."""

    def test_buffer_stores_full_trajectory(self):
        """Ensure all trajectory fields are preserved."""
        from synthstats.training.buffers import ReplayBuffer

        buffer = ReplayBuffer(capacity=10)
        original = Trajectory(
            messages=[
                Message(role="system", content="sys"),
                Message(role="user", content="hello"),
                Message(role="assistant", content="hi"),
            ],
            token_ids=[[10, 20], [30, 40, 50]],
            token_logprobs=[[-0.1, -0.2], [-0.3, -0.4, -0.5]],
            loss_mask=[[True, False], [True, True, True]],
            reward=Reward(total=0.75, components={"a": 0.5}, info={"key": "val"}),
        )
        buffer.add(original)

        sampled = buffer.sample(batch_size=1)[0]

        assert len(sampled.messages) == 3
        assert sampled.messages[1].content == "hello"
        assert sampled.token_ids == [[10, 20], [30, 40, 50]]
        assert sampled.token_logprobs == [[-0.1, -0.2], [-0.3, -0.4, -0.5]]
        assert sampled.loss_mask == [[True, False], [True, True, True]]
        assert sampled.reward.total == 0.75
        assert sampled.reward.components == {"a": 0.5}
        assert sampled.reward.info == {"key": "val"}


class TestReplayBufferStateDictCheckpointing:
    """Test state_dict/load_state_dict for checkpointing."""

    def test_state_dict_returns_serializable_dict(self):
        from synthstats.training.buffers import ReplayBuffer

        buffer = ReplayBuffer(capacity=10, prioritized=True, alpha=0.5)
        buffer.add(make_trajectory(reward_total=0.5))
        buffer.add(make_trajectory(reward_total=0.8))

        state = buffer.state_dict()
        assert isinstance(state, dict)
        assert state["capacity"] == 10
        assert state["prioritized"] is True
        assert state["alpha"] == 0.5
        assert len(state["trajectories"]) == 2

    def test_load_state_dict_restores_buffer(self):
        from synthstats.training.buffers import ReplayBuffer

        buffer = ReplayBuffer(capacity=10, prioritized=True, alpha=0.5)
        buffer.add(make_trajectory(reward_total=0.5))
        buffer.add(make_trajectory(reward_total=0.8))
        state = buffer.state_dict()

        new_buffer = ReplayBuffer(capacity=1)  # different initial config
        new_buffer.load_state_dict(state)

        assert len(new_buffer) == 2
        assert new_buffer._capacity == 10
        assert new_buffer._prioritized is True
        assert new_buffer._alpha == 0.5

    def test_state_dict_roundtrip_preserves_trajectories(self):
        from synthstats.training.buffers import ReplayBuffer

        buffer = ReplayBuffer(capacity=10)
        original = Trajectory(
            messages=[
                Message(role="user", content="test"),
                Message(role="assistant", content="response"),
            ],
            token_ids=[[1, 2], [3, 4]],
            token_logprobs=[[-0.1, -0.2], [-0.3, -0.4]],
            loss_mask=[[True, True], [False, True]],
            reward=Reward(total=0.9, components={"c": 0.9}, info={"k": 1}),
            eos_logprobs=[[-0.5], [-0.6]],
        )
        buffer.add(original)
        state = buffer.state_dict()

        new_buffer = ReplayBuffer(capacity=5)
        new_buffer.load_state_dict(state)

        restored = list(new_buffer)[0]
        assert len(restored.messages) == 2
        assert restored.messages[0].content == "test"
        assert restored.token_ids == original.token_ids
        assert restored.token_logprobs == original.token_logprobs
        assert restored.loss_mask == original.loss_mask
        assert restored.reward.total == original.reward.total
        assert restored.eos_logprobs == original.eos_logprobs

    def test_empty_buffer_state_dict(self):
        from synthstats.training.buffers import ReplayBuffer

        buffer = ReplayBuffer(capacity=10)
        state = buffer.state_dict()

        assert state["trajectories"] == []
        assert state["capacity"] == 10

        new_buffer = ReplayBuffer(capacity=5)
        new_buffer.load_state_dict(state)
        assert len(new_buffer) == 0
        assert new_buffer._capacity == 10
