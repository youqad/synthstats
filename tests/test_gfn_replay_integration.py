"""Integration tests for GFN replay + EOS logprobs.

Tests the critical path where fresh trajectories (with EOS logprobs) are
mixed with replay trajectories in the same batch.
"""

import math

import pytest
import torch


class MockScoreFn:
    """Score function that tracks EOS logprob like HFPolicy."""

    def __init__(self, eos_logprob: float = -0.5):
        self._eos_logprob = eos_logprob
        self._last_eos_logprob_final = eos_logprob

    def __call__(self, obs: str, action: dict, temperature: float = 1.0):
        self._last_eos_logprob_final = self._eos_logprob
        logp = torch.tensor(-0.1)
        ent = torch.tensor(0.5)
        return logp, ent

    def score_action_with_eos(
        self,
        obs: str,
        action: dict,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._last_eos_logprob_final = self._eos_logprob
        logp = torch.tensor(-0.1)
        ent = torch.tensor(0.5)
        eos = torch.tensor(self._eos_logprob)
        return logp, ent, eos


class MockEnv:
    """Minimal env for testing collectors."""

    def __init__(self):
        self.chat_history = []
        self._step_count = 0

    def init(self):
        self.chat_history = [{"role": "user", "content": "hello"}]
        self._step_count = 0
        return self.chat_history, {}

    def step(self, action: str):
        self._step_count += 1
        done = self._step_count >= 2
        return {
            "observations": [{"role": "user", "content": "ok"}],
            "reward": 1.0 if done else 0.0,
            "done": done,
        }


class MockPolicyFn:
    """Policy that sets _last_eos_logprob_final like HFPolicy."""

    def __init__(self):
        self._last_eos_logprob_final = -0.5

    def __call__(self, obs: str, temperature: float = 1.0):
        action = {"type": "query", "payload": "test"}
        logp = torch.tensor(-0.2)
        ent = torch.tensor(0.6)
        return action, logp, ent

    def sample_with_eos(
        self,
        obs: str,
        temperature: float = 1.0,
    ) -> tuple[dict[str, str], torch.Tensor, torch.Tensor, torch.Tensor]:
        action = {"type": "query", "payload": "test"}
        logp = torch.tensor(-0.2)
        ent = torch.tensor(0.6)
        eos = torch.tensor(self._last_eos_logprob_final)
        return action, logp, ent, eos

    def score_action(self, obs: str, action: dict, temperature: float = 1.0):
        self._last_eos_logprob_final = -0.5
        logp = torch.tensor(-0.2)
        ent = torch.tensor(0.6)
        return logp, ent

    def score_action_with_eos(
        self,
        obs: str,
        action: dict,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._last_eos_logprob_final = -0.5
        logp = torch.tensor(-0.2)
        ent = torch.tensor(0.6)
        eos = torch.tensor(self._last_eos_logprob_final)
        return logp, ent, eos


class TestEOSLogprobsInReplay:
    """EOS logprobs in replay vs fresh trajectories."""

    def test_collect_sets_eos_logprobs(self):
        from synthstats.train.loop.collectors import TrajectoryCollector

        env = MockEnv()
        policy_fn = MockPolicyFn()
        collector = TrajectoryCollector(env, policy_fn)

        trajs = collector.collect(episodes=1, temperature=1.0)

        assert len(trajs) == 1
        traj = trajs[0]
        assert traj.eos_logprobs is not None
        assert traj.eos_logprobs.shape[0] == traj.log_probs.shape[0]

    def test_replay_entry_sets_eos_logprobs(self):
        from synthstats.train.loop.collectors import TrajectoryCollector
        from synthstats.train.loop.replay import BufferEntry

        env = MockEnv()
        policy_fn = MockPolicyFn()
        score_fn = MockScoreFn(eos_logprob=-0.5)
        collector = TrajectoryCollector(env, policy_fn, score_fn=score_fn)

        entry = BufferEntry(
            actions=[{"type": "query", "payload": "a"}, {"type": "query", "payload": "b"}],
            log_reward=0.0,
            observations=['[{"role": "user", "content": "x"}]'] * 2,
            policy_version=0,
            temperature=1.0,
        )

        traj = collector.replay_entry(entry, temperature=1.0)

        assert traj is not None
        assert traj.eos_logprobs is not None
        assert traj.eos_logprobs.shape[0] == traj.log_probs.shape[0]

    def test_batch_builder_accepts_mixed_eos(self):
        from synthstats.train.loop.batching import build_subtb_batch
        from synthstats.train.loop.collectors import CollectedTrajectory

        trajs = [
            CollectedTrajectory(
                observations=["a", "b"],
                actions=[{"x": 1}, {"x": 2}],
                log_probs=torch.tensor([-0.1, -0.2]),
                entropy=torch.tensor([0.5, 0.5]),
                reward=1.0,
                eos_logprobs=torch.tensor([-0.5, -0.5]),
            ),
            CollectedTrajectory(
                observations=["c"],
                actions=[{"y": 1}],
                log_probs=torch.tensor([-0.3]),
                entropy=torch.tensor([0.4]),
                reward=0.5,
                eos_logprobs=torch.tensor([-0.6]),
            ),
        ]

        batch = build_subtb_batch(trajs)

        assert "eos_logprobs" in batch
        assert batch["eos_logprobs"].shape == batch["log_probs"].shape

    def test_batch_builder_rejects_mixed_eos(self):
        from synthstats.train.loop.batching import build_subtb_batch
        from synthstats.train.loop.collectors import CollectedTrajectory

        trajs = [
            CollectedTrajectory(
                observations=["a"],
                actions=[{"x": 1}],
                log_probs=torch.tensor([-0.1]),
                entropy=torch.tensor([0.5]),
                reward=1.0,
                eos_logprobs=torch.tensor([-0.5]),
            ),
            CollectedTrajectory(
                observations=["b"],
                actions=[{"y": 1}],
                log_probs=torch.tensor([-0.3]),
                entropy=torch.tensor([0.4]),
                reward=0.5,
                eos_logprobs=None,
            ),
        ]

        with pytest.raises(ValueError, match="mixed eos_logprobs"):
            build_subtb_batch(trajs)


class TestPrioritizedSamplingMath:
    """Prioritized sampling weight computation."""

    def test_prioritized_sampling_is_reward_proportional(self):
        from synthstats.train.loop.replay import BufferEntry, GFNReplayBuffer

        buffer = GFNReplayBuffer(capacity=100, prioritized=True, alpha=1.0)

        log_rewards = [1.0, 2.0, 3.0]
        for lr in log_rewards:
            buffer.add(
                BufferEntry(
                    actions=[{"x": 1}],
                    log_reward=lr,
                    observations=["obs"],
                )
            )

        max_lr = max(log_rewards)
        expected_weights = [math.exp(1.0 * (lr - max_lr)) for lr in log_rewards]
        total = sum(expected_weights)
        expected_probs = [w / total for w in expected_weights]

        counts = {lr: 0 for lr in log_rewards}
        n_samples = 5000

        class MockCollector:
            def replay_entry(self, entry, temperature=1.0):
                return entry

        MockCollector()
        for _ in range(n_samples):
            entries = buffer._select_entries(1)
            counts[entries[0].log_reward] += 1

        empirical_probs = [counts[lr] / n_samples for lr in log_rewards]

        for lr, expected, empirical in zip(
            log_rewards, expected_probs, empirical_probs, strict=False
        ):
            assert abs(expected - empirical) < 0.05, (
                f"log_reward={lr}: expected {expected:.3f}, got {empirical:.3f}"
            )

    def test_alpha_zero_gives_uniform(self):
        from synthstats.train.loop.replay import BufferEntry, GFNReplayBuffer

        buffer = GFNReplayBuffer(capacity=100, prioritized=True, alpha=0.0)

        for lr in [0.0, 10.0, 20.0]:
            buffer.add(
                BufferEntry(
                    actions=[{"x": 1}],
                    log_reward=lr,
                    observations=["obs"],
                )
            )

        counts = {0.0: 0, 10.0: 0, 20.0: 0}
        n_samples = 3000

        for _ in range(n_samples):
            entries = buffer._select_entries(1)
            counts[entries[0].log_reward] += 1

        for lr, count in counts.items():
            prop = count / n_samples
            assert 0.25 < prop < 0.42, f"log_reward={lr}: expected ~0.33, got {prop:.3f}"


class TestGFNReplayBufferIntegration:
    """End-to-end GFN replay buffer with collector."""

    def test_full_replay_cycle(self):
        from synthstats.train.loop.collectors import TrajectoryCollector
        from synthstats.train.loop.replay import BufferEntry, GFNReplayBuffer

        env = MockEnv()
        policy_fn = MockPolicyFn()
        score_fn = MockScoreFn()
        collector = TrajectoryCollector(env, policy_fn, score_fn=score_fn)

        buffer = GFNReplayBuffer(capacity=10, prioritized=True, alpha=1.0)

        for i in range(5):
            entry = BufferEntry(
                actions=[{"type": "query", "payload": str(i)}],
                log_reward=float(i),
                observations=['[{"role": "user", "content": "test"}]'],
            )
            buffer.add(entry)

        trajs = buffer.sample(batch_size=3, collector=collector)

        assert len(trajs) == 3
        for traj in trajs:
            assert traj.log_probs.numel() == 1
            assert traj.entropy.numel() == 1
            assert traj.eos_logprobs is not None
