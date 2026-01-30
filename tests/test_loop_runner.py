"""Integration tests for the training loop: replay buffers, collectors, batching.

Exercises the GFN replay + EOS code path and prioritized sampling math.
"""

import math
import random
from typing import Any

import pytest
import torch

from synthstats.train.loop.collectors import CollectedTrajectory, TrajectoryCollector
from synthstats.train.loop.replay import BufferEntry, GFNReplayBuffer

# ---- helpers ----


def _make_entry(log_reward: float = 0.0, n_actions: int = 2) -> BufferEntry:
    return BufferEntry(
        actions=[{"type": "query", "payload": str(i)} for i in range(n_actions)],
        log_reward=log_reward,
        observations=[f"obs_{i}" for i in range(n_actions)],
        policy_version=0,
        temperature=1.0,
    )


def _make_trajectory(
    n_steps: int = 2,
    reward: float = 1.0,
    with_eos: bool = False,
) -> CollectedTrajectory:
    eos = torch.randn(n_steps) if with_eos else None
    return CollectedTrajectory(
        observations=[f"obs_{i}" for i in range(n_steps)],
        actions=[{"type": "query", "payload": str(i)} for i in range(n_steps)],
        log_probs=torch.randn(n_steps),
        entropy=torch.randn(n_steps).abs(),
        reward=reward,
        eos_logprobs=eos,
    )


class _MockPolicy:
    """Minimal policy that supports score_action with EOS logprob."""

    def __init__(self) -> None:
        self._last_eos_logprob_final: float | None = None

    def __call__(self, obs: str) -> tuple[dict, float, float]:
        action = {"type": "query", "payload": "mock"}
        self._last_eos_logprob_final = -2.0
        return action, -0.5, 0.1

    def score_action(self, obs: str, action: dict) -> tuple[float, float]:
        self._last_eos_logprob_final = -2.0
        return -0.5, 0.1


class _MockEnv:
    """Minimal env stub for collector tests."""

    chat_history: list[dict[str, str]] | None = None

    def init(self) -> tuple[list[dict[str, str]], dict]:
        self.chat_history = [{"role": "system", "content": "test"}]
        return self.chat_history, {}

    def step(self, action: str) -> dict:
        return {"reward": 1.0, "done": True, "observations": []}


# ---- GFN Replay Buffer tests ----


class TestGFNReplayBufferPrioritizedMath:
    """Verify prioritized sampling uses exp(α * log R)."""

    def test_uniform_when_alpha_zero(self):
        random.seed(99)  # seed at start for determinism
        buf = GFNReplayBuffer(capacity=100, prioritized=True, alpha=0.0)
        for i in range(5):
            buf.add(_make_entry(log_reward=float(i)))

        # alpha=0 → uniform: each of 5 entries should get ~20% of selections
        selected = buf._select_entries(2000)
        counts = [0] * 5
        for e in selected:
            counts[int(e.log_reward)] += 1
        # each should be ~400; 3-sigma bound for n=2000, p=0.2 is ~54
        for i, c in enumerate(counts):
            assert 300 < c < 500, f"Entry {i} got {c}/2000 samples, expected ~400"

    def test_high_reward_sampled_more(self):
        """Entries with higher log_reward should be sampled more frequently."""
        buf = GFNReplayBuffer(capacity=100, prioritized=True, alpha=1.0)

        # add 9 low-reward and 1 high-reward entries
        for _ in range(9):
            buf.add(_make_entry(log_reward=0.0))
        buf.add(_make_entry(log_reward=5.0))

        # sample many entries and check selection distribution
        random.seed(42)
        selected = buf._select_entries(1000)
        high_count = sum(1 for e in selected if e.log_reward == 5.0)

        # exp(1 * 5) / (9 * exp(0) + exp(5)) ≈ 148.4 / 157.4 ≈ 0.943
        # so high-reward entry should be selected ~94% of the time
        proportion = high_count / 1000
        assert proportion > 0.8, f"Expected >80% high-reward, got {proportion:.1%}"

    def test_numerical_stability_extreme_log_rewards(self):
        """Should not overflow with very large log_rewards."""
        buf = GFNReplayBuffer(capacity=100, prioritized=True, alpha=1.0)
        buf.add(_make_entry(log_reward=500.0))
        buf.add(_make_entry(log_reward=700.0))
        buf.add(_make_entry(log_reward=-100.0))

        # max-subtraction should prevent overflow
        selected = buf._select_entries(100)
        assert len(selected) == 100

    def test_all_equal_log_rewards_gives_uniform(self):
        buf = GFNReplayBuffer(capacity=100, prioritized=True, alpha=1.0)
        for _ in range(10):
            buf.add(_make_entry(log_reward=3.0))

        random.seed(123)
        selected = buf._select_entries(1000)
        assert len(selected) == 1000

    def test_negative_log_rewards(self):
        """Typical regime: log(reward) < 0 for reward in (0, 1)."""
        buf = GFNReplayBuffer(capacity=100, prioritized=True, alpha=1.0)
        for _ in range(9):
            buf.add(_make_entry(log_reward=-3.0))
        buf.add(_make_entry(log_reward=-0.5))

        random.seed(42)
        selected = buf._select_entries(1000)
        high_count = sum(1 for e in selected if e.log_reward == -0.5)
        # exp(1*(-0.5 - (-0.5))) = 1, exp(1*(-3 - (-0.5))) = exp(-2.5) ≈ 0.082
        # so P(high) ≈ 1 / (1 + 9*0.082) ≈ 0.576
        assert high_count / 1000 > 0.4, f"Expected >40%, got {high_count / 1000:.1%}"


class TestGFNReplayBufferPrioritizedMathOldBuffer:
    """Same tests for training/buffers/gfn_replay.py."""

    def test_high_reward_sampled_more(self):
        from synthstats.training.buffers import (
            BufferEntry as OldBufferEntry,
        )
        from synthstats.training.buffers import (
            GFNReplayBuffer as OldGFNReplayBuffer,
        )

        buf = OldGFNReplayBuffer(capacity=100, prioritized=True, alpha=1.0)
        for _ in range(9):
            buf.add(
                OldBufferEntry(
                    actions=[{"a": 1}],
                    log_reward=0.0,
                    observations=["obs"],
                )
            )
        buf.add(
            OldBufferEntry(
                actions=[{"a": 1}],
                log_reward=5.0,
                observations=["obs"],
            )
        )

        random.seed(42)
        selected = buf._select_entries(1000)
        high_count = sum(1 for e in selected if e.log_reward == 5.0)
        proportion = high_count / 1000
        assert proportion > 0.8, f"Expected >80% high-reward, got {proportion:.1%}"


# ---- EOS logprob in replay_entry tests ----


class _MockCollector:
    """Minimal ReplayCollector for buffer.sample() tests."""

    def replay_entry(self, entry: Any, temperature: float = 1.0) -> CollectedTrajectory | None:
        n = len(entry.actions)
        if n == 0:
            return None
        return CollectedTrajectory(
            observations=entry.observations[:n],
            actions=list(entry.actions),
            log_probs=torch.randn(n),
            entropy=torch.randn(n).abs(),
            reward=math.exp(entry.log_reward),
            eos_logprobs=torch.randn(n),
        )


class TestReplayEntryEOSLogprobs:
    """Verify replay_entry captures EOS logprobs from score_fn."""

    def test_replay_entry_captures_eos(self):
        """replay_entry should set eos_logprobs when score_fn provides them."""
        env = _MockEnv()
        policy = _MockPolicy()
        collector = TrajectoryCollector(env=env, policy_fn=policy)

        entry = _make_entry(log_reward=0.5, n_actions=2)
        result = collector.replay_entry(entry, temperature=1.0)

        assert result is not None
        assert result.eos_logprobs is not None
        assert result.eos_logprobs.shape == (2,)
        # each should be -2.0 (from _MockPolicy.score_action)
        assert torch.allclose(result.eos_logprobs, torch.tensor([-2.0, -2.0]))

    def test_replay_entry_no_eos_when_score_fn_doesnt_set_it(self):
        """replay_entry should have eos_logprobs=None if score_fn doesn't set attr."""

        class PolicyNoEOS:
            _last_eos_logprob_final = None

            def __call__(self, obs):
                return {"type": "query"}, -0.5, 0.1

            def score_action(self, obs, action):
                # does NOT set _last_eos_logprob_final
                self._last_eos_logprob_final = None
                return -0.5, 0.1

        env = _MockEnv()
        collector = TrajectoryCollector(env=env, policy_fn=PolicyNoEOS())
        entry = _make_entry(log_reward=0.5)
        result = collector.replay_entry(entry)

        assert result is not None
        assert result.eos_logprobs is None


class TestBatchBuildingEOSMixing:
    """Verify build_subtb_batch handles EOS logprobs consistently."""

    def test_all_with_eos_succeeds(self):
        from synthstats.train.loop.batching import build_subtb_batch

        trajs = [_make_trajectory(with_eos=True) for _ in range(3)]
        batch = build_subtb_batch(trajs)
        assert "eos_logprobs" in batch
        assert batch["eos_logprobs"].shape[0] == 3

    def test_none_with_eos_succeeds(self):
        from synthstats.train.loop.batching import build_subtb_batch

        trajs = [_make_trajectory(with_eos=False) for _ in range(3)]
        batch = build_subtb_batch(trajs)
        assert "eos_logprobs" not in batch

    def test_mixed_eos_raises(self):
        """Mixing trajectories with/without eos_logprobs should raise ValueError."""
        from synthstats.train.loop.batching import build_subtb_batch

        trajs = [
            _make_trajectory(with_eos=True),
            _make_trajectory(with_eos=False),
        ]
        with pytest.raises(ValueError, match="mixed eos_logprobs"):
            build_subtb_batch(trajs)

    def test_replay_and_fresh_both_have_eos(self):
        """Both collect() and replay_entry() must produce matching EOS presence."""
        from synthstats.train.loop.batching import build_subtb_batch

        env = _MockEnv()
        policy = _MockPolicy()
        collector = TrajectoryCollector(env=env, policy_fn=policy)

        # _MockPolicy sets EOS in both __call__ and score_action,
        # so both paths should produce EOS logprobs
        fresh = collector.collect(episodes=1)[0]
        assert fresh.eos_logprobs is not None, "fresh trajectory should have EOS"

        entry = _make_entry(log_reward=0.5, n_actions=1)
        entry.observations = entry.observations[:1]
        replay = collector.replay_entry(entry)
        assert replay is not None
        assert replay.eos_logprobs is not None, "replay trajectory should have EOS"

        # mixing should succeed (both have EOS)
        batch = build_subtb_batch([fresh, replay])
        assert "eos_logprobs" in batch
