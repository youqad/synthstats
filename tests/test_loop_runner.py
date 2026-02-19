"""Training loop integration: replay buffers, collectors, batching.

Exercises GFN replay + EOS code path and prioritized sampling math.
"""

import math
import random
from typing import Any

import pytest
import torch

from synthstats.train.data.collectors import CollectedTrajectory, TrajectoryCollector
from synthstats.train.data.replay import BufferEntry, GFNReplayBuffer

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
    def __init__(self) -> None:
        self._last_eos_logprob_final: float | None = None

    def __call__(self, obs: str) -> tuple[dict, float, float]:
        action = {"type": "query", "payload": "mock"}
        self._last_eos_logprob_final = -2.0
        return action, -0.5, 0.1

    def sample_with_eos(
        self, obs: str, temperature: float = 1.0
    ) -> tuple[dict, float, float, float]:
        action = {"type": "query", "payload": "mock"}
        self._last_eos_logprob_final = -2.0
        return action, -0.5, 0.1, -2.0

    def score_action(self, obs: str, action: dict) -> tuple[float, float]:
        self._last_eos_logprob_final = -2.0
        return -0.5, 0.1

    def score_action_with_eos(
        self,
        obs: str,
        action: dict,
        temperature: float = 1.0,
    ) -> tuple[float, float, float]:
        self._last_eos_logprob_final = -2.0
        return -0.5, 0.1, -2.0


class _MockEnv:
    chat_history: list[dict[str, str]] | None = None

    def init(self) -> tuple[list[dict[str, str]], dict]:
        self.chat_history = [{"role": "system", "content": "test"}]
        return self.chat_history, {}

    def step(self, action: str) -> dict:
        return {"reward": 1.0, "done": True, "observations": []}


# ---- GFN Replay Buffer tests ----


class TestGFNReplayBufferPrioritizedMath:
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
        """typical regime: log(reward) < 0 for reward in (0, 1)."""
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


# ---- EOS logprob in replay_entry tests ----


class _MockCollector:
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
    def test_replay_entry_captures_eos(self):
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
    def test_all_with_eos_succeeds(self):
        from synthstats.train.data.collate import build_subtb_batch

        trajs = [_make_trajectory(with_eos=True) for _ in range(3)]
        batch = build_subtb_batch(trajs)
        assert "eos_logprobs" in batch
        assert batch["eos_logprobs"].shape[0] == 3

    def test_none_with_eos_succeeds(self):
        from synthstats.train.data.collate import build_subtb_batch

        trajs = [_make_trajectory(with_eos=False) for _ in range(3)]
        batch = build_subtb_batch(trajs)
        assert "eos_logprobs" not in batch

    def test_mixed_eos_raises(self):
        from synthstats.train.data.collate import build_subtb_batch

        trajs = [
            _make_trajectory(with_eos=True),
            _make_trajectory(with_eos=False),
        ]
        with pytest.raises(ValueError, match="mixed eos_logprobs"):
            build_subtb_batch(trajs)

    def test_replay_and_fresh_both_have_eos(self):
        from synthstats.train.data.collate import build_subtb_batch

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


class TestLoopRunnerEOSStripping:
    def test_strips_eos_when_fresh_has_eos_and_replay_lacks_it(self):
        from synthstats.train.loop.loop_runner import LoopConfig, LoopRunner

        captured_batches: list[dict] = []

        class _CaptureLearner:
            def update(self, batch: dict) -> dict[str, float]:
                captured_batches.append(batch)
                return {"loss": 0.0}

            @property
            def logZ(self) -> float:
                return 0.0

        class _FreshEOSCollector:
            def collect(self, episodes: int, temperature: float, **kw):
                return [
                    _make_trajectory(n_steps=2, reward=1.0, with_eos=True) for _ in range(episodes)
                ]

            def replay_entry(self, entry, temperature: float = 1.0):
                return _make_trajectory(n_steps=2, reward=1.0, with_eos=False)

        collector = _FreshEOSCollector()
        learner = _CaptureLearner()
        cfg = LoopConfig(
            batch_size=4,
            replay_buffer_size=20,
            replay_ratio=0.5,
            use_gfn_replay=True,
        )
        loop = LoopRunner(collector=collector, learner=learner, config=cfg)

        # step 1: fills buffer, no replay yet (buffer too small)
        loop.train_step()
        # step 2: now buffer has entries, should mix fresh + replay
        loop.train_step()

        assert len(captured_batches) == 2
        # second batch should not have eos_logprobs (stripped by guard)
        assert "eos_logprobs" not in captured_batches[1]

    def test_no_strip_when_all_have_eos(self):
        from synthstats.train.loop.loop_runner import LoopConfig, LoopRunner

        captured_batches: list[dict] = []

        class _CaptureLearner:
            def update(self, batch: dict) -> dict[str, float]:
                captured_batches.append(batch)
                return {"loss": 0.0}

            @property
            def logZ(self) -> float:
                return 0.0

        class _AllEOSCollector:
            def collect(self, episodes: int, temperature: float, **kw):
                return [
                    _make_trajectory(n_steps=2, reward=1.0, with_eos=True) for _ in range(episodes)
                ]

            def replay_entry(self, entry, temperature: float = 1.0):
                return _make_trajectory(n_steps=2, reward=1.0, with_eos=True)

        collector = _AllEOSCollector()
        learner = _CaptureLearner()
        cfg = LoopConfig(
            batch_size=4,
            replay_buffer_size=20,
            replay_ratio=0.5,
            use_gfn_replay=True,
        )
        loop = LoopRunner(collector=collector, learner=learner, config=cfg)

        loop.train_step()  # fill buffer
        loop.train_step()  # mix fresh + replay, both have EOS

        assert len(captured_batches) == 2
        assert "eos_logprobs" in captured_batches[1]

    def test_non_dataclass_trajectory_not_replaced(self):
        """Non-dataclass trajectories skip replace() without crashing."""
        from synthstats.train.loop.loop_runner import LoopConfig, LoopRunner

        captured_trajectories: list[list] = []

        class _CaptureLearner:
            def update(self, batch: Any) -> dict[str, float]:
                return {"loss": 0.0}

            @property
            def logZ(self) -> float:
                return 0.0

        class _PlainTrajectory:
            """Non-dataclass trajectory without eos_logprobs attr."""

            def __init__(self, reward: float = 1.0, n_steps: int = 2):
                self.observations = [f"obs_{i}" for i in range(n_steps)]
                self.actions = [{"type": "query", "payload": str(i)} for i in range(n_steps)]
                self.log_probs = torch.randn(n_steps)
                self.entropy = torch.randn(n_steps).abs()
                self.reward = reward

        class _MixedCollector:
            def collect(self, episodes: int, temperature: float, **kw):
                return [
                    _make_trajectory(n_steps=2, reward=1.0, with_eos=True) for _ in range(episodes)
                ]

            def replay_entry(self, entry, temperature: float = 1.0):
                return _PlainTrajectory()

        def _capture_batch_builder(trajs, **kw):
            captured_trajectories.append(trajs)
            return {"log_pf": torch.zeros(1)}

        collector = _MixedCollector()
        learner = _CaptureLearner()
        cfg = LoopConfig(
            batch_size=4,
            replay_buffer_size=20,
            replay_ratio=0.5,
            use_gfn_replay=True,
        )
        loop = LoopRunner(
            collector=collector,
            learner=learner,
            config=cfg,
            batch_builder=_capture_batch_builder,
        )

        loop.train_step()
        loop.train_step()

        assert len(captured_trajectories) == 2
        # second batch mixes dataclass + plain; plain should pass through unchanged
        mixed_batch = captured_trajectories[1]
        plain_count = sum(1 for t in mixed_batch if isinstance(t, _PlainTrajectory))
        assert plain_count > 0, "non-dataclass trajectories should survive the guard"
        # all should have eos_logprobs stripped or absent
        for t in mixed_batch:
            eos = getattr(t, "eos_logprobs", None)
            assert eos is None, f"expected eos_logprobs=None, got {eos}"

    def test_no_strip_when_none_have_eos(self):
        from synthstats.train.loop.loop_runner import LoopConfig, LoopRunner

        captured_batches: list[dict] = []

        class _CaptureLearner:
            def update(self, batch: dict) -> dict[str, float]:
                captured_batches.append(batch)
                return {"loss": 0.0}

            @property
            def logZ(self) -> float:
                return 0.0

        class _NoEOSCollector:
            def collect(self, episodes: int, temperature: float, **kw):
                return [
                    _make_trajectory(n_steps=2, reward=1.0, with_eos=False) for _ in range(episodes)
                ]

            def replay_entry(self, entry, temperature: float = 1.0):
                return _make_trajectory(n_steps=2, reward=1.0, with_eos=False)

        collector = _NoEOSCollector()
        learner = _CaptureLearner()
        cfg = LoopConfig(
            batch_size=4,
            replay_buffer_size=20,
            replay_ratio=0.5,
            use_gfn_replay=True,
        )
        loop = LoopRunner(collector=collector, learner=learner, config=cfg)

        loop.train_step()
        loop.train_step()

        assert len(captured_batches) == 2
        assert "eos_logprobs" not in captured_batches[1]
