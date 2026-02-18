import math

import pytest

from synthstats.train.data.collectors import TrajectoryCollector
from synthstats.train.data.replay import BufferEntry


class DummyEnv:
    pass


def dummy_policy(obs: str):
    return {"type": "submit_program", "payload": "x"}, 0.0, 0.0


def dummy_score(obs: str, action: dict):
    return 0.0, 0.0


def test_simple_collector_replay_entry_clamps_log_reward():
    collector = TrajectoryCollector(DummyEnv(), dummy_policy, score_fn=dummy_score)
    entry = BufferEntry(
        actions=[{"type": "submit_program", "payload": "x"}],
        log_reward=1000.0,  # deliberately extreme; exp(1000) overflows in Python
        observations=["obs"],
    )

    traj = collector.replay_entry(entry)

    assert traj is not None
    assert math.isfinite(traj.reward)
    assert traj.reward == pytest.approx(math.exp(700.0))
