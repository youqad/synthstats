from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from synthstats.envs.boxing_env import BoxingEnv, BoxingEnvConfig
from synthstats.judges.likelihood import LikelihoodJudge


def test_boxing_env_accepts_hydra_metadata_kwargs() -> None:
    """Hydra env configs include name/max_steps keys at the top level."""
    env = BoxingEnv(
        task=SimpleNamespace(name="dummy"),
        codec=SimpleNamespace(),
        executors={},
        judge=None,
        config=BoxingEnvConfig(max_turns=7),
        name="dugongs",
        max_steps=20,
    )

    assert env.max_turns == 7


def test_score_program_uses_judge(monkeypatch) -> None:
    env = object.__new__(BoxingEnv)
    env.task = SimpleNamespace(name="dummy")
    env.judge = LikelihoodJudge(beta=0.5)
    env.config = BoxingEnvConfig(reward_floor=1e-4, reward_scale=1.0)

    monkeypatch.setattr(BoxingEnv, "_execute_program_for_elpd", lambda self, code: 10.0)

    reward = BoxingEnv.score_program(env, "x = 1\n")
    assert reward == pytest.approx(math.exp(5.0))


def test_score_program_returns_floor_on_elpd_failure(monkeypatch) -> None:
    env = object.__new__(BoxingEnv)
    env.task = SimpleNamespace(name="dummy")
    env.judge = LikelihoodJudge(beta=0.5)
    env.config = BoxingEnvConfig(reward_floor=0.123, reward_scale=1.0)

    monkeypatch.setattr(BoxingEnv, "_execute_program_for_elpd", lambda self, code: None)

    reward = BoxingEnv.score_program(env, "x = 1\n")
    assert reward == pytest.approx(0.123)


def test_score_program_without_judge_returns_floor(monkeypatch, caplog) -> None:
    env = object.__new__(BoxingEnv)
    env.task = SimpleNamespace(name="dummy")
    env.judge = None
    env.config = BoxingEnvConfig(reward_floor=0.5, reward_scale=1.0)

    monkeypatch.setattr(BoxingEnv, "_execute_program_for_elpd", lambda self, code: 10.0)

    with caplog.at_level("WARNING"):
        reward = BoxingEnv.score_program(env, "x = 1\n")

    assert reward == pytest.approx(0.5)
    assert any("score_program called without a Judge" in r.message for r in caplog.records)
