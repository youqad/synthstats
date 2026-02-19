"""Tests for environment builders."""

import pytest
from omegaconf import OmegaConf

from synthstats.envs.builders import build_env


def test_build_env_requires_target() -> None:
    with pytest.raises(ValueError, match="cfg.env must define _target_"):
        build_env(OmegaConf.create({"env": {}}))


def test_build_env_uses_env_judge_without_top_level_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = OmegaConf.create(
        {
            "env": {
                "_target_": "synthstats.envs.boxing_env.BoxingEnv",
                "judge": {
                    "_target_": "synthstats.judges.likelihood.LikelihoodJudge",
                    "beta": 0.5,
                },
            }
        }
    )
    captured: dict[str, object] = {}

    import hydra.utils

    def _fake_instantiate(env_cfg):
        captured["env_cfg"] = env_cfg
        return object()

    monkeypatch.setattr(hydra.utils, "instantiate", _fake_instantiate)

    _ = build_env(cfg)

    env_cfg = captured["env_cfg"]
    assert env_cfg["judge"]["_target_"] == "synthstats.judges.likelihood.LikelihoodJudge"
    assert env_cfg["judge"]["beta"] == 0.5


def test_build_env_injects_top_level_judge_override(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = OmegaConf.create(
        {
            "env": {
                "_target_": "synthstats.envs.boxing_env.BoxingEnv",
                "judge": {
                    "_target_": "synthstats.judges.likelihood.LikelihoodJudge",
                    "beta": 1.0,
                },
            },
            "judge": {
                "_target_": "synthstats.judges.llm_critic.LLMCriticJudge",
                "model_name": "openai/gpt-5-mini",
            },
        }
    )
    captured: dict[str, object] = {}

    import hydra.utils

    def _fake_instantiate(env_cfg):
        captured["env_cfg"] = env_cfg
        return object()

    monkeypatch.setattr(hydra.utils, "instantiate", _fake_instantiate)

    _ = build_env(cfg)

    env_cfg = captured["env_cfg"]
    assert env_cfg["judge"]["_target_"] == "synthstats.judges.llm_critic.LLMCriticJudge"
    assert env_cfg["judge"]["model_name"] == "openai/gpt-5-mini"
    # build_env should not mutate the base env config in-place
    assert cfg.env.judge._target_ == "synthstats.judges.likelihood.LikelihoodJudge"
