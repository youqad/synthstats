from __future__ import annotations

import json
import math

import pytest
from omegaconf import OmegaConf


def test_warmstart_preserves_think_by_default(tmp_path) -> None:
    jsonl_file = tmp_path / "sft.jsonl"
    example = {
        "prompt": "P",
        "completion": "<think>reasoning</think>\n```python\nx = 1\n```",
    }
    jsonl_file.write_text(json.dumps(example) + "\n", encoding="utf-8")

    from synthstats.train.loop.replay import GFNReplayBuffer
    from synthstats.train.runners.local import LocalRunner

    runner = LocalRunner(OmegaConf.create({}))
    buffer = GFNReplayBuffer(capacity=10)

    class _LoopStub:
        def __init__(self, buf) -> None:
            self.gfn_replay_buffer = buf

    runner._warmstart_from_sft(_LoopStub(buffer), {"data_path": str(jsonl_file)})

    assert len(buffer) == 1
    entry = next(iter(buffer))
    assert entry.actions[0]["type"] == "submit_program"
    assert "<think>reasoning</think>" in entry.actions[0]["payload"]


def test_warmstart_compute_rewards_uses_env_score_program(tmp_path) -> None:
    jsonl_file = tmp_path / "sft.jsonl"
    examples = [
        {"prompt": "P1", "completion": "<think>a</think>\n```python\nx = 1\n```"},
        {"prompt": "P2", "completion": "<think>b</think>\n```python\ny = 2\n```"},
    ]
    jsonl_file.write_text(
        "\n".join(json.dumps(ex) for ex in examples) + "\n",
        encoding="utf-8",
    )

    from synthstats.train.loop.replay import GFNReplayBuffer
    from synthstats.train.runners.local import LocalRunner

    runner = LocalRunner(OmegaConf.create({}))
    buffer = GFNReplayBuffer(capacity=10)

    class _EnvStub:
        def score_program(self, program: str) -> float:
            if "x = 1" in program:
                return math.e  # log=1.0
            if "y = 2" in program:
                return math.e**2  # log=2.0
            return 1.0

    class _CollectorStub:
        def __init__(self, env) -> None:
            self.env = env

    class _LoopStub:
        def __init__(self, buf, env) -> None:
            self.gfn_replay_buffer = buf
            self.collector = _CollectorStub(env)

    runner._warmstart_from_sft(
        _LoopStub(buffer, _EnvStub()),
        {
            "data_path": str(jsonl_file),
            "compute_rewards": True,
            "dedupe": False,
        },
    )

    assert len(buffer) == 2
    entries = list(buffer)
    assert entries[0].log_reward == pytest.approx(1.0)
    assert entries[1].log_reward == pytest.approx(2.0)


def test_warmstart_log_clamp_accepts_omegaconf_listconfig(tmp_path) -> None:
    jsonl_file = tmp_path / "sft.jsonl"
    examples = [
        {"prompt": "P1", "completion": "<think>a</think>\n```python\nx = 1\n```"},
        {"prompt": "P2", "completion": "<think>b</think>\n```python\ny = 2\n```"},
    ]
    jsonl_file.write_text(
        "\n".join(json.dumps(ex) for ex in examples) + "\n",
        encoding="utf-8",
    )

    from synthstats.train.loop.replay import GFNReplayBuffer
    from synthstats.train.runners.local import LocalRunner

    runner = LocalRunner(OmegaConf.create({}))
    buffer = GFNReplayBuffer(capacity=10)

    class _EnvStub:
        def score_program(self, program: str) -> float:
            if "x = 1" in program:
                return math.e  # log=1.0
            if "y = 2" in program:
                return math.e**2  # log=2.0 (will be clamped)
            return 1.0

    class _CollectorStub:
        def __init__(self, env) -> None:
            self.env = env

    class _LoopStub:
        def __init__(self, buf, env) -> None:
            self.gfn_replay_buffer = buf
            self.collector = _CollectorStub(env)

    warmstart_cfg = OmegaConf.create(
        {
            "data_path": str(jsonl_file),
            "compute_rewards": True,
            "dedupe": False,
            "log_clamp": [-1.0, 1.0],
        }
    )

    runner._warmstart_from_sft(_LoopStub(buffer, _EnvStub()), warmstart_cfg)

    entries = list(buffer)
    assert entries[0].log_reward == pytest.approx(1.0)
    assert entries[1].log_reward == pytest.approx(1.0)  # clamped from 2.0 to 1.0


def test_warmstart_compute_rewards_requires_env_score_program(tmp_path) -> None:
    jsonl_file = tmp_path / "sft.jsonl"
    example = {
        "prompt": "P",
        "completion": "<think>reasoning</think>\n```python\nx = 1\n```",
    }
    jsonl_file.write_text(json.dumps(example) + "\n", encoding="utf-8")

    from synthstats.train.loop.replay import GFNReplayBuffer
    from synthstats.train.runners.local import LocalRunner

    runner = LocalRunner(OmegaConf.create({}))
    buffer = GFNReplayBuffer(capacity=10)

    class _EnvWithoutScoring:
        pass

    class _CollectorStub:
        def __init__(self, env) -> None:
            self.env = env

    class _LoopStub:
        def __init__(self, buf, env) -> None:
            self.gfn_replay_buffer = buf
            self.collector = _CollectorStub(env)

    with pytest.raises(ValueError, match="score_program"):
        runner._warmstart_from_sft(
            _LoopStub(buffer, _EnvWithoutScoring()),
            {
                "data_path": str(jsonl_file),
                "compute_rewards": True,
            },
        )
