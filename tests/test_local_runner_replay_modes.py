from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from synthstats.train.data.replay import GFNReplayBuffer, ReplayBuffer
from synthstats.train.runners.local import LocalRunner


def test_build_replay_buffer_defaults_to_gfn() -> None:
    runner = LocalRunner(
        OmegaConf.create(
            {
                "runner": {
                    "replay": {
                        "capacity": 16,
                        "prioritized": True,
                        "alpha": 0.7,
                    }
                }
            }
        )
    )

    buffer = runner._build_replay_buffer()
    assert isinstance(buffer, GFNReplayBuffer)


def test_build_replay_buffer_simple_mode() -> None:
    runner = LocalRunner(
        OmegaConf.create(
            {
                "runner": {
                    "replay": {
                        "capacity": 16,
                        "mode": "simple",
                        "prioritized": True,
                        "alpha": 0.7,
                    }
                }
            }
        )
    )

    buffer = runner._build_replay_buffer()
    assert isinstance(buffer, ReplayBuffer)


def test_build_replay_buffer_rejects_unknown_mode() -> None:
    runner = LocalRunner(
        OmegaConf.create(
            {
                "runner": {
                    "replay": {
                        "capacity": 16,
                        "mode": "unknown",
                    }
                }
            }
        )
    )

    with pytest.raises(ValueError, match="runner.replay.mode"):
        runner._build_replay_buffer()


def test_optimizer_includes_boundary_critic_params(monkeypatch: pytest.MonkeyPatch) -> None:
    import synthstats.train.runners.local as local_module

    class _Policy:
        def __init__(self) -> None:
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def parameters(self):
            return [self.weight]

        def __call__(self, obs: str, temperature: float | None = None):
            del obs, temperature
            return {"type": "answer", "payload": "ok"}, 0.0, 0.0

    class _Objective:
        def __init__(self) -> None:
            self.logZ = torch.nn.Parameter(torch.tensor(0.0))
            self.boundary_critic = torch.nn.Sequential(
                torch.nn.Linear(4, 8),
                torch.nn.ReLU(),
                torch.nn.Linear(8, 1),
            )

    policy = _Policy()
    objective = _Objective()

    runner = LocalRunner(
        OmegaConf.create(
            {
                "seed": 123,
                "device": "cpu",
                "runner": {
                    "train": {"steps": 0, "batch_size": 1},
                    "replay": {"capacity": 0},
                },
                "checkpoint": {"every_steps": 0},
                "logging": {},
                "learner": {"optim": {"policy_lr": 1e-5, "weight_decay": 0.0}},
            }
        )
    )

    monkeypatch.setattr(local_module, "build_env", lambda cfg: object())
    monkeypatch.setattr(LocalRunner, "_build_policy", lambda self, device: policy)
    monkeypatch.setattr(LocalRunner, "_build_objective", lambda self, device: objective)
    monkeypatch.setattr(LocalRunner, "_build_logger", lambda self: None)
    monkeypatch.setattr(LocalRunner, "_build_checkpoint_manager", lambda self: None)

    real_adamw = torch.optim.AdamW
    captured: dict[str, object] = {}

    def _capture_adamw(param_groups, *args, **kwargs):
        captured["param_groups"] = param_groups
        return real_adamw(param_groups, *args, **kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", _capture_adamw)

    result = runner.run()
    assert result.error is None

    param_groups = captured["param_groups"]
    assert isinstance(param_groups, list)

    optimizer_param_ids = {id(p) for group in param_groups for p in group["params"]}
    boundary_param_ids = {id(p) for p in objective.boundary_critic.parameters()}

    assert boundary_param_ids.issubset(optimizer_param_ids)
