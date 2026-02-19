"""Training CLI tests."""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from synthstats.cli.train import get_runner
from synthstats.train.runners.base import RunResult


class TestGetRunner:
    def test_local_runner(self):
        cfg = OmegaConf.create(
            {
                "runner": {"type": "local"},
                "seed": 42,
                "device": "cpu",
            }
        )
        runner = get_runner(cfg)
        from synthstats.train.runners.local import LocalRunner

        assert isinstance(runner, LocalRunner)

    def test_tinker_runner(self):
        cfg = OmegaConf.create(
            {
                "runner": {"type": "tinker"},
                "seed": 42,
                "device": "cpu",
            }
        )
        runner = get_runner(cfg)
        from synthstats.train.runners.tinker import TinkerRunner

        assert isinstance(runner, TinkerRunner)

    def test_default_is_local(self):
        cfg = OmegaConf.create(
            {
                "runner": {},
                "seed": 42,
                "device": "cpu",
            }
        )
        runner = get_runner(cfg)
        from synthstats.train.runners.local import LocalRunner

        assert isinstance(runner, LocalRunner)

    def test_unknown_runner_raises(self):
        cfg = OmegaConf.create(
            {
                "runner": {"type": "unknown"},
            }
        )
        with pytest.raises(ValueError, match="Unknown runner type"):
            get_runner(cfg)


class TestRunnerIntegration:
    def test_local_runner_initializes(self):
        cfg = OmegaConf.create(
            {
                "runner": {
                    "type": "local",
                    "train": {"steps": 1, "batch_size": 1},
                },
                "seed": 42,
                "device": "cpu",
                "env": {"name": "dugongs", "max_steps": 5},
                "policy": {"model_name": "mock"},
                "objective": {},
                "learner": {"optim": {}},
                "checkpoint": {"every_steps": 0},
                "logging": {},
            }
        )
        from synthstats.train.runners.local import LocalRunner

        runner = LocalRunner(cfg)
        assert runner.cfg == cfg


class TestTrainUtils:
    def test_resolve_device_cpu(self):
        from synthstats.train.utils.device import resolve_device

        assert resolve_device("cpu") == "cpu"

    def test_resolve_device_auto(self):
        from synthstats.train.utils.device import resolve_device

        device = resolve_device("auto")
        assert device in ["cpu", "cuda", "mps"]

    def test_seed_everything(self):
        from synthstats.train.utils.seeding import seed_everything

        seed_everything(42)
        a = torch.rand(3)
        seed_everything(42)
        b = torch.rand(3)
        assert torch.allclose(a, b)


class TestConfigFiles:
    @pytest.fixture
    def config_dir(self):
        return Path(__file__).parent.parent / "configs"

    def test_train_config_exists(self, config_dir):
        train_config = config_dir / "train.yaml"
        if not train_config.exists():
            pytest.skip("train.yaml not found")
        cfg = OmegaConf.load(train_config)
        assert "defaults" in cfg

    def test_runner_configs_exist(self, config_dir):
        runner_dir = config_dir / "runner"
        if not runner_dir.exists():
            pytest.skip("runner configs not found")
        assert (runner_dir / "local.yaml").exists()
        assert (runner_dir / "tinker.yaml").exists()

    def test_local_runner_config(self, config_dir):
        local_config = config_dir / "runner" / "local.yaml"
        if not local_config.exists():
            pytest.skip("local.yaml not found")
        cfg = OmegaConf.load(local_config)
        assert cfg.type == "local"
        assert "train" in cfg


class TestRunResult:
    def test_run_result_defaults(self):
        result = RunResult()
        assert result.metrics == {}
        assert result.checkpoints == []
        assert result.error is None
        assert result.interrupted is False

    def test_run_result_with_metrics(self):
        result = RunResult(
            metrics={"loss": 0.5, "logZ": 1.2},
            checkpoints=["/path/to/ckpt.pt"],
        )
        assert result.metrics["loss"] == 0.5
        assert len(result.checkpoints) == 1

    def test_run_result_with_error(self):
        result = RunResult(error="Training failed")
        assert result.error == "Training failed"


class TestHydraConfigIntegration:
    @pytest.fixture
    def config_dir(self):
        return Path(__file__).parent.parent / "configs"

    def test_old_config_loads(self, config_dir):
        if not config_dir.exists():
            pytest.skip("Config directory not found")

        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=str(config_dir.absolute()), version_base=None):
            cfg = compose(config_name="config")
            # old config structure
            assert "model" in cfg or "policy" in cfg
            assert "trainer" in cfg or "runner" in cfg

    def test_model_override(self, config_dir):
        if not (config_dir / "model").exists():
            pytest.skip("Model configs not found")

        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=str(config_dir.absolute()), version_base=None):
            cfg = compose(config_name="config", overrides=["model=qwen3_4b"])
            assert "Qwen3-4B" in cfg.model.name
