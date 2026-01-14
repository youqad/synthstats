"""Tests for the training CLI."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

from synthstats.cli.train import (
    build_codec,
    build_judge,
    build_policy,
    build_task,
    build_trainer_config,
    create_log_callback,
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    setup_wandb,
)


class TestGetDevice:
    def test_cpu_explicit(self):
        assert get_device("cpu") == "cpu"

    def test_cuda_explicit(self):
        assert get_device("cuda") == "cuda"

    def test_mps_explicit(self):
        assert get_device("mps") == "mps"

    def test_auto_returns_valid_device(self):
        device = get_device("auto")
        assert device in ["cpu", "cuda", "mps"]

    @patch("torch.cuda.is_available", return_value=True)
    def test_auto_prefers_cuda(self, mock_cuda):
        assert get_device("auto") == "cuda"

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_auto_falls_back_to_mps(self, mock_mps, mock_cuda):
        assert get_device("auto") == "mps"

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_auto_falls_back_to_cpu(self, mock_mps, mock_cuda):
        assert get_device("auto") == "cpu"


class TestSetSeed:
    def test_set_seed_is_deterministic(self):
        set_seed(42)
        a = torch.rand(3)
        set_seed(42)
        b = torch.rand(3)
        assert torch.allclose(a, b)

    def test_different_seeds_give_different_results(self):
        set_seed(42)
        a = torch.rand(3)
        set_seed(123)
        b = torch.rand(3)
        assert not torch.allclose(a, b)


class TestBuildPolicy:
    def test_mock_policy(self):
        cfg = OmegaConf.create(
            {
                "model": {
                    "mock": True,
                    "fixed_text": '{"answer": "test"}',
                    "fixed_token_ids": [1, 2, 3],
                    "fixed_token_logprobs": [-0.5, -0.3, -0.2],
                }
            }
        )
        policy = build_policy(cfg, "cpu")
        from synthstats.policies.hf_policy import MockPolicy

        assert isinstance(policy, MockPolicy)

    def test_mock_policy_defaults(self):
        cfg = OmegaConf.create({"model": {"mock": True}})
        policy = build_policy(cfg, "cpu")
        from synthstats.policies.hf_policy import MockPolicy

        assert isinstance(policy, MockPolicy)


class TestBuildTask:
    def test_toy_task(self):
        cfg = OmegaConf.create({"task": {"name": "toy"}})
        task = build_task(cfg)
        from synthstats.training.trainer import ToyTask

        assert isinstance(task, ToyTask)

    def test_boxing_task(self):
        cfg = OmegaConf.create(
            {"task": {"name": "boxing", "env": "dugongs", "max_steps": 15}}
        )
        task = build_task(cfg)
        from synthstats.tasks.boxing.task import BoxingTask

        assert isinstance(task, BoxingTask)
        assert task.max_steps == 15

    def test_unknown_task_raises(self):
        cfg = OmegaConf.create({"task": {"name": "unknown"}})
        with pytest.raises(ValueError, match="Unknown task"):
            build_task(cfg)


class TestBuildCodec:
    def test_json_codec(self):
        cfg = OmegaConf.create({"runtime": {"codec": "json"}})
        codec = build_codec(cfg)
        from synthstats.runtime.codecs import JSONToolCodec

        assert isinstance(codec, JSONToolCodec)

    def test_xml_codec(self):
        cfg = OmegaConf.create({"runtime": {"codec": "xml"}})
        codec = build_codec(cfg)
        from synthstats.runtime.codecs import XMLToolCodec

        assert isinstance(codec, XMLToolCodec)

    def test_unknown_codec_raises(self):
        cfg = OmegaConf.create({"runtime": {"codec": "unknown"}})
        with pytest.raises(ValueError, match="Unknown codec"):
            build_codec(cfg)


class TestBuildJudge:
    def test_composite_judge(self):
        cfg = OmegaConf.create(
            {
                "judge": {
                    "judges": [
                        {"type": "likelihood", "weight": 0.7},
                        {"type": "formatting", "weight": 0.3},
                    ]
                }
            }
        )
        judge = build_judge(cfg)
        from synthstats.judges.composite import CompositeJudge

        assert isinstance(judge, CompositeJudge)
        assert len(judge.judges) == 2

    def test_unknown_judge_type_raises(self):
        cfg = OmegaConf.create(
            {"judge": {"judges": [{"type": "unknown", "weight": 1.0}]}}
        )
        with pytest.raises(ValueError, match="Unknown judge type"):
            build_judge(cfg)


class TestBuildTrainerConfig:
    def test_basic_config(self):
        cfg = OmegaConf.create(
            {
                "trainer": {
                    "batch_size": 8,
                    "learning_rate": 1e-5,
                    "logZ_lr": 0.1,
                    "num_episodes": 500,
                    "max_grad_norm": 0.5,
                },
                "task": {"max_steps": 15},
                "seed": 42,
            }
        )
        trainer_config = build_trainer_config(cfg)

        assert trainer_config.batch_size == 8
        assert trainer_config.learning_rate == 1e-5
        assert trainer_config.logZ_learning_rate == 0.1
        assert trainer_config.max_episodes == 500
        assert trainer_config.max_steps_per_episode == 15
        assert trainer_config.max_grad_norm == 0.5
        assert trainer_config.seed == 42

    def test_defaults(self):
        cfg = OmegaConf.create({"trainer": {}, "task": {}, "seed": 42})
        trainer_config = build_trainer_config(cfg)

        assert trainer_config.batch_size == 4
        assert trainer_config.max_steps_per_episode == 10


class TestSetupWandB:
    def test_wandb_disabled(self):
        cfg = OmegaConf.create({"wandb": {"enabled": False}})
        assert setup_wandb(cfg) is False

    def test_wandb_missing(self):
        cfg = OmegaConf.create({})
        assert setup_wandb(cfg) is False

    def test_wandb_enabled(self):
        # skip if wandb not installed
        pytest.importorskip("wandb")

        with patch("wandb.init") as mock_init:
            mock_run = MagicMock()
            mock_run.name = "test-run"
            mock_init.return_value = mock_run
            cfg = OmegaConf.create(
                {"wandb": {"enabled": True, "project": "test", "entity": None}}
            )
            result = setup_wandb(cfg)
            assert result is True
            mock_init.assert_called_once()


class TestLogCallback:
    def test_callback_logs_without_wandb(self, caplog):
        import logging

        caplog.set_level(logging.INFO)
        callback = create_log_callback(use_wandb=False)

        from synthstats.training.trainer import TrainMetrics

        metrics = TrainMetrics(
            loss=0.5, logZ=1.2, avg_reward=0.8, num_episodes=4, replay_fraction=0.25
        )
        callback(metrics, step=10)
        # just verify no exception


class TestCheckpointing:
    @pytest.fixture
    def mock_trainer(self):
        trainer = MagicMock()
        trainer.logZ = 1.5
        trainer._logZ = torch.nn.Parameter(torch.tensor(1.5))
        trainer.optimizer = torch.optim.Adam([trainer._logZ], lr=0.01)
        return trainer

    def test_save_checkpoint(self, mock_trainer, tmp_path):
        ckpt_path = save_checkpoint(mock_trainer, tmp_path, step=100)
        assert ckpt_path.exists()
        assert "checkpoint_000100.pt" in str(ckpt_path)

    def test_save_final_checkpoint(self, mock_trainer, tmp_path):
        ckpt_path = save_checkpoint(mock_trainer, tmp_path, step=100, is_final=True)
        assert "checkpoint_final.pt" in str(ckpt_path)

    def test_load_checkpoint(self, mock_trainer, tmp_path):
        # save first
        save_checkpoint(mock_trainer, tmp_path, step=50)

        # modify trainer state
        mock_trainer._logZ.data.fill_(999.0)

        # load
        ckpt_path = tmp_path / "checkpoint_000050.pt"
        step = load_checkpoint(mock_trainer, ckpt_path)

        assert step == 50
        assert mock_trainer._logZ.item() == pytest.approx(1.5)


class TestHydraConfigIntegration:
    """Integration tests using actual Hydra config files."""

    @pytest.fixture
    def config_dir(self):
        return Path(__file__).parent.parent / "configs"

    def test_config_loads(self, config_dir):
        """Verify Hydra config loads correctly."""
        if not config_dir.exists():
            pytest.skip("Config directory not found")

        from hydra import compose, initialize_config_dir

        with initialize_config_dir(
            config_dir=str(config_dir.absolute()), version_base=None
        ):
            cfg = compose(config_name="config")
            assert "model" in cfg
            assert "trainer" in cfg
            assert "task" in cfg
            assert "runtime" in cfg

    def test_model_override(self, config_dir):
        """Test model config override."""
        if not config_dir.exists():
            pytest.skip("Config directory not found")

        from hydra import compose, initialize_config_dir

        with initialize_config_dir(
            config_dir=str(config_dir.absolute()), version_base=None
        ):
            cfg = compose(config_name="config", overrides=["model=qwen3_4b"])
            assert "Qwen3-4B" in cfg.model.name


class TestTrainCLIIntegration:
    """Full integration tests for the training CLI."""

    @pytest.fixture
    def mock_config(self):
        return OmegaConf.create(
            {
                "model": {
                    "mock": True,
                    "fixed_text": '{"answer": "test"}',
                    "fixed_token_ids": [1, 2, 3],
                    "fixed_token_logprobs": [-0.5, -0.3, -0.2],
                },
                "trainer": {
                    "batch_size": 2,
                    "learning_rate": 1e-4,
                    "logZ_lr": 0.1,
                    "num_episodes": 3,
                    "max_grad_norm": 1.0,
                },
                "task": {"name": "toy", "max_steps": 5},
                "runtime": {"codec": "json"},
                "judge": {"judges": [{"type": "likelihood", "weight": 1.0}]},
                "seed": 42,
                "device": "cpu",
                "output_dir": "/tmp/test_output",
                "wandb": {"enabled": False},
            }
        )

    def test_mock_policy_training(self, mock_config, tmp_path):
        """Run training with MockPolicy (no real model)."""
        mock_config.output_dir = str(tmp_path)

        # build all components
        device = get_device(mock_config.device)
        set_seed(mock_config.seed)

        policy = build_policy(mock_config, device)
        task = build_task(mock_config)
        codec = build_codec(mock_config)
        judge = build_judge(mock_config)
        trainer_config = build_trainer_config(mock_config)

        from synthstats.training.trainer import Trainer

        trainer = Trainer(
            config=trainer_config,
            policy=policy,
            task=task,
            codec=codec,
            judge=judge,
            device=device,
        )

        # run a few steps
        metrics_list = trainer.train(num_steps=2)
        assert len(metrics_list) == 2
        assert all(m.loss >= 0 for m in metrics_list)

    def test_wandb_disabled_by_default(self, mock_config):
        """WandB should not run unless explicitly enabled."""
        assert mock_config.wandb.enabled is False
        result = setup_wandb(mock_config)
        assert result is False

    def test_checkpoint_roundtrip(self, mock_config, tmp_path):
        """Test saving and loading checkpoints."""
        mock_config.output_dir = str(tmp_path)

        device = get_device(mock_config.device)
        set_seed(mock_config.seed)

        policy = build_policy(mock_config, device)
        task = build_task(mock_config)
        codec = build_codec(mock_config)
        judge = build_judge(mock_config)
        trainer_config = build_trainer_config(mock_config)

        from synthstats.training.trainer import Trainer

        trainer = Trainer(
            config=trainer_config,
            policy=policy,
            task=task,
            codec=codec,
            judge=judge,
            device=device,
        )

        # train and save
        trainer.train_step()
        original_logZ = trainer.logZ
        ckpt_path = save_checkpoint(trainer, tmp_path, step=1)

        # create new trainer and load
        from synthstats.training.trainer import ToyTask

        trainer2 = Trainer(
            config=trainer_config,
            policy=policy,
            task=ToyTask(),
            codec=codec,
            judge=judge,
            device=device,
        )

        loaded_step = load_checkpoint(trainer2, ckpt_path)
        assert loaded_step == 1
        assert trainer2.logZ == pytest.approx(original_logZ)
