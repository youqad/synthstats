import pytest  # noqa: F401 (provides tmp_path fixture)
import torch

from synthstats.integrations.tinker.adapter import TinkerConfig, TinkerTrainer


class TestRewardTemperatureAnnealing:
    def test_step_counter_starts_at_zero(self):
        config = TinkerConfig(model="test-model")
        trainer = TinkerTrainer(config)
        assert trainer.step == 0

    def test_default_temperature_is_one(self):
        config = TinkerConfig(model="test-model")
        trainer = TinkerTrainer(config)
        assert trainer.reward_temperature == 1.0

    def test_temperature_follows_linear_schedule(self):
        config = TinkerConfig(model="test-model")
        config.reward_schedule = {
            "start": 1.0,
            "end": 0.1,
            "horizon": 100,
            "mode": "linear",
        }
        trainer = TinkerTrainer(config)

        assert trainer.reward_temperature == 1.0

        trainer._step = 50
        assert abs(trainer.reward_temperature - 0.55) < 1e-6

        trainer._step = 100
        assert abs(trainer.reward_temperature - 0.1) < 1e-6

    def test_temperature_clamps_beyond_horizon(self):
        config = TinkerConfig(model="test-model")
        config.reward_schedule = {
            "start": 1.0,
            "end": 0.1,
            "horizon": 100,
        }
        trainer = TinkerTrainer(config)
        trainer._step = 200

        assert abs(trainer.reward_temperature - 0.1) < 1e-6

    def test_zero_temperature_rejected_at_init(self):
        config = TinkerConfig(model="test-model")
        config.reward_schedule = {
            "start": 1.0,
            "end": 0.0,  # Invalid: must be positive
            "horizon": 100,
        }

        with pytest.raises(ValueError, match="must be positive"):
            TinkerTrainer(config)

    def test_negative_temperature_rejected_at_init(self):
        config = TinkerConfig(model="test-model")
        config.reward_schedule = {
            "start": 1.0,
            "end": -0.5,  # Invalid: must be positive
            "horizon": 100,
        }

        with pytest.raises(ValueError, match="must be positive"):
            TinkerTrainer(config)

    def test_checkpoint_saves_step(self, tmp_path):
        config = TinkerConfig(model="test-model")
        trainer = TinkerTrainer(config)
        trainer._step = 42

        checkpoint_path = str(tmp_path / "checkpoint.pt")
        trainer.save_checkpoint(checkpoint_path)

        checkpoint = torch.load(checkpoint_path, weights_only=True)
        assert checkpoint["step"] == 42

    def test_checkpoint_restores_step(self, tmp_path):
        config = TinkerConfig(model="test-model")
        trainer = TinkerTrainer(config)
        trainer._step = 42

        checkpoint_path = str(tmp_path / "checkpoint.pt")
        trainer.save_checkpoint(checkpoint_path)

        new_trainer = TinkerTrainer(config)
        assert new_trainer.step == 0

        new_trainer.load_checkpoint(checkpoint_path)
        assert new_trainer.step == 42

    def test_old_checkpoint_without_step_field(self, tmp_path):
        config = TinkerConfig(model="test-model")
        trainer = TinkerTrainer(config)
        trainer._step = 100

        checkpoint_path = str(tmp_path / "old_checkpoint.pt")
        torch.save(
            {"logZ": torch.tensor(1.5), "config": {"model": "test-model"}},
            checkpoint_path,
        )

        trainer.load_checkpoint(checkpoint_path)

        assert trainer.step == 100  # unchanged
        assert abs(trainer.logZ.item() - 1.5) < 1e-6


class TestScheduleInitialization:
    def test_initializes_from_dict_config(self):
        config = TinkerConfig(model="test-model")
        config.reward_schedule = {
            "start": 2.0,
            "end": 0.5,
            "horizon": 500,
            "mode": "cosine",
        }
        trainer = TinkerTrainer(config)

        assert trainer._reward_schedule is not None
        assert trainer._reward_schedule.start == 2.0
        assert trainer._reward_schedule.end == 0.5
        assert trainer._reward_schedule.horizon == 500
        assert trainer._reward_schedule.mode == "cosine"

    def test_none_schedule_when_not_configured(self):
        config = TinkerConfig(model="test-model")
        trainer = TinkerTrainer(config)

        assert trainer._reward_schedule is None
        assert trainer.reward_temperature == 1.0
