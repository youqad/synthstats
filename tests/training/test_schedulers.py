import pytest
import torch

from synthstats.train.utils.schedulers import (
    BehaviorMixConfig,
    ExplorationTemperatureConfig,
    LogZLearningRateConfig,
    RewardTemperatureSchedule,
    create_warmup_scheduler,
)


class TestRewardTemperatureSchedule:
    def test_linear_schedule(self):
        schedule = RewardTemperatureSchedule(start=1.0, end=0.1, horizon=1000)

        assert schedule.get(0) == 1.0
        assert abs(schedule.get(500) - 0.55) < 1e-6
        assert abs(schedule.get(1000) - 0.1) < 1e-6
        assert abs(schedule.get(2000) - 0.1) < 1e-6

    def test_cosine_schedule(self):
        schedule = RewardTemperatureSchedule(start=1.0, end=0.1, horizon=1000, mode="cosine")

        assert schedule.get(0) == 1.0
        mid = schedule.get(500)
        assert 0.4 < mid < 0.6
        assert abs(schedule.get(1000) - 0.1) < 1e-6

    def test_scale_reward(self):
        schedule = RewardTemperatureSchedule(start=2.0, end=0.5, horizon=100)

        log_r = 1.0
        assert schedule.scale_reward(log_r, 0) == 0.5
        assert schedule.scale_reward(log_r, 100) == 2.0

    def test_zero_horizon(self):
        schedule = RewardTemperatureSchedule(start=1.0, end=0.1, horizon=0)

        assert schedule.get(0) == 0.1
        assert schedule.get(100) == 0.1


class TestExplorationTemperatureConfig:
    def test_sample_temperature_on_policy(self):
        config = ExplorationTemperatureConfig(perturb_prob=0.0)

        for _ in range(10):
            assert config.sample_temperature() == 1.0

    def test_sample_temperature_perturbed(self):
        config = ExplorationTemperatureConfig(pf_temp_low=0.5, pf_temp_high=2.0, perturb_prob=1.0)

        for _ in range(10):
            temp = config.sample_temperature()
            assert 0.5 <= temp <= 2.0

    def test_sample_temperature_mixed(self):
        config = ExplorationTemperatureConfig()

        temps = [config.sample_temperature() for _ in range(100)]

        on_policy = [t for t in temps if t == 1.0]
        perturbed = [t for t in temps if t != 1.0]

        assert len(on_policy) > 20
        assert len(perturbed) > 20


class TestBehaviorMixConfig:
    def test_default_sums_to_one(self):
        config = BehaviorMixConfig()

        total = config.on_policy + config.replay + config.perturbed
        assert abs(total - 1.0) < 1e-6

    def test_invalid_sum_raises(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            BehaviorMixConfig(on_policy=0.5, replay=0.5, perturbed=0.5)

    def test_sample_source_distribution(self):
        config = BehaviorMixConfig(on_policy=0.25, replay=0.25, perturbed=0.50)

        sources = [config.sample_source() for _ in range(1000)]

        on_policy_count = sources.count("on_policy")
        replay_count = sources.count("replay")
        perturbed_count = sources.count("perturbed")

        assert 150 < on_policy_count < 350
        assert 150 < replay_count < 350
        assert 350 < perturbed_count < 650


class TestLogZLearningRateConfig:
    def test_multiplier(self):
        config = LogZLearningRateConfig(multiplier=100.0)

        assert config.get(1e-5) == 1e-3
        assert config.get(1e-4) == 1e-2

    def test_absolute_override(self):
        config = LogZLearningRateConfig(multiplier=100.0, absolute=0.01)

        assert config.get(1e-5) == 0.01
        assert config.get(1e-4) == 0.01


class TestCreateWarmupScheduler:
    def _make_optimizer(self) -> torch.optim.Optimizer:
        param = torch.nn.Parameter(torch.zeros(1))
        return torch.optim.SGD([param], lr=1.0)

    def test_warmup_phase(self):
        opt = self._make_optimizer()
        sched = create_warmup_scheduler(opt, warmup_steps=100, total_steps=1000)

        assert sched.get_last_lr()[0] == 0.0

        for _ in range(50):
            sched.step()
        assert abs(sched.get_last_lr()[0] - 0.5) < 1e-6

    def test_decay_phase(self):
        opt = self._make_optimizer()
        sched = create_warmup_scheduler(opt, warmup_steps=100, total_steps=200)

        for _ in range(100):
            sched.step()
        assert abs(sched.get_last_lr()[0] - 1.0) < 1e-6

        for _ in range(50):
            sched.step()
        assert abs(sched.get_last_lr()[0] - 0.5) < 1e-6

    def test_zero_warmup_steps(self):
        opt = self._make_optimizer()
        sched = create_warmup_scheduler(opt, warmup_steps=0, total_steps=100)

        assert sched.get_last_lr()[0] == 1.0

    def test_warmup_equals_total(self):
        opt = self._make_optimizer()
        sched = create_warmup_scheduler(opt, warmup_steps=100, total_steps=100)

        for _ in range(150):
            sched.step()

        assert sched.get_last_lr()[0] == 1.0
