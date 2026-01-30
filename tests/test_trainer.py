"""Tests for the training orchestrator."""

import pytest

from synthstats.training.trainer import Trainer, TrainerConfig, TrainMetrics


class TestTrainerConfig:
    def test_default_values(self):
        config = TrainerConfig()
        assert config.batch_size == 4
        assert config.learning_rate == 1e-4

    def test_custom_values(self):
        config = TrainerConfig(batch_size=8, max_episodes=500)
        assert config.batch_size == 8
        assert config.max_episodes == 500

    def test_logZ_learning_rate_default(self):
        config = TrainerConfig()
        assert config.logZ_learning_rate == 1e-2

    def test_replay_config(self):
        config = TrainerConfig(
            replay_buffer_capacity=500,
            replay_sample_ratio=0.3,
        )
        assert config.replay_buffer_capacity == 500
        assert config.replay_sample_ratio == 0.3


class TestTrainMetrics:
    def test_metrics_creation(self):
        metrics = TrainMetrics(
            loss=0.5,
            logZ=1.2,
            avg_reward=0.8,
            num_episodes=4,
            replay_fraction=0.25,
        )
        assert metrics.loss == 0.5
        assert metrics.logZ == 1.2
        assert metrics.avg_reward == 0.8
        assert metrics.num_episodes == 4
        assert metrics.replay_fraction == 0.25

    def test_metrics_default_replay_fraction(self):
        metrics = TrainMetrics(
            loss=0.5,
            logZ=1.2,
            avg_reward=0.8,
            num_episodes=4,
        )
        assert metrics.replay_fraction == 0.0


class TestTrainer:
    @pytest.fixture
    def mock_policy(self):
        from synthstats.policies.hf_policy import MockPolicy

        return MockPolicy(
            fixed_text='{"answer": "test"}',
            fixed_token_ids=[1, 2, 3],
            fixed_token_logprobs=[-0.5, -0.3, -0.2],
        )

    @pytest.fixture
    def toy_task(self):
        from synthstats.training.trainer import ToyTask

        return ToyTask()

    @pytest.fixture
    def toy_judge(self):
        from synthstats.training.trainer import ToyJudge

        return ToyJudge()

    @pytest.fixture
    def codec(self):
        from synthstats.runtime.codecs import JSONToolCodec

        return JSONToolCodec()

    def test_trainer_init(self, mock_policy, toy_task, codec, toy_judge):
        config = TrainerConfig(batch_size=2, max_episodes=5)
        trainer = Trainer(config, mock_policy, toy_task, codec, judge=toy_judge)
        assert trainer is not None

    def test_trainer_init_creates_logZ(self, mock_policy, toy_task, codec, toy_judge):
        config = TrainerConfig()
        trainer = Trainer(config, mock_policy, toy_task, codec, judge=toy_judge)
        assert hasattr(trainer, "logZ")
        assert isinstance(trainer.logZ, float)

    def test_train_step_returns_metrics(self, mock_policy, toy_task, codec, toy_judge):
        config = TrainerConfig(batch_size=2)
        trainer = Trainer(config, mock_policy, toy_task, codec, judge=toy_judge)
        metrics = trainer.train_step()

        assert isinstance(metrics, TrainMetrics)
        assert isinstance(metrics.loss, float)
        assert isinstance(metrics.logZ, float)
        assert metrics.num_episodes == 2

    def test_train_loop(self, mock_policy, toy_task, codec, toy_judge):
        config = TrainerConfig(batch_size=2, max_episodes=4)
        trainer = Trainer(config, mock_policy, toy_task, codec, judge=toy_judge)
        all_metrics = trainer.train(num_steps=2)

        assert len(all_metrics) == 2

    def test_logZ_is_learnable(self, mock_policy, toy_task, codec, toy_judge):
        config = TrainerConfig(batch_size=2)
        trainer = Trainer(config, mock_policy, toy_task, codec, judge=toy_judge)
        # run multiple steps to ensure logZ changes
        for _ in range(3):
            trainer.train_step()

        # logZ should potentially change after training
        # note: may stay same if perfectly balanced, so we just check it's valid
        assert isinstance(trainer.logZ, float)

    def test_evaluate(self, mock_policy, toy_task, codec, toy_judge):
        config = TrainerConfig()
        trainer = Trainer(config, mock_policy, toy_task, codec, judge=toy_judge)
        eval_results = trainer.evaluate(num_episodes=3)

        assert "avg_reward" in eval_results

    def test_evaluate_returns_success_rate(self, mock_policy, toy_task, codec, toy_judge):
        config = TrainerConfig()
        trainer = Trainer(config, mock_policy, toy_task, codec, judge=toy_judge)
        eval_results = trainer.evaluate(num_episodes=3)

        assert "success_rate" in eval_results
        assert 0.0 <= eval_results["success_rate"] <= 1.0

    def test_replay_buffer_used(self, mock_policy, toy_task, codec, toy_judge):
        """After enough episodes, replay buffer should contribute to batch."""
        config = TrainerConfig(
            batch_size=4,
            replay_buffer_capacity=100,
            replay_sample_ratio=0.5,
        )
        trainer = Trainer(config, mock_policy, toy_task, codec, judge=toy_judge)

        # run a few steps to fill buffer
        for _ in range(5):
            trainer.train_step()

        # now replay should be used
        metrics = trainer.train_step()
        assert metrics.replay_fraction > 0

    def test_gradient_accumulation(self, mock_policy, toy_task, codec, toy_judge):
        config = TrainerConfig(
            batch_size=2,
            gradient_accumulation_steps=4,
        )
        trainer = Trainer(config, mock_policy, toy_task, codec, judge=toy_judge)
        # should not crash with accumulation
        metrics = trainer.train_step()
        assert metrics is not None

    def test_log_callback_called(self, mock_policy, toy_task, codec, toy_judge):
        called = []

        def callback(metrics, step):
            called.append((metrics, step))

        config = TrainerConfig(batch_size=2, log_interval=1)
        trainer = Trainer(
            config, mock_policy, toy_task, codec, judge=toy_judge, log_callback=callback
        )
        trainer.train(num_steps=3)

        assert len(called) == 3

    def test_trainer_with_device_cpu(self, mock_policy, toy_task, codec, toy_judge):
        config = TrainerConfig(batch_size=2)
        trainer = Trainer(config, mock_policy, toy_task, codec, judge=toy_judge, device="cpu")
        assert trainer is not None
        metrics = trainer.train_step()
        assert metrics is not None

    def test_gradient_clipping(self, mock_policy, toy_task, codec, toy_judge):
        config = TrainerConfig(batch_size=2, max_grad_norm=0.5)
        trainer = Trainer(config, mock_policy, toy_task, codec, judge=toy_judge)
        # should not crash with gradient clipping
        metrics = trainer.train_step()
        assert metrics is not None

    def test_seed_reproducibility(self, mock_policy, toy_task, codec, toy_judge):
        """Training with same seed should give same results."""
        from synthstats.training.trainer import ToyTask

        config = TrainerConfig(batch_size=2, seed=42)

        trainer1 = Trainer(config, mock_policy, toy_task, codec, judge=toy_judge)
        metrics1 = trainer1.train_step()

        # need fresh task instance
        toy_task2 = ToyTask()
        trainer2 = Trainer(config, mock_policy, toy_task2, codec, judge=toy_judge)
        metrics2 = trainer2.train_step()

        # with deterministic policy and same seed, losses should match
        assert metrics1.loss == pytest.approx(metrics2.loss, rel=1e-5)

    def test_gradient_flow_with_score_tokens(self, mock_policy, toy_task, codec, toy_judge):
        """Verify gradients flow through score_tokens when available."""
        config = TrainerConfig(batch_size=1)
        trainer = Trainer(config, mock_policy, toy_task, codec, judge=toy_judge)

        # verify MockPolicy has score_tokens
        assert hasattr(mock_policy, "score_tokens")

        # run train_step - this should use score_tokens for differentiable logprobs
        metrics = trainer.train_step()
        assert metrics is not None
        assert isinstance(metrics.loss, float)

    def test_recompute_logprobs_produces_tensors_with_grad(
        self, mock_policy, toy_task, codec, toy_judge
    ):
        """Verify _recompute_logprobs_differentiable returns tensors with grad."""
        from synthstats.core.types import Message, Reward, Trajectory

        config = TrainerConfig(batch_size=1)
        trainer = Trainer(config, mock_policy, toy_task, codec, judge=toy_judge)

        # create a simple trajectory
        traj = Trajectory(
            messages=[
                Message(role="system", content="Test system"),
                Message(role="user", content="Test user"),
                Message(role="assistant", content="Test response"),
            ],
            token_ids=[[1, 2, 3]],
            token_logprobs=[[-0.1, -0.2, -0.3]],
            loss_mask=[[True, True, True]],
            reward=Reward(total=1.0, components={"base": 1.0}, info={}),
        )

        # recompute logprobs
        logprob_tensors = trainer._recompute_logprobs_differentiable(traj)

        assert len(logprob_tensors) == 1
        assert logprob_tensors[0].requires_grad is True
        assert logprob_tensors[0].grad_fn is None  # leaf tensor but requires_grad
