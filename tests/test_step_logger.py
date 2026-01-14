"""Tests for StepLogger - WRITTEN FIRST per TDD."""

import pytest


class TestStepLoggerImport:
    """Verify StepLogger is importable."""

    def test_import_step_logger_class(self):
        """StepLogger should be importable."""
        from synthstats.logging.step_logger import StepLogger

        assert StepLogger is not None


class TestStepLoggerInit:
    """Test StepLogger initialization."""

    def test_step_logger_init_without_wandb(self):
        """StepLogger should initialize without wandb."""
        from synthstats.logging.step_logger import StepLogger

        logger = StepLogger()

        assert logger is not None
        assert logger.wandb_module is None

    def test_step_logger_init_with_mock_wandb(self):
        """StepLogger should accept wandb module."""
        from synthstats.logging.step_logger import StepLogger

        class MockWandB:
            pass

        logger = StepLogger(wandb_module=MockWandB())

        assert logger.wandb_module is not None


class TestStepLoggerMethods:
    """Test StepLogger logging methods."""

    def test_log_step_without_wandb(self):
        """log_step should not crash without wandb."""
        from synthstats.logging.step_logger import StepLogger

        logger = StepLogger()

        # should not raise
        logger.log_step(
            step_idx=1,
            loss=0.5,
            logZ=0.1,
            avg_reward=1.0,
        )

    def test_log_step_with_extra_metrics(self):
        """log_step should accept extra metrics."""
        from synthstats.logging.step_logger import StepLogger

        logger = StepLogger()

        # should not raise
        logger.log_step(
            step_idx=1,
            loss=0.5,
            logZ=0.1,
            avg_reward=1.0,
            extra_metric=42,
        )

    def test_log_evaluation(self):
        """log_evaluation should log eval metrics."""
        from synthstats.logging.step_logger import StepLogger

        logger = StepLogger()

        # should not raise
        logger.log_evaluation(
            step_idx=100,
            eval_reward=1.5,
            success_rate=0.8,
        )

    def test_log_episode(self):
        """log_episode should log per-episode data."""
        from synthstats.logging.step_logger import StepLogger

        logger = StepLogger()

        # should not raise
        logger.log_episode(
            episode_idx=0,
            reward=1.0,
            length=5,
            success=True,
        )


class TestStepLoggerWithMockWandB:
    """Test StepLogger with mock wandb."""

    def test_log_step_calls_wandb_log(self):
        """log_step should call wandb.log when available."""
        from synthstats.logging.step_logger import StepLogger

        class MockWandB:
            logged = []

            def log(self, data, step=None):
                self.logged.append((data, step))

        mock_wandb = MockWandB()
        logger = StepLogger(wandb_module=mock_wandb)

        logger.log_step(step_idx=5, loss=0.3, logZ=0.2, avg_reward=1.5)

        assert len(mock_wandb.logged) == 1
        data, step = mock_wandb.logged[0]
        assert step == 5
        assert data["train/loss"] == 0.3
        assert data["train/logZ"] == 0.2
        assert data["train/avg_reward"] == 1.5


class TestStepLoggerHistory:
    """Test StepLogger history tracking."""

    def test_step_logger_tracks_history(self):
        """StepLogger should track metrics history."""
        from synthstats.logging.step_logger import StepLogger

        logger = StepLogger()

        logger.log_step(step_idx=1, loss=0.5, logZ=0.1, avg_reward=1.0)
        logger.log_step(step_idx=2, loss=0.4, logZ=0.2, avg_reward=1.2)

        assert len(logger.history) == 2
        assert logger.history[0]["loss"] == 0.5
        assert logger.history[1]["loss"] == 0.4

    def test_step_logger_get_summary(self):
        """get_summary should return aggregate stats."""
        from synthstats.logging.step_logger import StepLogger

        logger = StepLogger()

        logger.log_step(step_idx=1, loss=0.5, logZ=0.1, avg_reward=1.0)
        logger.log_step(step_idx=2, loss=0.3, logZ=0.2, avg_reward=2.0)

        summary = logger.get_summary()

        assert "avg_loss" in summary
        assert "avg_reward" in summary
        assert summary["avg_loss"] == pytest.approx(0.4)
        assert summary["avg_reward"] == pytest.approx(1.5)
