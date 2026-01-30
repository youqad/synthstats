"""Tests for GFlowNetTrainer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from synthstats.distributed.driver_replay_buffer import DriverGFNReplayBuffer
from synthstats.distributed.gfn_trainer import (
    GFlowNetTrainer,
    GFNBatch,
    GFNConfig,
)


class TestGFNConfig:
    """Tests for GFNConfig dataclass."""

    def test_default_values(self) -> None:
        """Defaults are sensible."""
        config = GFNConfig()

        assert config.loss_type == "modified_subtb"
        assert config.subtb_lambda == 0.9
        assert config.replay_buffer_size == 10000
        assert config.replay_ratio == 0.5
        assert config.temperature == 1.0

    def test_custom_values(self) -> None:
        """Custom values override defaults."""
        config = GFNConfig(
            loss_type="tb",
            subtb_lambda=0.8,
            replay_buffer_size=5000,
            temperature=0.7,
        )

        assert config.loss_type == "tb"
        assert config.subtb_lambda == 0.8
        assert config.replay_buffer_size == 5000
        assert config.temperature == 0.7


class TestGFNBatch:
    """Tests for GFNBatch dataclass."""

    def test_creation(self) -> None:
        """Batch holds expected tensors."""
        B, L = 4, 10
        batch = GFNBatch(
            input_ids=torch.zeros(B, L, dtype=torch.long),
            attention_mask=torch.ones(B, L),
            response_mask=torch.ones(B, L - 1),
            prompt_lengths=torch.tensor([2, 2, 3, 3]),
            log_rewards=torch.tensor([-0.5, -0.3, -0.4, -0.2]),
            terminated=torch.ones(B, dtype=torch.bool),
            temperature=torch.full((B,), 0.7),
        )

        assert batch.input_ids.shape == (B, L)
        assert batch.log_rewards.shape == (B,)

    def test_to_device(self) -> None:
        """to() moves all tensors."""
        B, L = 2, 5
        batch = GFNBatch(
            input_ids=torch.zeros(B, L, dtype=torch.long),
            attention_mask=torch.ones(B, L),
            response_mask=torch.ones(B, L - 1),
            prompt_lengths=torch.tensor([2, 2]),
            log_rewards=torch.tensor([-0.5, -0.3]),
            terminated=torch.ones(B, dtype=torch.bool),
            temperature=torch.full((B,), 0.7),
        )

        # move to CPU (should work on any system)
        moved = batch.to("cpu")

        assert moved.input_ids.device == torch.device("cpu")
        assert moved.log_rewards.device == torch.device("cpu")


class TestGFlowNetTrainerStandalone:
    """Tests for GFlowNetTrainer in standalone mode (without SkyRL)."""

    @pytest.fixture
    def mock_cfg(self):
        """Mock Hydra config."""
        cfg = MagicMock()
        cfg.gfn = {
            "loss_type": "modified_subtb",
            "subtb_lambda": 0.9,
            "replay_buffer_size": 100,
            "replay_ratio": 0.5,
            "min_buffer_before_replay": 10,
            "temperature": 0.7,
        }
        cfg.trainer = MagicMock()
        cfg.trainer.lr = 1e-4
        cfg.trainer.policy = MagicMock()
        cfg.trainer.policy.model = MagicMock()
        cfg.trainer.policy.model.path = "mock-model"
        return cfg

    def test_initialization(self, mock_cfg) -> None:
        """Trainer initializes with expected components."""
        # patch SkyRL to simulate standalone mode
        with patch.dict("sys.modules", {"skyrl_train.trainer": None}):
            trainer = GFlowNetTrainer(mock_cfg)

            assert trainer.gfn_config.loss_type == "modified_subtb"
            assert isinstance(trainer.logZ, torch.nn.Parameter)
            assert isinstance(trainer.replay_buffer, DriverGFNReplayBuffer)

    def test_logZ_parameter(self, mock_cfg) -> None:
        """logZ is learnable."""
        trainer = GFlowNetTrainer(mock_cfg)

        assert trainer.logZ.requires_grad
        assert trainer.logZ.item() == 0.0  # default init

    def test_replay_buffer_created(self, mock_cfg) -> None:
        """Replay buffer uses config capacity."""
        trainer = GFlowNetTrainer(mock_cfg)

        assert len(trainer.replay_buffer) == 0
        assert trainer.replay_buffer._capacity == 100

    def test_compute_advantages_logs_raw_rewards(self, mock_cfg) -> None:
        """Raw rewards are log-transformed before TB/SubTB loss."""
        trainer = GFlowNetTrainer(mock_cfg)
        trainer.gfn_config.replay_ratio = 0.0  # keep batch fresh-only

        B, L = 2, 5
        rewards = torch.tensor([1.0, 0.5])
        batch = {
            "input_ids": torch.randint(0, 100, (B, L)),
            "attention_mask": torch.ones(B, L),
            "response_mask": torch.ones(B, L - 1),
            "prompt_lengths": torch.tensor([1, 1]),
            "rewards": rewards,
            "terminated": torch.ones(B, dtype=torch.bool),
            "temperature": torch.full((B,), 0.7),
        }

        combined = trainer.compute_advantages_and_returns(batch)
        expected = torch.log(rewards.clamp(min=trainer.gfn_config.reward_floor))
        assert torch.allclose(combined["log_rewards"], expected)

    def test_get_optimizer_param_groups(self, mock_cfg) -> None:
        """logZ gets its own LR."""
        mock_cfg.gfn["lr_logZ"] = 0.01
        trainer = GFlowNetTrainer(mock_cfg)

        groups = trainer.get_optimizer_param_groups()

        # should have at least logZ group
        assert len(groups) >= 1

        # find logZ group
        logZ_group = None
        for g in groups:
            if trainer.logZ in g["params"]:
                logZ_group = g
                break

        assert logZ_group is not None
        assert logZ_group["lr"] == 0.01


class TestTrainCriticAndPolicy:
    """Tests for train_critic_and_policy method."""

    @pytest.fixture
    def trainer_with_optimizer(self):
        """Trainer with Adam optimizer."""
        cfg = MagicMock()
        cfg.gfn = {
            "loss_type": "tb",  # use simpler TB for testing
            "subtb_lambda": 0.9,
            "replay_buffer_size": 100,
            "replay_ratio": 0.0,  # no replay for this test
            "min_buffer_before_replay": 10,
            "entropy_coef": 0.0,  # disable entropy for predictable loss
            "temperature": 1.0,
        }
        cfg.trainer = MagicMock()
        cfg.trainer.lr = 1e-4

        trainer = GFlowNetTrainer(cfg)
        trainer.optimizer = torch.optim.Adam([trainer.logZ], lr=0.01)

        return trainer

    def test_loss_computation(self, trainer_with_optimizer) -> None:
        """Returns loss metrics."""
        trainer = trainer_with_optimizer
        B, T = 4, 10

        batch = {
            "log_probs": torch.randn(B, T),
            "eos_logprobs": None,  # TB doesn't need EOS
            "log_rewards": torch.randn(B),
            "response_mask": torch.ones(B, T),
            "logZ": trainer.logZ,
        }

        metrics = trainer.train_critic_and_policy(batch)

        assert "loss" in metrics
        assert "tb_loss" in metrics
        assert "logZ" in metrics
        assert isinstance(metrics["loss"], float)

    def test_logZ_updates(self, trainer_with_optimizer) -> None:
        """logZ moves toward equilibrium."""
        trainer = trainer_with_optimizer
        B, T = 4, 10

        initial_logZ = trainer.logZ.item()

        # create batch with non-zero loss (log_probs sum != log_reward - logZ)
        # use fixed tensors to ensure consistent gradient direction
        batch = {
            "log_probs": torch.full((B, T), -0.5, requires_grad=True),
            "eos_logprobs": None,
            "log_rewards": torch.full((B,), 5.0),  # high reward -> logZ should increase
            "response_mask": torch.ones(B, T),
            "logZ": trainer.logZ,
        }

        for _ in range(10):
            trainer.train_critic_and_policy(batch)

        final_logZ = trainer.logZ.item()

        # logZ must actually change
        assert final_logZ != initial_logZ, (
            f"logZ should update during training but stayed at {initial_logZ}"
        )
        # with high log_reward (5.0) and negative log_probs, logZ should increase
        # TB loss: (logZ + sum(log_probs) - log_reward)^2
        # sum(log_probs) = -0.5 * 10 = -5.0
        # residual = logZ + (-5.0) - 5.0 = logZ - 10
        # to minimize, logZ should move toward 10
        assert final_logZ > initial_logZ, (
            f"logZ should increase toward equilibrium: initial={initial_logZ}, final={final_logZ}"
        )

    def test_buffer_version_incremented(self, trainer_with_optimizer) -> None:
        """Policy version bumps after each step."""
        trainer = trainer_with_optimizer
        B, T = 4, 10

        initial_version = trainer.replay_buffer.policy_version

        batch = {
            "log_probs": torch.randn(B, T),
            "eos_logprobs": None,
            "log_rewards": torch.randn(B),
            "response_mask": torch.ones(B, T),
            "logZ": trainer.logZ,
        }

        trainer.train_critic_and_policy(batch)

        assert trainer.replay_buffer.policy_version == initial_version + 1

    def test_metrics_include_buffer_stats(self, trainer_with_optimizer) -> None:
        """Metrics include buffer stats."""
        trainer = trainer_with_optimizer
        B, T = 4, 10

        batch = {
            "log_probs": torch.randn(B, T),
            "eos_logprobs": None,
            "log_rewards": torch.randn(B),
            "response_mask": torch.ones(B, T),
            "logZ": trainer.logZ,
        }

        metrics = trainer.train_critic_and_policy(batch)

        assert "buffer_size" in metrics
        assert "buffer_mean_staleness" in metrics


class TestSubTBLoss:
    """Tests specifically for SubTB loss computation."""

    @pytest.fixture
    def subtb_trainer(self):
        """Trainer with SubTB loss."""
        cfg = MagicMock()
        cfg.gfn = {
            "loss_type": "modified_subtb",
            "subtb_lambda": 0.9,
            "tb_max_residual": 100.0,
            "replay_buffer_size": 100,
            "replay_ratio": 0.0,
            "min_buffer_before_replay": 10,
            "entropy_coef": 0.0,
            "temperature": 1.0,
        }
        cfg.trainer = MagicMock()
        cfg.trainer.lr = 1e-4

        trainer = GFlowNetTrainer(cfg)
        trainer.optimizer = torch.optim.Adam([trainer.logZ], lr=0.01)

        return trainer

    def test_subtb_with_eos_logprobs(self, subtb_trainer) -> None:
        """Uses EOS logprobs when provided."""
        trainer = subtb_trainer
        B, T = 4, 10

        # need requires_grad=True for log_probs since SubTB loss backprops through them
        batch = {
            "log_probs": torch.randn(B, T, requires_grad=True),
            "eos_logprobs": torch.randn(B, T),  # provide EOS logprobs
            "log_rewards": torch.randn(B),
            "response_mask": torch.ones(B, T),
            "logZ": trainer.logZ,
        }

        metrics = trainer.train_critic_and_policy(batch)

        assert "loss" in metrics
        assert metrics["loss"] >= 0  # squared loss should be non-negative

    def test_subtb_fallback_without_eos(self, subtb_trainer) -> None:
        """Falls back to TB without EOS logprobs."""
        trainer = subtb_trainer
        B, T = 4, 10

        batch = {
            "log_probs": torch.randn(B, T, requires_grad=True),
            "eos_logprobs": None,  # no EOS logprobs
            "log_rewards": torch.randn(B),
            "response_mask": torch.ones(B, T),
            "logZ": trainer.logZ,
        }

        # should not raise, should fall back to TB
        metrics = trainer.train_critic_and_policy(batch)
        assert "loss" in metrics


class TestGradientFlow:
    """Gradient flow through policy model (fixes detached tensor bug)."""

    @pytest.fixture
    def small_model(self):
        """Small LM for testing."""

        class SmallLM(torch.nn.Module):
            def __init__(self, vocab_size: int = 100, hidden_dim: int = 32):
                super().__init__()
                self.embed = torch.nn.Embedding(vocab_size, hidden_dim)
                self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim)
                self.linear2 = torch.nn.Linear(hidden_dim, vocab_size)

            def forward(self, input_ids, attention_mask=None):
                x = self.embed(input_ids)  # [B, L, H]
                x = torch.relu(self.linear1(x))  # [B, L, H]
                logits = self.linear2(x)  # [B, L, V]
                # return object with .logits attribute (like HF models)
                return type("Output", (), {"logits": logits})()

        return SmallLM()

    @pytest.fixture
    def trainer_with_model(self, small_model):
        """Trainer with a small model attached."""
        cfg = MagicMock()
        cfg.gfn = {
            "loss_type": "tb",  # simpler TB for testing
            "subtb_lambda": 0.9,
            "replay_buffer_size": 100,
            "replay_ratio": 0.0,
            "min_buffer_before_replay": 10,
            "entropy_coef": 0.0,
            "use_local_scoring_for_training": True,  # CRITICAL: enable fix
            "temperature": 1.0,
        }
        cfg.trainer = MagicMock()
        cfg.trainer.lr = 1e-4
        cfg.trainer.max_grad_norm = 1.0
        cfg.trainer.policy = MagicMock()
        cfg.trainer.policy.model = MagicMock()
        cfg.trainer.policy.model.path = "test-model"

        trainer = GFlowNetTrainer(cfg)
        trainer.policy_model = small_model

        # optimizer includes both model and logZ
        trainer.optimizer = torch.optim.Adam(
            [
                {"params": small_model.parameters(), "lr": 1e-3},
                {"params": [trainer.logZ], "lr": 0.01},
            ]
        )

        return trainer

    def test_policy_model_receives_gradients(self, trainer_with_model) -> None:
        """CRITICAL: Verify backward() creates gradients for policy_model params."""
        trainer = trainer_with_model
        B, L = 4, 16

        # create batch with all required fields for re-scoring
        input_ids = torch.randint(0, 100, (B, L))
        batch = {
            "input_ids": input_ids,
            "attention_mask": torch.ones(B, L),
            "response_mask": torch.ones(B, L - 1),
            "prompt_lengths": torch.tensor([2, 2, 2, 2]),
            "log_rewards": torch.full((B,), 5.0),
            "terminated": torch.ones(B, dtype=torch.bool),
            "temperature": torch.ones(B),
            # pre-computed log_probs will be replaced by re-scoring
            "log_probs": torch.randn(B, L - 1),
            "eos_logprobs": torch.randn(B, L - 1),
            "logZ": trainer.logZ,
        }

        # zero grads and run training step
        trainer.optimizer.zero_grad()
        trainer.train_critic_and_policy(batch)

        # check that policy model parameters have gradients
        has_grad = False
        for _name, param in trainer.policy_model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, (
            "Policy model parameters have NO gradients after backward()! "
            "The gradient flow fix (use_local_scoring_for_training) may not be working."
        )

    def test_policy_model_parameters_update(self, trainer_with_model) -> None:
        """CRITICAL: Verify optimizer.step() actually changes policy_model params."""
        trainer = trainer_with_model
        B, L = 4, 16

        # capture initial parameter values
        initial_params = {
            name: param.clone().detach() for name, param in trainer.policy_model.named_parameters()
        }

        input_ids = torch.randint(0, 100, (B, L))
        batch = {
            "input_ids": input_ids,
            "attention_mask": torch.ones(B, L),
            "response_mask": torch.ones(B, L - 1),
            "prompt_lengths": torch.tensor([2, 2, 2, 2]),
            "log_rewards": torch.full((B,), 5.0),
            "terminated": torch.ones(B, dtype=torch.bool),
            "temperature": torch.ones(B),
            "log_probs": torch.randn(B, L - 1),
            "eos_logprobs": torch.randn(B, L - 1),
            "logZ": trainer.logZ,
        }

        trainer.train_critic_and_policy(batch)

        # check that at least one parameter changed
        params_changed = False
        for name, param in trainer.policy_model.named_parameters():
            if not torch.allclose(param, initial_params[name]):
                params_changed = True
                break

        assert params_changed, (
            "Policy model parameters UNCHANGED after training step! "
            "Either gradients are not flowing or optimizer.step() is not being called."
        )

    def test_logZ_and_model_both_update(self, trainer_with_model) -> None:
        """Both logZ and policy_model should update in the same step."""
        trainer = trainer_with_model
        B, L = 4, 16

        initial_logZ = trainer.logZ.item()
        initial_embed_weight = trainer.policy_model.embed.weight.clone().detach()

        input_ids = torch.randint(0, 100, (B, L))
        batch = {
            "input_ids": input_ids,
            "attention_mask": torch.ones(B, L),
            "response_mask": torch.ones(B, L - 1),
            "prompt_lengths": torch.tensor([2, 2, 2, 2]),
            "log_rewards": torch.full((B,), 5.0),
            "terminated": torch.ones(B, dtype=torch.bool),
            "temperature": torch.ones(B),
            "log_probs": torch.randn(B, L - 1),
            "eos_logprobs": torch.randn(B, L - 1),
            "logZ": trainer.logZ,
        }

        for _ in range(5):
            trainer.train_critic_and_policy(batch)

        final_logZ = trainer.logZ.item()
        final_embed_weight = trainer.policy_model.embed.weight

        logZ_changed = final_logZ != initial_logZ
        embed_changed = not torch.allclose(final_embed_weight, initial_embed_weight)

        assert logZ_changed, f"logZ did not change: {initial_logZ} -> {final_logZ}"
        assert embed_changed, "Embedding weights did not change"

    def test_gradient_flow_disabled(self) -> None:
        """With use_local_scoring_for_training=False, model should NOT get gradients."""
        cfg = MagicMock()
        cfg.gfn = {
            "loss_type": "tb",
            "subtb_lambda": 0.9,
            "replay_buffer_size": 100,
            "replay_ratio": 0.0,
            "min_buffer_before_replay": 10,
            "entropy_coef": 0.0,
            "use_local_scoring_for_training": False,  # DISABLE the fix
            "temperature": 1.0,
        }
        cfg.trainer = MagicMock()
        cfg.trainer.lr = 1e-4

        trainer = GFlowNetTrainer(cfg)

        # no model attached, so it should use pre-computed log_probs
        B, T = 4, 10
        batch = {
            "log_probs": torch.randn(B, T, requires_grad=True),
            "eos_logprobs": None,
            "log_rewards": torch.randn(B),
            "response_mask": torch.ones(B, T),
            "logZ": trainer.logZ,
        }

        trainer.optimizer = torch.optim.Adam([trainer.logZ], lr=0.01)

        # this should still work, just with no model gradient flow
        metrics = trainer.train_critic_and_policy(batch)
        assert "loss" in metrics


class TestCheckpointing:
    """Tests for checkpoint save/load functionality."""

    def test_save_and_load_logz(self, tmp_path) -> None:
        """logZ should be saved and restored correctly."""
        cfg = MagicMock()
        cfg.gfn = {"logZ_init": 0.0}
        cfg.trainer = MagicMock()
        cfg.trainer.lr = 1e-4

        # create trainer and modify logZ
        trainer = GFlowNetTrainer(cfg)
        trainer.logZ.data.fill_(5.0)
        trainer._train_step_count = 100
        trainer.replay_buffer._policy_version = 50

        # save checkpoint
        trainer.save_checkpoint(tmp_path)

        # verify checkpoint file exists
        assert (tmp_path / "logZ.pt").exists()

        # create fresh trainer and load
        trainer2 = GFlowNetTrainer(cfg)
        assert trainer2.logZ.item() == 0.0  # initial value
        assert trainer2._train_step_count == 0

        trainer2.load_checkpoint(tmp_path)

        # verify restored values
        assert trainer2.logZ.item() == pytest.approx(5.0)
        assert trainer2._train_step_count == 100
        assert trainer2.replay_buffer.policy_version == 50

    def test_load_missing_checkpoint(self, tmp_path) -> None:
        """Loading from missing checkpoint should warn but not crash."""
        cfg = MagicMock()
        cfg.gfn = {"logZ_init": 1.0}
        cfg.trainer = MagicMock()
        cfg.trainer.lr = 1e-4

        trainer = GFlowNetTrainer(cfg)
        initial_logZ = trainer.logZ.item()

        # load from empty directory - should warn but not crash
        trainer.load_checkpoint(tmp_path)

        # logZ should be unchanged
        assert trainer.logZ.item() == initial_logZ
