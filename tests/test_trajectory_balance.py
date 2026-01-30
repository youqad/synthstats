"""Tests for Trajectory Balance loss and tb_identity advantage estimator.

Tests the SkyRL-integrated TB training components.
"""

import pytest
import torch

from synthstats.training.losses.trajectory_balance import (
    SKYRL_REGISTERED,
    compute_tb_identity_advantage,
    compute_trajectory_balance_loss,
)


class TestTBIdentityAdvantageEstimator:
    """Test the tb_identity advantage estimator."""

    def test_sums_rewards_per_trajectory(self):
        """Should sum token-level rewards to get trajectory reward."""
        # rewards only at final token (typical for GFlowNets)
        token_rewards = torch.tensor(
            [
                [0.0, 0.0, 0.0, 2.0],  # trajectory 1: reward = 2.0
                [0.0, 0.0, 1.5, 0.0],  # trajectory 2: reward = 1.5
            ]
        )
        mask = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 0.0],  # last token masked out
            ]
        )

        advantages, returns = compute_tb_identity_advantage(token_rewards, mask)

        # should broadcast trajectory reward back to sequence
        assert advantages.shape == token_rewards.shape
        assert returns.shape == token_rewards.shape

        # trajectory 1: 2.0 broadcast to all positions (masked)
        assert torch.allclose(advantages[0], torch.tensor([2.0, 2.0, 2.0, 2.0]))

        # trajectory 2: 1.5 broadcast (last position masked)
        assert torch.allclose(advantages[1], torch.tensor([1.5, 1.5, 1.5, 0.0]))

    def test_respects_mask(self):
        """Masked positions should have zero advantage."""
        token_rewards = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])  # last token masked

        advantages, _ = compute_tb_identity_advantage(token_rewards, mask)

        # sum is 1+2=3 (masked positions don't count)
        # broadcast to masked shape: [3, 3, 0]
        assert advantages[0, 0] == 3.0
        assert advantages[0, 1] == 3.0
        assert advantages[0, 2] == 0.0  # masked

    def test_returns_same_as_advantages(self):
        """For TB, returns and advantages should be identical."""
        token_rewards = torch.randn(4, 10)
        mask = torch.ones(4, 10)

        advantages, returns = compute_tb_identity_advantage(token_rewards, mask)

        assert torch.equal(advantages, returns)

    def test_no_gradients(self):
        """Output should not require gradients."""
        token_rewards = torch.randn(2, 5, requires_grad=True)
        mask = torch.ones(2, 5)

        advantages, returns = compute_tb_identity_advantage(token_rewards, mask)

        assert not advantages.requires_grad
        assert not returns.requires_grad

    def test_ignores_kwargs(self):
        """Should accept and ignore GAE-style kwargs."""
        token_rewards = torch.tensor([[1.0, 2.0]])
        mask = torch.ones(1, 2)

        # these are GAE params that tb_identity ignores
        advantages, returns = compute_tb_identity_advantage(
            token_rewards,
            mask,
            gamma=0.99,
            lambd=0.95,
            values=torch.randn(1, 2),
            index=[0],
        )

        assert advantages.shape == token_rewards.shape


class TestTrajectoryBalanceLoss:
    """Test the trajectory balance policy loss."""

    def test_computes_squared_residual(self):
        """TB loss should be (logZ + log_pi - log_R)^2."""
        log_probs = torch.tensor(
            [
                [-1.0, -1.0, -1.0],  # sum = -3
                [-0.5, -0.5, -0.5],  # sum = -1.5
            ]
        )
        advantages = torch.tensor(
            [  # actually log_rewards from tb_identity
                [2.0, 2.0, 2.0],  # log_R = 2.0 (broadcast)
                [1.0, 1.0, 1.0],  # log_R = 1.0 (broadcast)
            ]
        )
        loss_mask = torch.ones(2, 3)

        config = {"logZ": 0.5, "tb_max_residual": 100.0}

        loss, clip_ratio = compute_trajectory_balance_loss(
            log_probs, log_probs, advantages, config, loss_mask
        )

        # trajectory 1: logZ=0.5 + log_pi=-3 - log_R=2.0 = -4.5, sq = 20.25
        # trajectory 2: logZ=0.5 + log_pi=-1.5 - log_R=1.0 = -2.0, sq = 4.0
        # mean = (20.25 + 4.0) / 2 = 12.125
        expected_loss = 12.125
        assert torch.isclose(loss, torch.tensor(expected_loss), atol=0.01)

        # TB doesn't use clipping
        assert clip_ratio == 0.0

    def test_requires_logZ_in_config(self):
        """Should raise if logZ not in config."""
        log_probs = torch.randn(2, 5)
        advantages = torch.randn(2, 5)
        config = {"tb_max_residual": 100.0}  # missing logZ!

        with pytest.raises(RuntimeError, match="logZ not found"):
            compute_trajectory_balance_loss(log_probs, log_probs, advantages, config)

    def test_clamps_residual(self):
        """Should clamp residual to prevent extreme values."""
        log_probs = torch.tensor([[-1000.0]])  # extreme negative
        advantages = torch.tensor([[1000.0]])  # extreme positive
        config = {"logZ": 0.0, "tb_max_residual": 50.0}

        loss, _ = compute_trajectory_balance_loss(log_probs, log_probs, advantages, config)

        # residual would be 0 - 1000 - 1000 = -2000, clamped to -50
        # loss = 50^2 = 2500
        assert loss <= 2500.0

    def test_handles_nan_in_rewards(self):
        """Should replace NaN rewards with -max_residual."""
        log_probs = torch.tensor([[-1.0, -1.0]])
        advantages = torch.tensor([[float("nan"), float("nan")]])
        config = {"logZ": 0.0, "tb_max_residual": 100.0}

        loss, _ = compute_trajectory_balance_loss(log_probs, log_probs, advantages, config)

        # should not be NaN
        assert not torch.isnan(loss)
        assert loss.item() > 0

    def test_handles_inf_in_rewards(self):
        """Should replace inf rewards with -max_residual."""
        log_probs = torch.tensor([[-1.0, -1.0]])
        advantages = torch.tensor([[float("inf"), float("inf")]])
        config = {"logZ": 0.0, "tb_max_residual": 100.0}

        loss, _ = compute_trajectory_balance_loss(log_probs, log_probs, advantages, config)

        # should not be inf
        assert not torch.isinf(loss)

    def test_works_without_mask(self):
        """Should work when loss_mask is None."""
        log_probs = torch.randn(2, 5)
        advantages = torch.randn(2, 5)
        config = {"logZ": 0.0}

        loss, clip_ratio = compute_trajectory_balance_loss(
            log_probs, log_probs, advantages, config, loss_mask=None
        )

        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_respects_mask(self):
        """Should only sum log_probs at masked positions."""
        log_probs = torch.tensor(
            [
                [-1.0, -1.0, -999.0],  # last should be ignored
            ]
        )
        advantages = torch.tensor([[1.0, 1.0, 1.0]])
        loss_mask = torch.tensor([[1.0, 1.0, 0.0]])  # mask out last
        config = {"logZ": 0.0}

        loss, _ = compute_trajectory_balance_loss(
            log_probs, log_probs, advantages, config, loss_mask
        )

        # log_pi = -1 + -1 = -2 (not -1001)
        # log_R = 1.0 (averaged from masked positions)
        # residual = 0 + (-2) - 1 = -3
        # loss = 9
        expected_loss = 9.0
        assert torch.isclose(loss, torch.tensor(expected_loss), atol=0.1)

    def test_config_types(self):
        """Should work with dict and DictConfig."""
        log_probs = torch.randn(2, 5)
        advantages = torch.randn(2, 5)

        # plain dict
        loss1, _ = compute_trajectory_balance_loss(log_probs, log_probs, advantages, {"logZ": 0.5})

        # omegaconf DictConfig
        from omegaconf import DictConfig

        cfg = DictConfig({"logZ": 0.5})
        loss2, _ = compute_trajectory_balance_loss(log_probs, log_probs, advantages, cfg)

        # should give same result
        assert torch.isclose(loss1, loss2)


class TestSkyRLRegistration:
    """Test registration with SkyRL."""

    @pytest.mark.skipif(not SKYRL_REGISTERED, reason="SkyRL not available")
    def test_tb_identity_registered(self):
        """tb_identity should be registered as advantage estimator."""
        from skyrl_train.utils.ppo_utils import AdvantageEstimatorRegistry

        available = AdvantageEstimatorRegistry.list_available()
        assert "tb_identity" in available

    @pytest.mark.skipif(not SKYRL_REGISTERED, reason="SkyRL not available")
    def test_trajectory_balance_registered(self):
        """trajectory_balance should be registered as policy loss."""
        from skyrl_train.utils.ppo_utils import PolicyLossRegistry

        available = PolicyLossRegistry.list_available()
        assert "trajectory_balance" in available

    @pytest.mark.skipif(not SKYRL_REGISTERED, reason="SkyRL not available")
    def test_can_retrieve_tb_identity(self):
        """Should be able to get tb_identity from registry."""
        from skyrl_train.utils.ppo_utils import AdvantageEstimatorRegistry

        estimator = AdvantageEstimatorRegistry.get("tb_identity")
        assert callable(estimator)

        # test it works
        token_rewards = torch.randn(2, 5)
        mask = torch.ones(2, 5)
        advantages, returns = estimator(token_rewards, mask)
        assert advantages.shape == token_rewards.shape

    @pytest.mark.skipif(not SKYRL_REGISTERED, reason="SkyRL not available")
    def test_can_retrieve_trajectory_balance(self):
        """Should be able to get trajectory_balance from registry."""
        from skyrl_train.utils.ppo_utils import PolicyLossRegistry

        loss_fn = PolicyLossRegistry.get("trajectory_balance")
        assert callable(loss_fn)

        # test it works
        log_probs = torch.randn(2, 5)
        advantages = torch.randn(2, 5)
        config = {"logZ": 0.0}
        loss, clip = loss_fn(log_probs, log_probs, advantages, config)
        assert loss.shape == ()


class TestTBTrainerIntegration:
    """Test TBTrainer with the new TB components."""

    def test_logZ_module(self):
        """LogZModule should be a learnable parameter."""
        from synthstats.training.tb_trainer import LogZModule

        module = LogZModule(init_value=1.5)

        assert module.logZ.item() == 1.5
        assert module.logZ.requires_grad

    def test_trainer_mixin_inject_logZ(self):
        """TBTrainerMixin should inject logZ into config."""
        from synthstats.training.tb_trainer import TBTrainerMixin

        class DummyTrainer(TBTrainerMixin):
            pass

        trainer = DummyTrainer(logZ_init=2.5)
        config = {"algorithm": {}}

        trainer.inject_logZ_into_config(config)

        # float value should be set for compatibility
        assert config["algorithm"]["logZ"] == 2.5

    def test_trainer_mixin_inject_logZ_tensor(self):
        """TBTrainerMixin should inject logZ tensor on compatible config objects."""
        from omegaconf import DictConfig

        from synthstats.training.tb_trainer import TBTrainerMixin

        class DummyTrainer(TBTrainerMixin):
            pass

        trainer = DummyTrainer(logZ_init=2.5)
        # OmegaConf DictConfig supports attribute assignment
        config = DictConfig({"algorithm": {"logZ": 0.0}})

        trainer.inject_logZ_into_config(config)

        # float value should be set for compatibility
        assert config.algorithm.logZ == 2.5
        # tensor should be set for gradient flow (only on objects that support it)
        assert hasattr(config, "_logZ_tensor")
        assert config._logZ_tensor is trainer.logZ
        assert config._logZ_tensor.requires_grad

    def test_trainer_mixin_step(self):
        """TBTrainerMixin should track step count."""
        from synthstats.training.tb_trainer import TBTrainerMixin

        class DummyTrainer(TBTrainerMixin):
            pass

        trainer = DummyTrainer(logZ_init=0.0)

        metrics1 = trainer.tb_optimizer_step()
        metrics2 = trainer.tb_optimizer_step()

        assert metrics1["tb_step"] == 1
        assert metrics2["tb_step"] == 2


class TestEndToEndTBLoss:
    """End-to-end tests for TB training flow."""

    def test_full_tb_pipeline(self):
        """Test complete TB training pipeline."""
        # simulate rewards from environment
        raw_rewards = torch.tensor(
            [
                [0.0, 0.0, 0.0, 10.0],  # reward at final token
                [0.0, 0.0, 5.0, 0.0],
            ]
        )
        response_mask = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 0.0],
            ]
        )

        # step 1: tb_identity converts to advantages
        advantages, _ = compute_tb_identity_advantage(raw_rewards, response_mask)

        # step 2: policy generates log_probs
        log_probs = torch.randn(2, 4)

        # step 3: TB loss with config logZ
        config = {"logZ": 1.0, "tb_max_residual": 100.0}
        loss, clip = compute_trajectory_balance_loss(
            log_probs, log_probs, advantages, config, response_mask
        )

        assert loss.shape == ()
        assert not torch.isnan(loss)
        assert clip == 0.0

    def test_gradient_flows_correctly(self):
        """Gradients should flow through TB loss."""
        log_probs = torch.randn(2, 5, requires_grad=True)
        advantages = torch.randn(2, 5)  # from tb_identity (no grad)
        config = {"logZ": 0.0}

        loss, _ = compute_trajectory_balance_loss(log_probs, log_probs, advantages, config)
        loss.backward()

        assert log_probs.grad is not None
        assert not torch.all(log_probs.grad == 0)

    def test_logZ_gradients_flow_via_tensor(self):
        """Gradients should flow to logZ when passed as tensor via _logZ_tensor.

        This test verifies the fix for the logZ gradient disconnect bug:
        - When logZ is passed as a float, a new detached tensor is created
        - When _logZ_tensor attribute is set, the tensor is used directly
        - This preserves the computational graph so logZ can learn
        """
        log_probs = torch.randn(2, 5, requires_grad=True)
        advantages = torch.randn(2, 5)

        # create logZ as a parameter (requires grad)
        logZ_param = torch.tensor(0.5, requires_grad=True)

        # simulate TBTrainer's tensor injection pattern
        class ConfigWithTensor:
            def __init__(self):
                self.logZ = 0.5  # float fallback
                self._logZ_tensor = logZ_param  # tensor for gradient flow

            def get(self, key, default=None):
                return getattr(self, key, default)

        config = ConfigWithTensor()

        loss, _ = compute_trajectory_balance_loss(log_probs, log_probs, advantages, config)
        loss.backward()

        # logZ should have gradients if tensor was used
        assert logZ_param.grad is not None, "logZ gradient is None - tensor not used!"
        assert logZ_param.grad != 0.0, "logZ gradient is zero"
