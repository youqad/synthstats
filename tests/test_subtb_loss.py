"""Tests for TB (Trajectory Balance) loss - WRITTEN FIRST per TDD."""

import torch
import torch.nn as nn


class TestSubTBLossShape:
    """Verify output shapes and basic behavior."""

    def test_subtb_loss_returns_scalar(self):
        from synthstats.training.losses.tb_loss import subtb_loss

        log_probs = torch.tensor([[-0.5, -0.3, -0.2], [-0.4, -0.6, -0.1]])  # [2, 3]
        loss_mask = torch.ones(2, 3, dtype=torch.bool)
        log_rewards = torch.tensor([0.5, 0.8])  # [2]
        logZ = nn.Parameter(torch.tensor(0.0))

        loss = subtb_loss(log_probs, loss_mask, log_rewards, logZ)

        assert loss.dim() == 0, "Loss should be a scalar"
        assert loss.dtype == torch.float32 or loss.dtype == torch.float64

    def test_subtb_loss_batch_size_one(self):
        from synthstats.training.losses.tb_loss import subtb_loss

        log_probs = torch.tensor([[-0.5, -0.3]])  # [1, 2]
        loss_mask = torch.ones(1, 2, dtype=torch.bool)
        log_rewards = torch.tensor([0.5])  # [1]
        logZ = nn.Parameter(torch.tensor(0.0))

        loss = subtb_loss(log_probs, loss_mask, log_rewards, logZ)

        assert loss.dim() == 0
        assert not torch.isnan(loss)


class TestSubTBLossMasking:
    """Verify masking correctly excludes tokens."""

    def test_mask_excludes_tokens(self):
        from synthstats.training.losses.tb_loss import subtb_loss

        # set up so masked tokens have very different values
        log_probs = torch.tensor([[-1.0, -1000.0, -1.0]])  # [1, 3]
        mask_all = torch.ones(1, 3, dtype=torch.bool)
        mask_middle = torch.tensor([[True, False, True]])  # exclude -1000
        log_rewards = torch.tensor([0.0])
        logZ = nn.Parameter(torch.tensor(0.0))

        loss_all = subtb_loss(log_probs, mask_all, log_rewards, logZ)
        loss_masked = subtb_loss(log_probs, mask_middle, log_rewards, logZ)

        # with mask excluding the -1000, the loss should be much smaller
        assert loss_masked < loss_all

    def test_fully_masked_sequence(self):
        """All tokens masked out should give loss based on logZ - log_R only."""
        from synthstats.training.losses.tb_loss import subtb_loss

        log_probs = torch.tensor([[-1.0, -2.0, -3.0]])  # [1, 3]
        mask_none = torch.zeros(1, 3, dtype=torch.bool)  # all masked
        log_rewards = torch.tensor([1.0])
        logZ = nn.Parameter(torch.tensor(2.0))

        loss = subtb_loss(log_probs, mask_none, log_rewards, logZ)

        # (logZ + 0 - log_R)^2 = (2.0 + 0 - 1.0)^2 = 1.0
        expected = torch.tensor(1.0)
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_partial_mask(self):
        """Verify partial masking sums only unmasked tokens."""
        from synthstats.training.losses.tb_loss import subtb_loss

        log_probs = torch.tensor([[-1.0, -2.0, -3.0]])  # [1, 3]
        mask = torch.tensor([[True, False, True]])  # sum = -1 + -3 = -4
        log_rewards = torch.tensor([0.0])
        logZ = nn.Parameter(torch.tensor(0.0))

        loss = subtb_loss(log_probs, mask, log_rewards, logZ)

        # (0 + (-4) - 0)^2 = 16.0
        expected = torch.tensor(16.0)
        assert torch.isclose(loss, expected, atol=1e-5)


class TestSubTBLossGradient:
    """Verify gradients flow correctly."""

    def test_gradient_flows_to_logZ(self):
        from synthstats.training.losses.tb_loss import subtb_loss

        log_probs = torch.tensor([[-0.5, -0.3]])
        loss_mask = torch.ones(1, 2, dtype=torch.bool)
        log_rewards = torch.tensor([0.5])
        logZ = nn.Parameter(torch.tensor(1.0))

        loss = subtb_loss(log_probs, loss_mask, log_rewards, logZ)
        loss.backward()

        assert logZ.grad is not None
        assert not torch.isnan(logZ.grad)

    def test_gradient_nonzero_when_unbalanced(self):
        from synthstats.training.losses.tb_loss import subtb_loss

        log_probs = torch.tensor([[-0.5, -0.3]])
        loss_mask = torch.ones(1, 2, dtype=torch.bool)
        log_rewards = torch.tensor([0.5])
        logZ = nn.Parameter(torch.tensor(10.0))  # way off balance

        loss = subtb_loss(log_probs, loss_mask, log_rewards, logZ)
        loss.backward()

        assert logZ.grad != 0.0

    def test_loss_is_differentiable(self):
        """Ensure entire computation graph is differentiable."""
        from synthstats.training.losses.tb_loss import subtb_loss

        log_probs = torch.tensor([[-0.5, -0.3]], requires_grad=True)
        loss_mask = torch.ones(1, 2, dtype=torch.bool)
        log_rewards = torch.tensor([0.5])
        logZ = nn.Parameter(torch.tensor(1.0))

        loss = subtb_loss(log_probs, loss_mask, log_rewards, logZ)
        loss.backward()

        assert log_probs.grad is not None


class TestSubTBLossBalance:
    """Verify loss is zero when trajectory is balanced."""

    def test_zero_loss_when_balanced(self):
        """When logZ + log_pi = log_R, loss should be 0."""
        from synthstats.training.losses.tb_loss import subtb_loss

        # set up balanced: logZ=1, sum(log_probs)=-2, log_reward=-1
        # 1 + (-2) = -1 = log_reward, so (1 + (-2) - (-1))^2 = 0
        log_probs = torch.tensor([[-1.0, -1.0]])  # sum = -2
        loss_mask = torch.ones(1, 2, dtype=torch.bool)
        log_rewards = torch.tensor([-1.0])
        logZ = nn.Parameter(torch.tensor(1.0))

        loss = subtb_loss(log_probs, loss_mask, log_rewards, logZ)

        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_nonzero_loss_when_unbalanced(self):
        """When logZ + log_pi != log_R, loss should be positive."""
        from synthstats.training.losses.tb_loss import subtb_loss

        log_probs = torch.tensor([[-1.0, -1.0]])  # sum = -2
        loss_mask = torch.ones(1, 2, dtype=torch.bool)
        log_rewards = torch.tensor([0.0])  # different from -1
        logZ = nn.Parameter(torch.tensor(0.0))  # 0 + (-2) - 0 = -2, loss = 4

        loss = subtb_loss(log_probs, loss_mask, log_rewards, logZ)

        assert loss > 0
        expected = torch.tensor(4.0)
        assert torch.isclose(loss, expected, atol=1e-5)


class TestSubTBLossRefPolicy:
    """Verify reference-policy correction."""

    def test_ref_policy_correction_zero_loss(self):
        """With ref_log_probs, balanced trajectories should yield near-zero loss."""
        from synthstats.training.losses.tb_loss import subtb_loss

        log_probs = torch.tensor([[-1.0, -1.0]])  # sum = -2
        ref_log_probs = torch.tensor([[-0.5, -0.5]])  # sum = -1
        loss_mask = torch.ones(1, 2, dtype=torch.bool)
        log_rewards = torch.tensor([-1.0])  # logZ + (-2 + 1) = -1
        logZ = nn.Parameter(torch.tensor(0.0))

        loss = subtb_loss(
            log_probs,
            loss_mask,
            log_rewards,
            logZ,
            ref_log_probs=ref_log_probs,
        )

        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


class TestSubTBLossLengthNormalization:
    """Verify optional length normalization behavior."""

    def test_length_normalization_balances_lengths(self):
        """Different lengths should normalize to comparable sums."""
        from synthstats.training.losses.tb_loss import subtb_loss

        log_probs = torch.tensor(
            [[-1.0, -1.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]]
        )
        loss_mask = torch.tensor(
            [[True, True, False, False], [True, True, True, True]]
        )
        log_rewards = torch.tensor([-1.0, -1.0])
        logZ = nn.Parameter(torch.tensor(0.0))

        loss = subtb_loss(
            log_probs,
            loss_mask,
            log_rewards,
            logZ,
            normalize_by_length=True,
        )

        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


class TestSubTBLossBatching:
    """Verify batch averaging works correctly."""

    def test_batch_averaging(self):
        """Mean over batch dimension."""
        from synthstats.training.losses.tb_loss import subtb_loss

        # batch 1: logZ + sum(lp) - log_R = 0 + (-2) - 0 = -2, squared = 4
        # batch 2: logZ + sum(lp) - log_R = 0 + (-4) - 0 = -4, squared = 16
        # mean = (4 + 16) / 2 = 10
        log_probs = torch.tensor([[-1.0, -1.0], [-2.0, -2.0]])  # [2, 2]
        loss_mask = torch.ones(2, 2, dtype=torch.bool)
        log_rewards = torch.tensor([0.0, 0.0])
        logZ = nn.Parameter(torch.tensor(0.0))

        loss = subtb_loss(log_probs, loss_mask, log_rewards, logZ)

        expected = torch.tensor(10.0)
        assert torch.isclose(loss, expected, atol=1e-5)


class TestSubTBLossEdgeCases:
    """Edge case handling."""

    def test_empty_sequence_length(self):
        """Handle sequences with zero length (all masked)."""
        from synthstats.training.losses.tb_loss import subtb_loss

        log_probs = torch.zeros(2, 0)  # [2, 0]
        loss_mask = torch.zeros(2, 0, dtype=torch.bool)
        log_rewards = torch.tensor([1.0, 2.0])
        logZ = nn.Parameter(torch.tensor(0.0))

        loss = subtb_loss(log_probs, loss_mask, log_rewards, logZ)

        # (0 - 1)^2 + (0 - 2)^2 = 1 + 4 = 5, mean = 2.5
        expected = torch.tensor(2.5)
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_single_token(self):
        """Single token per sequence."""
        from synthstats.training.losses.tb_loss import subtb_loss

        log_probs = torch.tensor([[-0.5]])  # [1, 1]
        loss_mask = torch.ones(1, 1, dtype=torch.bool)
        log_rewards = torch.tensor([0.0])
        logZ = nn.Parameter(torch.tensor(0.5))

        loss = subtb_loss(log_probs, loss_mask, log_rewards, logZ)

        # (0.5 + (-0.5) - 0)^2 = 0
        expected = torch.tensor(0.0)
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_negative_log_rewards(self):
        """Negative log rewards (R < 1) should work."""
        from synthstats.training.losses.tb_loss import subtb_loss

        log_probs = torch.tensor([[-1.0, -1.0]])
        loss_mask = torch.ones(1, 2, dtype=torch.bool)
        log_rewards = torch.tensor([-5.0])  # very small reward
        logZ = nn.Parameter(torch.tensor(0.0))

        loss = subtb_loss(log_probs, loss_mask, log_rewards, logZ)

        # (0 + (-2) - (-5))^2 = (3)^2 = 9
        expected = torch.tensor(9.0)
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_large_logZ(self):
        """Large logZ values should not cause numerical issues."""
        from synthstats.training.losses.tb_loss import subtb_loss

        log_probs = torch.tensor([[-1.0, -1.0]])
        loss_mask = torch.ones(1, 2, dtype=torch.bool)
        log_rewards = torch.tensor([0.0])
        logZ = nn.Parameter(torch.tensor(100.0))

        loss = subtb_loss(log_probs, loss_mask, log_rewards, logZ)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_inf_log_rewards_handled(self):
        """Infinite log_rewards (from zero reward) should be handled gracefully."""
        from synthstats.training.losses.tb_loss import subtb_loss

        log_probs = torch.tensor([[-1.0, -1.0]])
        loss_mask = torch.ones(1, 2, dtype=torch.bool)
        log_rewards = torch.tensor([float("-inf")])  # from log(0)
        logZ = nn.Parameter(torch.tensor(0.0))

        loss = subtb_loss(log_probs, loss_mask, log_rewards, logZ)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_nan_log_rewards_handled(self):
        """NaN log_rewards should be handled gracefully."""
        from synthstats.training.losses.tb_loss import subtb_loss

        log_probs = torch.tensor([[-1.0, -1.0]])
        loss_mask = torch.ones(1, 2, dtype=torch.bool)
        log_rewards = torch.tensor([float("nan")])
        logZ = nn.Parameter(torch.tensor(0.0))

        loss = subtb_loss(log_probs, loss_mask, log_rewards, logZ)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_residual_clamping(self):
        """Very large residuals should be clamped for numerical stability."""
        from synthstats.training.losses.tb_loss import subtb_loss

        log_probs = torch.tensor([[-500.0, -500.0]])  # very negative
        loss_mask = torch.ones(1, 2, dtype=torch.bool)
        log_rewards = torch.tensor([0.0])
        logZ = nn.Parameter(torch.tensor(0.0))

        loss = subtb_loss(log_probs, loss_mask, log_rewards, logZ)

        # without clamping, residual = -1000, squared = 1e6
        # with clamping to 100, residual = -100, squared = 10000
        assert loss <= 10001.0  # should be clamped
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
