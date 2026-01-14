"""Property-based tests for SubTB loss using Hypothesis.

Tests mathematical invariants and edge case handling of the TB loss function.
"""

import torch
import torch.nn as nn
from hypothesis import given, settings
from hypothesis import strategies as st

from synthstats.training.losses.tb_loss import subtb_loss


class TestSubTBLossProperties:
    """Property-based tests for subtb_loss."""

    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        seq_len=st.integers(min_value=1, max_value=20),
        logZ_val=st.floats(
            min_value=-10.0,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=50)
    def test_loss_always_non_negative(self, batch_size: int, seq_len: int, logZ_val: float):
        """TB loss is a squared error, so it must be >= 0."""
        log_probs = torch.randn(batch_size, seq_len) * 2 - 1  # roughly [-3, 1]
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        log_rewards = torch.randn(batch_size)
        logZ = nn.Parameter(torch.tensor(logZ_val))

        loss = subtb_loss(log_probs, loss_mask, log_rewards, logZ)

        assert loss.item() >= 0, f"Loss must be non-negative, got {loss.item()}"

    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        seq_len=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=50)
    def test_gradient_always_exists_for_logZ(self, batch_size: int, seq_len: int):
        """Gradient for logZ must exist and be finite after backward."""
        log_probs = torch.randn(batch_size, seq_len, requires_grad=True)
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        log_rewards = torch.randn(batch_size)
        logZ = nn.Parameter(torch.tensor(0.0))

        loss = subtb_loss(log_probs, loss_mask, log_rewards, logZ)
        loss.backward()

        assert logZ.grad is not None, "logZ.grad must not be None"
        assert torch.isfinite(logZ.grad), f"logZ.grad must be finite, got {logZ.grad}"

    @given(
        batch_size=st.integers(min_value=2, max_value=8),
        seq_len=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=30)
    def test_batch_order_invariance(self, batch_size: int, seq_len: int):
        """Loss should be invariant to batch ordering (since it's mean over batch)."""
        log_probs = torch.randn(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        log_rewards = torch.randn(batch_size)
        logZ = nn.Parameter(torch.tensor(0.5))

        # original order
        loss1 = subtb_loss(log_probs, loss_mask, log_rewards, logZ)

        # shuffled order
        perm = torch.randperm(batch_size)
        loss2 = subtb_loss(log_probs[perm], loss_mask[perm], log_rewards[perm], logZ)

        assert torch.allclose(loss1, loss2, atol=1e-5), (
            f"Loss should be batch-order invariant: {loss1.item()} vs {loss2.item()}"
        )

    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        seq_len=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=30)
    def test_all_masked_gives_logZ_minus_reward_squared(self, batch_size: int, seq_len: int):
        """When all tokens are masked out, loss = (logZ - log_R)^2."""
        log_probs = torch.randn(batch_size, seq_len)
        loss_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)  # all False
        log_rewards = torch.randn(batch_size)
        logZ_val = 1.5
        logZ = nn.Parameter(torch.tensor(logZ_val))

        loss = subtb_loss(log_probs, loss_mask, log_rewards, logZ)

        # expected: mean((logZ - log_R)^2)
        expected = ((logZ_val - log_rewards) ** 2).mean()
        assert torch.allclose(loss, expected, atol=1e-5), (
            "All-masked loss should be (logZ - log_R)^2: "
            f"got {loss.item()}, expected {expected.item()}"
        )

    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=30)
    def test_balanced_trajectory_zero_loss(self, batch_size: int, seq_len: int):
        """When logZ + sum(log_pi) = log_R exactly, loss should be 0."""
        logZ_val = 2.0
        logZ = nn.Parameter(torch.tensor(logZ_val))

        # construct log_probs such that sum = log_R - logZ for each trajectory
        log_rewards = torch.randn(batch_size)
        target_sums = log_rewards - logZ_val  # [B]

        # distribute target_sum across seq_len tokens
        log_probs = target_sums.unsqueeze(1).expand(-1, seq_len) / seq_len
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        loss = subtb_loss(log_probs, loss_mask, log_rewards, logZ)

        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5), (
            f"Balanced trajectory should have zero loss, got {loss.item()}"
        )

    @given(
        batch_size=st.integers(min_value=2, max_value=4),
        seq_len=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=30)
    def test_handles_inf_nan_rewards_gracefully(self, batch_size: int, seq_len: int):
        """Loss should handle inf/NaN in log_rewards without crashing or producing NaN.

        The implementation replaces inf/NaN with -max_residual (default -100.0),
        preserving the "low reward" signal for GFlowNet training. Using 0.0 would
        incorrectly treat failures as neutral/success cases.
        """
        log_probs = torch.randn(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        logZ = nn.Parameter(torch.tensor(0.5))

        # inject inf and nan (batch_size >= 2 guaranteed)
        log_rewards = torch.randn(batch_size)
        log_rewards[0] = float("inf")
        log_rewards[1] = float("nan")

        loss = subtb_loss(log_probs, loss_mask, log_rewards, logZ)

        assert torch.isfinite(loss), f"Loss should be finite even with inf/nan rewards, got {loss}"

    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=30)
    def test_residual_clamping(self, batch_size: int, seq_len: int):
        """Residual clamping should limit loss to max_residual^2."""
        log_probs = torch.ones(batch_size, seq_len) * -100  # very negative
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        log_rewards = torch.ones(batch_size) * 100  # very positive
        logZ = nn.Parameter(torch.tensor(100.0))

        max_residual = 100.0
        loss = subtb_loss(log_probs, loss_mask, log_rewards, logZ, max_residual=max_residual)

        # loss should be at most max_residual^2
        assert loss.item() <= max_residual**2 + 1.0, (
            f"Loss should be clamped to ~{max_residual**2}, got {loss.item()}"
        )
