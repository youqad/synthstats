import math

import pytest
import torch

from synthstats.train.objectives.subtb_endpoint import (
    LOG_SPARSE_REWARD_DEFAULT,
    broadcast_terminal_reward,
    compute_endpoint_subtb_loss,
    create_eos_unavailable_mask,
)


class TestComputeEndpointSubTBLoss:
    def test_basic_computation(self):
        B, T = 2, 3
        log_pf = torch.tensor([[-0.5, -0.3, -0.2], [-0.4, -0.2, -0.1]])
        log_reward = torch.zeros(B, T + 1)  # all zero rewards
        eos_logprob = torch.full((B, T + 1), -1.0)  # log(1/e) stop prob
        eos_available = torch.ones(B, T + 1, dtype=torch.bool)

        loss, metrics = compute_endpoint_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
        )

        assert loss.shape == ()
        assert loss.item() >= 0
        assert "subtb_loss" in metrics
        assert "subtb_coverage" in metrics
        assert metrics["subtb_coverage"] == 1.0

    def test_partial_eos_availability(self):
        B, T = 1, 4
        log_pf = torch.tensor([[-0.5, -0.3, -0.2, -0.1]])
        log_reward = torch.zeros(B, T + 1)
        eos_logprob = torch.full((B, T + 1), -1.0)

        eos_available = torch.zeros(B, T + 1, dtype=torch.bool)
        eos_available[0, 0] = True
        eos_available[0, 4] = True

        loss, metrics = compute_endpoint_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
        )

        assert loss.item() >= 0
        assert metrics["subtb_coverage"] < 1.0
        assert metrics["subtb_valid_count"] > 0

    def test_loss_mask_application(self):
        B, T = 1, 3
        log_pf = torch.tensor([[1.0, 2.0, 3.0]])
        log_reward = torch.zeros(B, T + 1)  # makes u constant when eos_logprob is constant
        eos_logprob = torch.full((B, T + 1), -1.0)
        eos_available = torch.ones(B, T + 1, dtype=torch.bool)

        mask_all = torch.ones(B, T, dtype=torch.bool)
        loss_all, _ = compute_endpoint_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
            loss_mask=mask_all,
        )

        mask_partial = torch.tensor([[True, False, True]])
        loss_partial, _ = compute_endpoint_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
            loss_mask=mask_partial,
        )

        assert loss_all.item() != loss_partial.item()

    def test_no_valid_subtrajectories(self):
        B, T = 1, 3
        log_pf = torch.randn(B, T, requires_grad=True)
        log_reward = torch.zeros(B, T + 1)
        eos_logprob = torch.full((B, T + 1), -1.0)

        eos_available = torch.zeros(B, T + 1, dtype=torch.bool)
        eos_available[0, 0] = True

        loss, metrics = compute_endpoint_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
        )

        assert loss.item() == 0.0
        assert metrics.get("subtb_warning", 0.0) == 1.0
        assert loss.requires_grad
        loss.backward()
        assert log_pf.grad is not None

    def test_no_valid_subtrajectories_nan_safe(self):
        B, T = 1, 3
        log_pf = torch.tensor([[float("nan"), -0.5, float("inf")]], requires_grad=True)
        log_reward = torch.zeros(B, T + 1)
        eos_logprob = torch.full((B, T + 1), -1.0)

        eos_available = torch.zeros(B, T + 1, dtype=torch.bool)
        eos_available[0, 0] = True

        loss, metrics = compute_endpoint_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
        )

        assert torch.isfinite(loss), f"loss is {loss.item()}, expected finite zero"
        assert loss.item() == 0.0
        assert loss.requires_grad
        loss.backward()
        assert log_pf.grad is not None
        assert torch.all(torch.isfinite(log_pf.grad))

    def test_lambda_weighting(self):
        B, T = 1, 3
        log_pf = torch.full((B, T), -0.5)
        log_reward = torch.zeros(B, T + 1)
        eos_logprob = torch.full((B, T + 1), -1.0)
        eos_available = torch.ones(B, T + 1, dtype=torch.bool)

        loss_low, _ = compute_endpoint_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
            subtb_lambda=0.1,
        )

        loss_high, _ = compute_endpoint_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
            subtb_lambda=0.99,
        )

        assert loss_low.item() != loss_high.item()

    def test_gradient_flow(self):
        B, T = 2, 3
        log_pf = torch.randn(B, T, requires_grad=True)
        log_reward = torch.randn(B, T + 1)
        eos_logprob = torch.full((B, T + 1), -1.0)
        eos_available = torch.ones(B, T + 1, dtype=torch.bool)

        loss, _ = compute_endpoint_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
        )

        loss.backward()
        assert log_pf.grad is not None
        assert not torch.all(log_pf.grad == 0)

    def test_residual_clamping(self):
        B, T = 1, 2
        log_pf = torch.tensor([[-100.0, -100.0]])  # extreme
        log_reward = torch.tensor([[100.0, 100.0, 100.0]])  # extreme
        eos_logprob = torch.zeros(B, T + 1)
        eos_available = torch.ones(B, T + 1, dtype=torch.bool)

        loss, _ = compute_endpoint_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
            max_residual=10.0,
        )

        assert loss.item() <= 100.0 * B * T


class TestBroadcastTerminalReward:
    def test_basic_broadcast(self):
        log_reward = torch.tensor([1.0, 2.0])  # B=2
        seq_len = 3

        per_prefix = broadcast_terminal_reward(log_reward, seq_len)

        assert per_prefix.shape == (2, 4)  # [B, T+1]
        assert per_prefix[0, -1] == 1.0
        assert per_prefix[1, -1] == 2.0
        assert per_prefix[0, 0] < 0

    def test_sparse_reward_value(self):
        log_reward = torch.tensor([0.0])
        seq_len = 2

        per_prefix = broadcast_terminal_reward(log_reward, seq_len)

        assert per_prefix[0, 0].item() == pytest.approx(LOG_SPARSE_REWARD_DEFAULT, abs=1e-5)
        assert per_prefix[0, 1].item() == pytest.approx(LOG_SPARSE_REWARD_DEFAULT, abs=1e-5)
        assert per_prefix[0, 2] == 0.0


class TestCreateEOSUnavailableMask:
    def test_all_unavailable(self):
        eos_logprob, eos_available = create_eos_unavailable_mask(2, 3)

        assert eos_logprob.shape == (2, 4)
        assert eos_available.shape == (2, 4)
        assert not eos_available.any()
        assert (eos_logprob == -1e6).all()

    def test_device_handling(self):
        eos_logprob, eos_available = create_eos_unavailable_mask(1, 2, device="cpu")

        assert eos_logprob.device.type == "cpu"
        assert eos_available.device.type == "cpu"


class TestTelescopedFormula:
    def test_telescoping_property(self):
        T = 5
        log_pf = torch.randn(T)
        log_reward = torch.randn(T + 1)
        eos_logprob = torch.randn(T + 1)

        u = log_reward - eos_logprob

        delta = torch.zeros(T)
        for t in range(T):
            delta[t] = u[t] + log_pf[t] - u[t + 1]

        i, j = 1, 4
        sum_delta = delta[i:j].sum()
        endpoint_formula = u[i] + log_pf[i:j].sum() - u[j]

        assert torch.allclose(sum_delta, endpoint_formula, atol=1e-5)

    def test_full_trajectory_matches_tb(self):
        B, T = 1, 3
        log_pf = torch.tensor([[-0.5, -0.3, -0.2]])
        terminal_log_reward = torch.tensor([0.5])

        eos_logprob = torch.full((B, T + 1), -1.0)
        log_reward = broadcast_terminal_reward(terminal_log_reward, T)
        eos_available = torch.ones(B, T + 1, dtype=torch.bool)

        loss, _ = compute_endpoint_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
        )

        assert loss.item() >= 0
        assert math.isfinite(loss.item())


class TestNaNHandling:
    def test_nan_eos_with_unavailable_mask(self):
        B, T = 2, 3
        log_pf = torch.randn(B, T)
        log_reward = torch.zeros(B, T + 1)

        eos_logprob = torch.full((B, T + 1), float("nan"))
        eos_logprob[:, 0] = -1.0
        eos_logprob[:, -1] = -0.5

        eos_available = torch.zeros(B, T + 1, dtype=torch.bool)
        eos_available[:, 0] = True
        eos_available[:, -1] = True

        loss, metrics = compute_endpoint_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
        )

        assert torch.isfinite(loss), f"Loss is NaN/Inf: {loss.item()}"
        assert loss.item() >= 0
        assert metrics["subtb_valid_count"] > 0

    def test_inf_eos_with_unavailable_mask(self):
        B, T = 1, 4
        log_pf = torch.randn(B, T)
        log_reward = torch.zeros(B, T + 1)

        eos_logprob = torch.full((B, T + 1), float("-inf"))
        eos_logprob[:, 0] = -1.0
        eos_logprob[:, -1] = -0.5

        eos_available = torch.zeros(B, T + 1, dtype=torch.bool)
        eos_available[:, 0] = True
        eos_available[:, -1] = True

        loss, metrics = compute_endpoint_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
        )

        assert torch.isfinite(loss), f"Loss is NaN/Inf: {loss.item()}"
        assert loss.item() >= 0

    def test_nan_in_log_reward(self):
        B, T = 1, 3
        log_pf = torch.randn(B, T)

        log_reward = torch.full((B, T + 1), float("nan"))
        log_reward[:, -1] = 0.5

        eos_logprob = torch.full((B, T + 1), -1.0)
        eos_available = torch.ones(B, T + 1, dtype=torch.bool)

        loss, _ = compute_endpoint_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
        )

        assert torch.isfinite(loss), f"Loss is NaN/Inf: {loss.item()}"

    def test_gradient_with_nan_masked(self):
        B, T = 2, 3
        log_pf = torch.randn(B, T, requires_grad=True)
        log_reward = torch.zeros(B, T + 1)

        eos_logprob = torch.full((B, T + 1), float("nan"))
        eos_logprob[:, 0] = -1.0
        eos_logprob[:, -1] = -0.5

        eos_available = torch.zeros(B, T + 1, dtype=torch.bool)
        eos_available[:, 0] = True
        eos_available[:, -1] = True

        loss, _ = compute_endpoint_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
        )

        loss.backward()

        assert log_pf.grad is not None
        assert torch.all(torch.isfinite(log_pf.grad)), "Gradient contains NaN/Inf"


class TestEdgeCases:
    def test_single_position(self):
        B, T = 2, 1
        log_pf = torch.randn(B, T)
        log_reward = torch.randn(B, T + 1)
        eos_logprob = torch.full((B, T + 1), -1.0)
        eos_available = torch.ones(B, T + 1, dtype=torch.bool)

        loss, metrics = compute_endpoint_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
        )

        assert loss.shape == ()
        assert metrics["subtb_valid_count"] > 0

    def test_large_batch(self):
        B, T = 32, 10
        log_pf = torch.randn(B, T)
        log_reward = torch.randn(B, T + 1)
        eos_logprob = torch.full((B, T + 1), -1.0)
        eos_available = torch.ones(B, T + 1, dtype=torch.bool)

        loss, _ = compute_endpoint_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
        )

        assert loss.shape == ()
        assert math.isfinite(loss.item())


class TestLogZIntegration:
    def test_logz_overrides_u0(self):
        B, T = 1, 3
        log_pf = torch.tensor([[-0.5, -0.3, -0.2]])
        log_reward = torch.zeros(B, T + 1)
        eos_logprob = torch.full((B, T + 1), -1.0)
        eos_available = torch.ones(B, T + 1, dtype=torch.bool)

        # Without logZ, u[0] = log_reward[0] - eos_logprob[0] = 0 - (-1) = 1
        loss_no_logz, _ = compute_endpoint_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
        )

        # With logZ=5.0, u[0] should be overridden to 5.0
        loss_with_logz, _ = compute_endpoint_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
            logZ=5.0,
        )

        assert loss_no_logz.item() != loss_with_logz.item()

    def test_logz_as_tensor(self):
        B, T = 2, 3
        log_pf = torch.randn(B, T, requires_grad=True)
        log_reward = torch.zeros(B, T + 1)
        eos_logprob = torch.full((B, T + 1), -1.0)
        eos_available = torch.ones(B, T + 1, dtype=torch.bool)
        logZ = torch.tensor(0.5, requires_grad=True)

        loss, _ = compute_endpoint_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
            logZ=logZ,
        )

        loss.backward()

        assert logZ.grad is not None
        assert torch.isfinite(logZ.grad)

    def test_logz_broadcasts_to_batch(self):
        B, T = 3, 2
        log_pf = torch.randn(B, T)
        log_reward = torch.zeros(B, T + 1)
        eos_logprob = torch.full((B, T + 1), -1.0)
        eos_available = torch.ones(B, T + 1, dtype=torch.bool)

        loss, metrics = compute_endpoint_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
            logZ=torch.tensor(1.0),
        )

        assert loss.shape == ()
        assert math.isfinite(loss.item())
        assert metrics["subtb_valid_count"] > 0

    def test_logz_none_uses_default(self):
        B, T = 1, 2
        log_pf = torch.tensor([[-0.5, -0.3]])
        log_reward = torch.tensor([[2.0, 2.0, 2.0]])
        eos_logprob = torch.tensor([[-1.0, -1.0, -1.0]])
        eos_available = torch.ones(B, T + 1, dtype=torch.bool)

        # u[0] = 2.0 - (-1.0) = 3.0
        loss_none, _ = compute_endpoint_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
            logZ=None,
        )

        loss_explicit, _ = compute_endpoint_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
            logZ=3.0,
        )

        assert abs(loss_none.item() - loss_explicit.item()) < 1e-5

    def test_logz_dtype_conversion(self):
        B, T = 2, 3
        log_pf = torch.randn(B, T, dtype=torch.float64)
        log_reward = torch.zeros(B, T + 1, dtype=torch.float64)
        eos_logprob = torch.full((B, T + 1), -1.0, dtype=torch.float64)
        eos_available = torch.ones(B, T + 1, dtype=torch.bool)
        logZ = torch.tensor(0.5, dtype=torch.float32)

        loss, metrics = compute_endpoint_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
            logZ=logZ,
        )

        assert loss.dtype == torch.float64
        assert math.isfinite(loss.item())
