import pytest
import torch

from synthstats.integrations.tinker import (
    compute_combined_tb_subtb_loss,
    compute_vanilla_tb_loss,
)
from synthstats.train.objectives.subtb_endpoint import LOG_SPARSE_REWARD_DEFAULT


class TestComputeVanillaTBLoss:
    def test_basic_computation(self):
        B, T = 2, 3
        log_pf = torch.randn(B, T)
        log_reward = torch.randn(B)
        logZ = torch.tensor(0.5)
        loss_mask = torch.ones(B, T, dtype=torch.bool)

        loss = compute_vanilla_tb_loss(log_pf, log_reward, logZ, loss_mask)

        assert loss.shape == ()
        assert loss.item() >= 0

    def test_mask_application(self):
        B, T = 1, 3
        log_pf = torch.tensor([[1.0, 2.0, 3.0]])
        log_reward = torch.tensor([0.0])
        logZ = torch.tensor(0.0)

        # include all
        mask_all = torch.ones(B, T, dtype=torch.bool)
        loss_all = compute_vanilla_tb_loss(log_pf, log_reward, logZ, mask_all)

        # exclude middle position
        mask_partial = torch.tensor([[True, False, True]])
        loss_partial = compute_vanilla_tb_loss(log_pf, log_reward, logZ, mask_partial)

        assert loss_all.item() != loss_partial.item()

    def test_inf_reward_handling(self):
        B, T = 1, 2
        log_pf = torch.randn(B, T)
        log_reward = torch.tensor([float("inf")])
        logZ = torch.tensor(0.0)
        loss_mask = torch.ones(B, T, dtype=torch.bool)

        loss = compute_vanilla_tb_loss(log_pf, log_reward, logZ, loss_mask)

        assert torch.isfinite(loss)

    def test_gradient_flow(self):
        B, T = 2, 3
        log_pf = torch.randn(B, T)
        log_reward = torch.randn(B)
        logZ = torch.tensor(0.5, requires_grad=True)
        loss_mask = torch.ones(B, T, dtype=torch.bool)

        loss = compute_vanilla_tb_loss(log_pf, log_reward, logZ, loss_mask)
        loss.backward()

        assert logZ.grad is not None
        assert not torch.all(logZ.grad == 0)


class TestComputeCombinedTBSubTBLoss:
    def test_tb_only_mode(self):
        B, T = 2, 3
        log_pf = torch.randn(B, T)
        log_reward = torch.randn(B)
        logZ = torch.tensor(0.5)
        loss_mask = torch.ones(B, T, dtype=torch.bool)

        loss, metrics = compute_combined_tb_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            logZ=logZ,
            loss_mask=loss_mask,
            # no eos_logprob/eos_available → TB-only
        )

        assert "loss/tb" in metrics
        assert "loss/total" in metrics
        assert metrics["coverage/eos_available"] == pytest.approx(0.0)  # no EOS

    def test_with_eos_info(self):
        B, T = 2, 3
        log_pf = torch.randn(B, T)
        log_reward = torch.randn(B)
        logZ = torch.tensor(0.5)
        loss_mask = torch.ones(B, T, dtype=torch.bool)
        eos_logprob = torch.full((B, T + 1), -1.0)
        eos_available = torch.ones(B, T + 1, dtype=torch.bool)

        loss, metrics = compute_combined_tb_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            logZ=logZ,
            loss_mask=loss_mask,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
        )

        assert "loss/tb" in metrics
        assert "loss/subtb" in metrics
        assert "loss/total" in metrics
        assert metrics["coverage/eos_available"] == pytest.approx(1.0)
        assert metrics["loss/subtb"] != pytest.approx(0.0)

    def test_with_eos_info_padded_from_bt(self):
        B, T = 2, 3
        log_pf = torch.randn(B, T)
        log_reward = torch.randn(B)
        logZ = torch.tensor(0.5)
        loss_mask = torch.ones(B, T, dtype=torch.bool)
        eos_logprob = torch.full((B, T), -1.0)
        eos_available = torch.ones(B, T, dtype=torch.bool)

        loss, metrics = compute_combined_tb_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            logZ=logZ,
            loss_mask=loss_mask,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
        )

        assert "loss/subtb" in metrics
        assert metrics["coverage/eos_available"] == pytest.approx(1.0)
        assert metrics["subtb/valid_count"] > 0

    def test_partial_eos_info_raises(self):
        B, T = 2, 3
        log_pf = torch.randn(B, T)
        log_reward = torch.randn(B)
        logZ = torch.tensor(0.5)
        loss_mask = torch.ones(B, T, dtype=torch.bool)
        eos_logprob = torch.full((B, T), -1.0)

        with pytest.raises(ValueError, match="eos_logprob and eos_available"):
            compute_combined_tb_subtb_loss(
                log_pf=log_pf,
                log_reward=log_reward,
                logZ=logZ,
                loss_mask=loss_mask,
                eos_logprob=eos_logprob,
                eos_available=None,
            )

    def test_subtb_respects_loss_mask(self):
        B, T = 1, 3
        log_pf = torch.tensor([[1.0, 2.0, 3.0]])
        log_reward = torch.tensor([0.0])
        logZ = torch.tensor(0.0)
        eos_logprob = torch.full((B, T + 1), -1.0)
        eos_available = torch.ones(B, T + 1, dtype=torch.bool)

        mask_all = torch.ones(B, T, dtype=torch.bool)
        _, metrics_all = compute_combined_tb_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            logZ=logZ,
            loss_mask=mask_all,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
        )

        mask_partial = torch.tensor([[True, False, True]])
        _, metrics_partial = compute_combined_tb_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            logZ=logZ,
            loss_mask=mask_partial,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
        )

        assert metrics_all["loss/subtb"] != metrics_partial["loss/subtb"]

    def test_ab_subtb_alpha_weighting(self):
        B, T = 1, 2
        log_pf = torch.zeros(B, T)
        log_reward = torch.zeros(B)
        logZ = torch.tensor(0.0)
        loss_mask = torch.ones(B, T, dtype=torch.bool)
        eos_logprob = torch.zeros(B, T + 1)
        eos_available = torch.ones(B, T + 1, dtype=torch.bool)

        # low alpha
        loss_low, metrics_low = compute_combined_tb_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            logZ=logZ,
            loss_mask=loss_mask,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
            ab_subtb_alpha=0.01,
        )

        # high alpha
        loss_high, metrics_high = compute_combined_tb_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            logZ=logZ,
            loss_mask=loss_mask,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
            ab_subtb_alpha=1.0,
        )

        assert abs(metrics_low["loss/tb"] - metrics_high["loss/tb"]) < 1e-5
        assert metrics_low["loss/subtb"] > 0
        assert loss_low.item() != loss_high.item()

    def test_gradient_flow_combined(self):
        B, T = 2, 3
        log_pf = torch.randn(B, T, requires_grad=True)
        log_reward = torch.randn(B)
        logZ = torch.tensor(0.5, requires_grad=True)
        loss_mask = torch.ones(B, T, dtype=torch.bool)
        eos_logprob = torch.full((B, T + 1), -1.0)
        eos_available = torch.ones(B, T + 1, dtype=torch.bool)

        loss, _ = compute_combined_tb_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            logZ=logZ,
            loss_mask=loss_mask,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
        )

        loss.backward()

        assert log_pf.grad is not None
        assert not torch.all(log_pf.grad == 0)
        assert logZ.grad is not None
        assert not torch.all(logZ.grad == 0)

    def test_partial_eos_coverage(self):
        B, T = 1, 4
        log_pf = torch.randn(B, T)
        log_reward = torch.randn(B)
        logZ = torch.tensor(0.5)
        loss_mask = torch.ones(B, T, dtype=torch.bool)
        eos_logprob = torch.full((B, T + 1), -1.0)

        # only endpoints have EOS
        eos_available = torch.zeros(B, T + 1, dtype=torch.bool)
        eos_available[0, 0] = True
        eos_available[0, -1] = True

        loss, metrics = compute_combined_tb_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            logZ=logZ,
            loss_mask=loss_mask,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
        )

        assert metrics["coverage/eos_available"] == pytest.approx(2 / 5)  # 2 out of 5 positions
        assert metrics["subtb/valid_count"] > 0  # full trajectory valid


class TestResidualClamping:
    def test_clamped_residual_returned(self):
        B, T = 1, 2
        log_pf = torch.tensor([[100.0, 100.0]])  # very high
        log_reward = torch.tensor([-100.0])  # very low
        logZ = torch.tensor(0.0)
        loss_mask = torch.ones(B, T, dtype=torch.bool)
        max_residual = 10.0

        loss, residual = compute_vanilla_tb_loss(
            log_pf,
            log_reward,
            logZ,
            loss_mask,
            max_residual=max_residual,
            return_residual=True,
        )

        # unclamped: logZ + sum(log_pf) - log_reward = 0 + 200 + 100 = 300
        unclamped_residual = 0.0 + 200.0 - (-100.0)
        assert unclamped_residual > max_residual, "Test setup: should trigger clamping"

        # Returned residual should be clamped
        assert abs(residual) <= max_residual, (
            f"Residual {residual} exceeds max_residual {max_residual}. "
            "The returned residual should be clamped like the loss uses."
        )

    def test_non_saturated_clamping_consistent(self):
        B, T = 1, 2
        # Use moderate values that won't saturate the clamp
        log_pf = torch.tensor([[1.0, 1.0]])
        log_reward = torch.tensor([0.0])
        logZ = torch.tensor(0.0, requires_grad=True)
        loss_mask = torch.ones(B, T, dtype=torch.bool)
        max_residual = 100.0  # large enough to not saturate

        loss, residual = compute_vanilla_tb_loss(
            log_pf,
            log_reward,
            logZ,
            loss_mask,
            max_residual=max_residual,
            return_residual=True,
        )

        loss.backward()

        expected_residual = 2.0  # logZ + sum(log_pf) - log_reward = 0 + 2 - 0
        assert abs(residual - expected_residual) < 1e-5

        expected_grad = 2.0 * expected_residual  # dL/dlogZ = 2 * residual
        assert abs(logZ.grad.item() - expected_grad) < 1e-5

    def test_saturated_clamp_gradient_zero(self):
        B, T = 1, 2
        log_pf = torch.tensor([[50.0, 50.0]])
        log_reward = torch.tensor([-50.0])
        logZ = torch.tensor(0.0, requires_grad=True)
        loss_mask = torch.ones(B, T, dtype=torch.bool)
        max_residual = 10.0  # will saturate

        loss, residual = compute_vanilla_tb_loss(
            log_pf,
            log_reward,
            logZ,
            loss_mask,
            max_residual=max_residual,
            return_residual=True,
        )

        loss.backward()

        assert residual == max_residual
        # saturated clamp zeros the gradient (expected PyTorch behavior)
        assert logZ.grad.item() == 0.0


class TestVanillaTBDeviceHandling:
    def test_logz_float_with_cpu_tensors(self):
        B, T = 2, 3
        log_pf = torch.randn(B, T)
        log_reward = torch.randn(B)
        logZ = 0.5  # Python float, not tensor
        loss_mask = torch.ones(B, T, dtype=torch.bool)

        loss = compute_vanilla_tb_loss(log_pf, log_reward, logZ, loss_mask)
        assert loss.device.type == "cpu"

    def test_logz_tensor_dtype_conversion(self):
        B, T = 2, 3
        log_pf = torch.randn(B, T, dtype=torch.float32)
        log_reward = torch.randn(B, dtype=torch.float32)
        logZ = torch.tensor(0.5, dtype=torch.float64)  # Different dtype
        loss_mask = torch.ones(B, T, dtype=torch.bool)

        loss = compute_vanilla_tb_loss(log_pf, log_reward, logZ, loss_mask)
        assert loss.dtype == torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_logz_cpu_with_gpu_tensors(self):
        B, T = 2, 3
        log_pf = torch.randn(B, T, device="cuda")
        log_reward = torch.randn(B, device="cuda")
        logZ = torch.tensor(0.5)  # CPU tensor
        loss_mask = torch.ones(B, T, dtype=torch.bool, device="cuda")

        loss = compute_vanilla_tb_loss(log_pf, log_reward, logZ, loss_mask)
        assert loss.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_logz_gpu_with_gpu_tensors(self):
        B, T = 2, 3
        log_pf = torch.randn(B, T, device="cuda")
        log_reward = torch.randn(B, device="cuda")
        logZ = torch.tensor(0.5, device="cuda")
        loss_mask = torch.ones(B, T, dtype=torch.bool, device="cuda")

        loss = compute_vanilla_tb_loss(log_pf, log_reward, logZ, loss_mask)
        assert loss.device.type == "cuda"

    def test_return_residual_preserves_device(self):
        B, T = 2, 3
        log_pf = torch.randn(B, T)
        log_reward = torch.randn(B)
        logZ = torch.tensor(0.5)
        loss_mask = torch.ones(B, T, dtype=torch.bool)

        loss, residual = compute_vanilla_tb_loss(
            log_pf, log_reward, logZ, loss_mask, return_residual=True
        )

        # residual is returned as .item(), should be Python float
        assert isinstance(residual, float)


class TestLogSparseRewardPassthrough:
    def test_default_uses_log_sparse_reward_default(self):
        B, T = 1, 3
        log_pf = torch.zeros(B, T)
        log_reward = torch.tensor([1.0])
        logZ = torch.tensor(0.0)
        loss_mask = torch.ones(B, T, dtype=torch.bool)
        eos_logprob = torch.full((B, T + 1), -1.0)
        eos_available = torch.ones(B, T + 1, dtype=torch.bool)

        _, metrics_default = compute_combined_tb_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            logZ=logZ,
            loss_mask=loss_mask,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
        )

        assert metrics_default["loss/subtb"] >= 0

    def test_custom_sparse_reward_changes_subtb_loss(self):
        B, T = 1, 3
        log_pf = torch.zeros(B, T)
        log_reward = torch.tensor([1.0])
        logZ = torch.tensor(0.0)
        loss_mask = torch.ones(B, T, dtype=torch.bool)
        eos_logprob = torch.full((B, T + 1), -1.0)
        eos_available = torch.ones(B, T + 1, dtype=torch.bool)

        # default sparse reward
        _, metrics_default = compute_combined_tb_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            logZ=logZ,
            loss_mask=loss_mask,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
        )

        # scaled sparse reward (simulating temp=0.5 annealing)
        scaled_sparse = LOG_SPARSE_REWARD_DEFAULT / 0.5
        _, metrics_scaled = compute_combined_tb_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            logZ=logZ,
            loss_mask=loss_mask,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
            log_sparse_reward=scaled_sparse,
        )

        assert metrics_default["loss/subtb"] != pytest.approx(
            metrics_scaled["loss/subtb"], abs=1e-6
        )

    def test_none_sparse_reward_same_as_omitted(self):
        B, T = 1, 2
        log_pf = torch.randn(B, T)
        log_reward = torch.randn(B)
        logZ = torch.tensor(0.5)
        loss_mask = torch.ones(B, T, dtype=torch.bool)
        eos_logprob = torch.full((B, T + 1), -1.0)
        eos_available = torch.ones(B, T + 1, dtype=torch.bool)

        loss_omitted, metrics_omitted = compute_combined_tb_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            logZ=logZ,
            loss_mask=loss_mask,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
        )

        loss_none, metrics_none = compute_combined_tb_subtb_loss(
            log_pf=log_pf,
            log_reward=log_reward,
            logZ=logZ,
            loss_mask=loss_mask,
            eos_logprob=eos_logprob,
            eos_available=eos_available,
            log_sparse_reward=None,
        )

        assert loss_omitted.item() == pytest.approx(loss_none.item(), abs=1e-7)
        assert metrics_omitted["loss/subtb"] == pytest.approx(metrics_none["loss/subtb"], abs=1e-7)
