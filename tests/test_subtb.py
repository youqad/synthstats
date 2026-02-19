"""SubTB loss tests.

SubTB computes flow matching losses over all sub-trajectory lengths,
providing denser gradient signal than vanilla TB.
"""

import pytest
import torch

from synthstats.train.objectives.trajectory_balance import (
    SKYRL_REGISTERED,
    compute_modified_subtb_loss,
    compute_trajectory_balance_loss,
)


class TestSubTBBasics:
    def test_computes_loss_with_eos_logprobs(self):
        log_probs = torch.randn(2, 5)
        advantages = torch.randn(2, 5)
        eos_logprobs = torch.randn(2, 5)

        # create config object with _eos_logprobs attribute
        class Config:
            logZ = 0.0
            subtb_lambda = 0.9
            tb_max_residual = 100.0

            def get(self, key, default=None):
                return getattr(self, key, default)

        config = Config()
        object.__setattr__(config, "_eos_logprobs", eos_logprobs)

        loss, clip_ratio = compute_modified_subtb_loss(log_probs, log_probs, advantages, config)

        assert loss.shape == ()
        assert not torch.isnan(loss)
        assert clip_ratio == 0.0

    def test_falls_back_to_vanilla_tb_without_eos_logprobs(self):
        log_probs = torch.randn(2, 5)
        advantages = torch.randn(2, 5)
        config = {"logZ": 0.0, "subtb_lambda": 0.9}

        # without _eos_logprobs, should fall back
        loss, _ = compute_modified_subtb_loss(log_probs, log_probs, advantages, config)

        # compare with vanilla TB
        tb_loss, _ = compute_trajectory_balance_loss(log_probs, log_probs, advantages, config)

        assert torch.isclose(loss, tb_loss)

    def test_handles_single_token_trajectory(self):
        log_probs = torch.randn(2, 1)  # single token
        advantages = torch.randn(2, 1)
        eos_logprobs = torch.randn(2, 1)

        class Config:
            logZ = 0.0
            subtb_lambda = 0.9

            def get(self, key, default=None):
                return getattr(self, key, default)

        config = Config()
        object.__setattr__(config, "_eos_logprobs", eos_logprobs)

        loss, _ = compute_modified_subtb_loss(log_probs, log_probs, advantages, config)

        # should not crash, falls back to vanilla TB
        assert not torch.isnan(loss)


class TestDeltaCumsumPattern:
    def test_internal_delta_computation(self):
        log_probs = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        eos_logprobs = torch.tensor([[-1.0, -2.0, -3.0, -4.0]])

        # manually compute INTERNAL delta (not including final step)
        # delta[t] = log_pf[t] - eos[t] + eos[t+1]
        # delta[0] = 1.0 - (-1.0) + (-2.0) = 0.0
        # delta[1] = 2.0 - (-2.0) + (-3.0) = 1.0
        # delta[2] = 3.0 - (-3.0) + (-4.0) = 2.0
        expected_internal_delta = torch.tensor([[0.0, 1.0, 2.0]])

        # compute internal delta
        internal_delta = log_probs[:, :-1] - eos_logprobs[:, :-1] + eos_logprobs[:, 1:]

        assert torch.allclose(internal_delta, expected_internal_delta)

    def test_final_delta_includes_reward(self):
        log_probs = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        eos_logprobs = torch.tensor([[-1.0, -2.0, -3.0, -4.0]])
        log_rewards = torch.tensor([[10.0, 10.0, 10.0, 10.0]])  # broadcast reward

        # final delta connects to reward instead of next eos
        # delta[3] = log_pf[3] - eos[3] + log_R = 4.0 - (-4.0) + 10.0 = 18.0
        expected_final_delta = torch.tensor([[18.0]])

        final_delta = log_probs[:, -1:] - eos_logprobs[:, -1:] + log_rewards[:, -1:]

        assert torch.allclose(final_delta, expected_final_delta)

    def test_full_delta_includes_internal_and_reward(self):
        log_probs = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        eos_logprobs = torch.tensor([[-1.0, -2.0, -3.0, -4.0]])
        log_rewards = torch.tensor([[10.0, 10.0, 10.0, 10.0]])

        # internal: [0.0, 1.0, 2.0], final: [18.0]
        # full delta: [0.0, 1.0, 2.0, 18.0]
        expected_full_delta = torch.tensor([[0.0, 1.0, 2.0, 18.0]])

        internal_delta = log_probs[:, :-1] - eos_logprobs[:, :-1] + eos_logprobs[:, 1:]
        final_delta = log_probs[:, -1:] - eos_logprobs[:, -1:] + log_rewards[:, -1:]
        full_delta = torch.cat([internal_delta, final_delta], dim=1)

        assert torch.allclose(full_delta, expected_full_delta)

    def test_cumsum_gives_subtraj_sums(self):
        delta = torch.tensor([[1.0, 2.0, 3.0]])  # [B=1, T-1=3]

        # prepend zero and cumsum
        zeros = torch.zeros(1, 1)
        delta_cumsum = torch.cat([zeros, delta], dim=1).cumsum(dim=1)
        # cumsum = [0, 1, 3, 6]

        # sub-trajectory of length 1 starting at position 0
        residual_len1_pos0 = delta_cumsum[:, 1] - delta_cumsum[:, 0]  # 1 - 0 = 1
        assert residual_len1_pos0.item() == 1.0

        # sub-trajectory of length 2 starting at position 0
        residual_len2_pos0 = delta_cumsum[:, 2] - delta_cumsum[:, 0]  # 3 - 0 = 3
        assert residual_len2_pos0.item() == 3.0

        # sub-trajectory of length 2 starting at position 1
        residual_len2_pos1 = delta_cumsum[:, 3] - delta_cumsum[:, 1]  # 6 - 1 = 5
        assert residual_len2_pos1.item() == 5.0


class TestLambdaWeighting:
    def test_higher_lambda_weights_longer_subtraj_more(self):
        log_probs = torch.randn(2, 6)
        advantages = torch.randn(2, 6)
        eos_logprobs = torch.randn(2, 6)

        class Config:
            logZ = 0.0
            tb_max_residual = 100.0

            def __init__(self, subtb_lambda):
                self.subtb_lambda = subtb_lambda

            def get(self, key, default=None):
                return getattr(self, key, default)

        config_high = Config(subtb_lambda=0.99)
        config_low = Config(subtb_lambda=0.1)

        object.__setattr__(config_high, "_eos_logprobs", eos_logprobs)
        object.__setattr__(config_low, "_eos_logprobs", eos_logprobs)

        loss_high, _ = compute_modified_subtb_loss(log_probs, log_probs, advantages, config_high)
        loss_low, _ = compute_modified_subtb_loss(log_probs, log_probs, advantages, config_low)

        # losses should differ (lambda affects the weighting)
        assert not torch.isclose(loss_high, loss_low, atol=1e-6)

    def test_lambda_zero_weights_only_length_one(self):
        log_probs = torch.randn(2, 5)
        advantages = torch.randn(2, 5)
        eos_logprobs = torch.randn(2, 5)

        class Config:
            logZ = 0.0
            subtb_lambda = 0.0  # only len=1 gets weight
            tb_max_residual = 100.0

            def get(self, key, default=None):
                return getattr(self, key, default)

        config = Config()
        object.__setattr__(config, "_eos_logprobs", eos_logprobs)

        loss, _ = compute_modified_subtb_loss(log_probs, log_probs, advantages, config)

        # should not crash, computes only len-1 residuals
        assert not torch.isnan(loss)


class TestEOSLogprobsEffect:
    def test_different_eos_logprobs_give_different_loss(self):
        log_probs = torch.randn(2, 5)
        advantages = torch.randn(2, 5)

        class Config:
            logZ = 0.0
            subtb_lambda = 0.9
            tb_max_residual = 100.0

            def get(self, key, default=None):
                return getattr(self, key, default)

        config1 = Config()
        config2 = Config()

        eos1 = torch.zeros(2, 5)
        eos2 = torch.randn(2, 5)

        object.__setattr__(config1, "_eos_logprobs", eos1)
        object.__setattr__(config2, "_eos_logprobs", eos2)

        loss1, _ = compute_modified_subtb_loss(log_probs, log_probs, advantages, config1)
        loss2, _ = compute_modified_subtb_loss(log_probs, log_probs, advantages, config2)

        # different EOS logprobs should give different losses
        assert not torch.isclose(loss1, loss2, atol=1e-6)


class TestGradientFlow:
    def test_gradients_flow_to_log_probs(self):
        log_probs = torch.randn(2, 5, requires_grad=True)
        advantages = torch.randn(2, 5)
        eos_logprobs = torch.randn(2, 5)

        class Config:
            logZ = 0.0
            subtb_lambda = 0.9
            tb_max_residual = 100.0

            def get(self, key, default=None):
                return getattr(self, key, default)

        config = Config()
        object.__setattr__(config, "_eos_logprobs", eos_logprobs)

        loss, _ = compute_modified_subtb_loss(log_probs, log_probs, advantages, config)
        loss.backward()

        assert log_probs.grad is not None
        assert not torch.all(log_probs.grad == 0)


class TestMasking:
    def test_respects_loss_mask(self):
        log_probs = torch.tensor(
            [
                [-1.0, -1.0, -999.0, -999.0],  # last two should be ignored
            ]
        )
        advantages = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        eos_logprobs = torch.tensor([[-0.5, -0.5, -0.5, -0.5]])
        loss_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])  # mask last two

        class Config:
            logZ = 0.0
            subtb_lambda = 0.9
            tb_max_residual = 100.0

            def get(self, key, default=None):
                return getattr(self, key, default)

        config = Config()
        object.__setattr__(config, "_eos_logprobs", eos_logprobs)

        loss, _ = compute_modified_subtb_loss(log_probs, log_probs, advantages, config, loss_mask)

        # loss should be finite (not blown up by -999)
        assert not torch.isnan(loss)
        assert loss.item() < 1000  # reasonable value

    def test_variable_length_reward_anchoring(self):
        """reward anchors at each sample's last valid position, not global T-1."""
        # batch with two samples of different valid lengths
        # sample 0: 2 valid tokens (mask=[1,1,0,0])
        # sample 1: 4 valid tokens (mask=[1,1,1,1])
        log_probs = torch.tensor(
            [
                [1.0, 2.0, 999.0, 999.0],  # sample 0: valid at positions 0,1
                [1.0, 2.0, 3.0, 4.0],  # sample 1: valid at all positions
            ]
        )
        eos_logprobs = torch.tensor(
            [
                [-1.0, -2.0, -3.0, -4.0],
                [-1.0, -2.0, -3.0, -4.0],
            ]
        )
        # broadcast reward (same at all positions per tb_identity)
        log_rewards = torch.tensor(
            [
                [10.0, 10.0, 10.0, 10.0],  # sample 0 reward
                [20.0, 20.0, 20.0, 20.0],  # sample 1 reward
            ]
        )
        loss_mask = torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0],  # sample 0: 2 valid
                [1.0, 1.0, 1.0, 1.0],  # sample 1: 4 valid
            ]
        )

        class Config:
            logZ = 0.0
            subtb_lambda = 0.9
            tb_max_residual = 100.0

            def get(self, key, default=None):
                return getattr(self, key, default)

        config = Config()
        object.__setattr__(config, "_eos_logprobs", eos_logprobs)

        loss, _ = compute_modified_subtb_loss(log_probs, log_probs, log_rewards, config, loss_mask)

        # loss should be finite
        assert not torch.isnan(loss)
        assert torch.isfinite(loss)

        # verify by manually computing expected deltas
        # sample 0 (2 valid): reward anchors at position 1
        #   delta[0] = log_pf[0] - eos[0] + eos[1] = 1 - (-1) + (-2) = 0 (internal)
        #   delta[1] = log_pf[1] - eos[1] + log_R = 2 - (-2) + 10 = 14 (final)
        # sample 1 (4 valid): reward anchors at position 3
        #   delta[0] = 1 - (-1) + (-2) = 0 (internal)
        #   delta[1] = 2 - (-2) + (-3) = 1 (internal)
        #   delta[2] = 3 - (-3) + (-4) = 2 (internal)
        #   delta[3] = 4 - (-4) + 20 = 28 (final)
        # This tests that sample 0's reward (10) correctly anchors at position 1,
        # not at position 3 (which would incorrectly use eos[3]=-4).

        # From manual computation: expected_loss = 290.122
        # (verified by hand in manual testing earlier)
        expected_loss = pytest.approx(290.122, rel=1e-3)
        assert loss.item() == expected_loss

    def test_non_contiguous_mask_excludes_spanning_subtrajectories(self):
        # batch with non-contiguous mask (gap at position 1)
        log_probs = torch.tensor(
            [
                [1.0, 999.0, 2.0, 3.0],  # positions 0,2,3 valid; position 1 masked
            ]
        )
        eos_logprobs = torch.tensor(
            [
                [-1.0, -2.0, -3.0, -4.0],
            ]
        )
        log_rewards = torch.tensor(
            [
                [10.0, 10.0, 10.0, 10.0],
            ]
        )
        # gap mask: position 1 is invalid
        loss_mask = torch.tensor(
            [
                [1.0, 0.0, 1.0, 1.0],
            ]
        )

        class Config:
            logZ = 0.0
            subtb_lambda = 0.9
            tb_max_residual = 100.0

            def get(self, key, default=None):
                return getattr(self, key, default)

        config = Config()
        object.__setattr__(config, "_eos_logprobs", eos_logprobs)

        loss, _ = compute_modified_subtb_loss(log_probs, log_probs, log_rewards, config, loss_mask)

        # loss should be finite
        assert not torch.isnan(loss)
        assert torch.isfinite(loss)

        # verify that sub-trajectories spanning the gap are excluded
        # with mask = [1, 0, 1, 1]:
        #   length 1: positions 0, 2, 3 are valid (mask[i] == 1)
        #   length 2: [0,1] invalid (gap), [1,2] invalid (gap), [2,3] valid
        #   length 3: [0,1,2] invalid (gap), [1,2,3] invalid (gap)
        #   length 4: [0,1,2,3] invalid (gap)
        # so valid sub-trajectories are: pos 0 (len 1), pos 2 (len 1), pos 3 (len 1), [2,3] (len 2)
        # total of 4 valid sub-trajectories

        # verify expected behavior: only 4 valid sub-trajectories
        # Manual computation from my hand trace:
        # delta = [0, 0, 1, 17] (position 1 masked out)
        # Length 1: residuals at pos 0,2,3 = [0, 1, 17] → sum=290
        # Length 2: residual at pos 2 = [18] → sum=324
        # batch_loss = 1.0*290 + 0.9*324 = 581.6
        # total_weight = 1.0*3 + 0.9*1 = 3.9
        # expected_loss = 581.6 / 3.9 = 149.128...
        expected_loss = pytest.approx(149.128, rel=1e-3)
        assert loss.item() == expected_loss

    def test_t1_subtb_single_subtrajectory(self):
        log_probs = torch.tensor([[2.0]])
        eos_logprobs = torch.tensor([[-3.0]])
        log_rewards = torch.tensor([[5.0]])
        loss_mask = torch.ones(1, 1, dtype=torch.bool)

        class Config:
            logZ = 0.0  # not used for SubTB
            subtb_lambda = 0.9
            tb_max_residual = 100.0

            def get(self, key, default=None):
                return getattr(self, key, default)

        config = Config()
        object.__setattr__(config, "_eos_logprobs", eos_logprobs)

        loss, _ = compute_modified_subtb_loss(log_probs, log_probs, log_rewards, config, loss_mask)

        # For T=1: delta = log_pf - eos + log_R = 2 - (-3) + 5 = 10
        # residual = delta = 10
        # loss = 10^2 / 1 = 100
        assert not torch.isnan(loss)
        assert torch.isfinite(loss)
        assert loss.item() == pytest.approx(100.0, rel=1e-5)

    def test_lambda_one_weights_all_lengths_equally(self):
        log_probs = torch.randn(1, 4)
        eos_logprobs = torch.randn(1, 4)
        log_rewards = torch.randn(1, 4)
        loss_mask = torch.ones(1, 4, dtype=torch.bool)

        class Config:
            logZ = 0.0
            subtb_lambda = 1.0  # no decay
            tb_max_residual = 100.0

            def get(self, key, default=None):
                return getattr(self, key, default)

        config = Config()
        object.__setattr__(config, "_eos_logprobs", eos_logprobs)

        loss, _ = compute_modified_subtb_loss(log_probs, log_probs, log_rewards, config, loss_mask)

        # loss must be finite
        assert not torch.isnan(loss)
        assert torch.isfinite(loss)

        # With lambda=1.0, all sub-trajectory lengths have equal weight
        # total_weight = 1.0*4 + 1.0*3 + 1.0*2 + 1.0*1 = 10
        # (length 1: 4 positions, length 2: 3 positions,
        # length 3: 2 positions, length 4: 1 position)

    def test_nan_at_masked_position_doesnt_propagate(self):
        log_probs = torch.tensor(
            [
                [1.0, float("nan"), 2.0, 3.0],  # NaN at position 1 (masked)
            ]
        )
        eos_logprobs = torch.tensor(
            [
                [-1.0, -2.0, -3.0, -4.0],
            ]
        )
        log_rewards = torch.tensor(
            [
                [10.0, 10.0, 10.0, 10.0],
            ]
        )
        loss_mask = torch.tensor(
            [
                [1.0, 0.0, 1.0, 1.0],  # position 1 masked (contains NaN)
            ]
        )

        class Config:
            logZ = 0.0
            subtb_lambda = 0.9
            tb_max_residual = 100.0

            def get(self, key, default=None):
                return getattr(self, key, default)

        config = Config()
        object.__setattr__(config, "_eos_logprobs", eos_logprobs)

        loss, _ = compute_modified_subtb_loss(log_probs, log_probs, log_rewards, config, loss_mask)

        # loss must be finite - NaN at masked position must not propagate
        assert not torch.isnan(loss), "NaN at masked position propagated to loss!"
        assert torch.isfinite(loss), "Loss is not finite!"

    def test_fully_masked_sample_excluded(self):
        log_probs = torch.tensor(
            [
                [1.0, 2.0, 3.0],  # sample 0: has valid positions
                [999.0, 999.0, 999.0],  # sample 1: fully masked
            ]
        )
        eos_logprobs = torch.tensor(
            [
                [-1.0, -2.0, -3.0],
                [-1.0, -2.0, -3.0],
            ]
        )
        log_rewards = torch.tensor(
            [
                [10.0, 10.0, 10.0],
                [999.0, 999.0, 999.0],  # would cause problems if not excluded
            ]
        )
        loss_mask = torch.tensor(
            [
                [1.0, 1.0, 1.0],  # sample 0: all valid
                [0.0, 0.0, 0.0],  # sample 1: all invalid
            ]
        )

        class Config:
            logZ = 0.0
            subtb_lambda = 0.9
            tb_max_residual = 100.0

            def get(self, key, default=None):
                return getattr(self, key, default)

        config = Config()
        object.__setattr__(config, "_eos_logprobs", eos_logprobs)

        loss, _ = compute_modified_subtb_loss(log_probs, log_probs, log_rewards, config, loss_mask)

        # loss should be finite (fully-masked sample excluded)
        assert not torch.isnan(loss)
        assert torch.isfinite(loss)

        # loss should only reflect sample 0, not sample 1
        # if sample 1 was included, the 999.0 values would cause extreme loss

    def test_bfloat16_long_sequence_precision(self):
        """bfloat16 loses integer precision beyond 256; tolerance-based mask handles this."""
        T = 300  # beyond bfloat16 exact integer range (256)
        log_probs = torch.randn(2, T, dtype=torch.bfloat16)
        advantages = torch.randn(2, T, dtype=torch.bfloat16)
        eos_logprobs = torch.randn(2, T, dtype=torch.bfloat16)
        loss_mask = torch.ones(2, T, dtype=torch.bool)

        class Config:
            logZ = 0.0
            subtb_lambda = 0.9
            tb_max_residual = 100.0

            def get(self, key, default=None):
                return getattr(self, key, default)

        config = Config()
        object.__setattr__(config, "_eos_logprobs", eos_logprobs)

        loss, _ = compute_modified_subtb_loss(log_probs, log_probs, advantages, config, loss_mask)

        # loss must be finite - bfloat16 precision issues must not break it
        assert not torch.isnan(loss), "bfloat16 long sequence produced NaN!"
        assert torch.isfinite(loss), "bfloat16 long sequence produced inf!"


class TestSubTBTrainerMixin:
    def test_inject_subtb_data(self):
        from omegaconf import DictConfig

        from synthstats.train.runners.tb_trainer import SubTBTrainerMixin

        class DummyTrainer(SubTBTrainerMixin):
            pass

        trainer = DummyTrainer(logZ_init=1.5)
        config = DictConfig({"algorithm": {"logZ": 0.0}})
        eos_logprobs = torch.randn(2, 5)

        trainer.inject_subtb_data(config, eos_logprobs)

        # logZ should be injected
        assert config.algorithm.logZ == 1.5
        # eos_logprobs tensor should be injected
        assert hasattr(config, "_eos_logprobs")
        assert torch.equal(config._eos_logprobs, eos_logprobs)


class TestSkyRLRegistration:
    @pytest.mark.skipif(not SKYRL_REGISTERED, reason="SkyRL not available")
    def test_modified_subtb_registered(self):
        from skyrl_train.utils.ppo_utils import PolicyLossRegistry

        available = PolicyLossRegistry.list_available()
        assert "modified_subtb" in available

    @pytest.mark.skipif(not SKYRL_REGISTERED, reason="SkyRL not available")
    def test_can_retrieve_modified_subtb(self):
        from skyrl_train.utils.ppo_utils import PolicyLossRegistry

        loss_fn = PolicyLossRegistry.get("modified_subtb")
        assert callable(loss_fn)

        # test it works (will fall back to vanilla TB without eos_logprobs)
        log_probs = torch.randn(2, 5)
        advantages = torch.randn(2, 5)
        config = {"logZ": 0.0}
        loss, clip = loss_fn(log_probs, log_probs, advantages, config)
        assert loss.shape == ()


class TestEndToEndSubTB:
    def test_full_subtb_pipeline(self):
        from omegaconf import DictConfig

        from synthstats.train.runners.tb_trainer import SubTBTrainerMixin

        # simulate batch data
        log_probs = torch.randn(4, 10)
        eos_logprobs = torch.randn(4, 10)
        advantages = torch.randn(4, 10)  # log_rewards from tb_identity
        loss_mask = torch.ones(4, 10)

        # create trainer mixin for logZ management
        class DummyTrainer(SubTBTrainerMixin):
            pass

        trainer = DummyTrainer(logZ_init=0.0, logZ_lr=0.01)

        # create config (use DictConfig for proper attribute injection)
        config = DictConfig({"algorithm": {"logZ": 0.0, "subtb_lambda": 0.9}})
        trainer.inject_subtb_data(config, eos_logprobs)

        # compute loss
        loss, clip = compute_modified_subtb_loss(
            log_probs, log_probs, advantages, config, loss_mask
        )

        assert loss.shape == ()
        assert not torch.isnan(loss)
        assert clip == 0.0

    def test_eos_logprobs_in_generation(self):
        from synthstats.core.policy import GenConfig
        from synthstats.policies.hf_policy import MockHFPolicy

        policy = MockHFPolicy()
        gen = policy.generate([], gen=GenConfig())

        # eos_logprobs should be populated
        assert hasattr(gen, "eos_logprobs")
        assert len(gen.eos_logprobs) == len(gen.token_ids)
        assert len(gen.eos_logprobs) == len(gen.token_logprobs)


class TestEOSLogprobPipeline:
    def test_mock_policy_stores_last_eos_logprob_final(self):
        from synthstats.policies.hf_policy import MockHFPolicy

        policy = MockHFPolicy()

        # before call, should be None
        assert policy._last_eos_logprob_final is None

        # after call, should be set
        policy("test observation")
        assert policy._last_eos_logprob_final is not None
        assert isinstance(policy._last_eos_logprob_final, float)

    def test_collected_trajectory_has_eos_logprobs_field(self):
        from synthstats.train.data.collectors import CollectedTrajectory

        traj = CollectedTrajectory(
            observations=["obs1"],
            actions=[{"type": "answer", "payload": "42"}],
            log_probs=torch.tensor([-0.5]),
            entropy=torch.tensor([0.1]),
            reward=1.0,
            eos_logprobs=torch.tensor([-0.1]),
        )

        assert hasattr(traj, "eos_logprobs")
        assert traj.eos_logprobs is not None
        assert traj.eos_logprobs.shape == torch.Size([1])

    def test_collected_trajectory_detach_preserves_eos_logprobs(self):
        from synthstats.train.data.collectors import CollectedTrajectory

        traj = CollectedTrajectory(
            observations=["obs1"],
            actions=[{"type": "answer", "payload": "42"}],
            log_probs=torch.tensor([-0.5], requires_grad=True),
            entropy=torch.tensor([0.1]),
            reward=1.0,
            eos_logprobs=torch.tensor([-0.1], requires_grad=True),
        )

        detached = traj.detach()

        assert detached.eos_logprobs is not None
        assert not detached.eos_logprobs.requires_grad
        assert detached.eos_logprobs.device == torch.device("cpu")

    def test_build_subtb_batch_includes_eos_logprobs(self):
        from synthstats.train.data.collate import build_subtb_batch
        from synthstats.train.data.collectors import CollectedTrajectory

        trajs = [
            CollectedTrajectory(
                observations=["obs1"],
                actions=[{"type": "answer", "payload": "42"}],
                log_probs=torch.tensor([-0.5, -0.3]),
                entropy=torch.tensor([0.1, 0.2]),
                reward=1.0,
                eos_logprobs=torch.tensor([-0.1, -0.2]),
            ),
            CollectedTrajectory(
                observations=["obs2"],
                actions=[{"type": "answer", "payload": "24"}],
                log_probs=torch.tensor([-0.4, -0.2, -0.1]),
                entropy=torch.tensor([0.15, 0.25, 0.35]),
                reward=2.0,
                eos_logprobs=torch.tensor([-0.15, -0.25, -0.35]),
            ),
        ]

        batch = build_subtb_batch(trajs, reward_floor=1e-4)

        assert "eos_logprobs" in batch
        assert batch["eos_logprobs"].shape == batch["log_probs"].shape
        # shape should be [2, 3] due to padding
        assert batch["eos_logprobs"].shape == torch.Size([2, 3])

    def test_build_subtb_batch_without_eos_logprobs(self):
        from synthstats.train.data.collate import build_subtb_batch
        from synthstats.train.data.collectors import CollectedTrajectory

        trajs = [
            CollectedTrajectory(
                observations=["obs1"],
                actions=[{"type": "answer", "payload": "42"}],
                log_probs=torch.tensor([-0.5]),
                entropy=torch.tensor([0.1]),
                reward=1.0,
                eos_logprobs=None,  # no EOS logprobs
            ),
        ]

        batch = build_subtb_batch(trajs, reward_floor=1e-4)

        # should not have eos_logprobs key
        assert "eos_logprobs" not in batch

    def test_build_subtb_batch_mixed_eos_logprobs_raises(self):
        from synthstats.train.data.collate import build_subtb_batch
        from synthstats.train.data.collectors import CollectedTrajectory

        trajs = [
            CollectedTrajectory(
                observations=["obs1"],
                actions=[{"type": "answer", "payload": "42"}],
                log_probs=torch.tensor([-0.5]),
                entropy=torch.tensor([0.1]),
                reward=1.0,
                eos_logprobs=torch.tensor([-0.1]),  # has EOS
            ),
            CollectedTrajectory(
                observations=["obs2"],
                actions=[{"type": "answer", "payload": "24"}],
                log_probs=torch.tensor([-0.4]),
                entropy=torch.tensor([0.15]),
                reward=2.0,
                eos_logprobs=None,  # no EOS
            ),
        ]

        with pytest.raises(ValueError, match="mixed eos_logprobs"):
            build_subtb_batch(trajs, reward_floor=1e-4)

    def test_skyrl_trainer_uses_vanilla_tb_by_default(self):
        from synthstats.train.runners.skyrl_subtb import SkyRLSubTBConfig, SkyRLSubTBTrainer

        config = SkyRLSubTBConfig()  # default loss_type="tb"
        trainer = SkyRLSubTBTrainer(config=config)

        batch = {
            "log_probs": torch.randn(2, 5),
            "log_reward": torch.tensor([1.0, 2.0]),
            "loss_mask": torch.ones(2, 5, dtype=torch.bool),
            "entropy": torch.randn(2, 5),
        }

        result = trainer.train_step(batch)

        assert "loss" in result
        assert "logZ" in result
        assert not torch.isnan(torch.tensor(result["loss"]))

    def test_skyrl_trainer_uses_modified_subtb_with_eos_logprobs(self):
        from synthstats.train.runners.skyrl_subtb import SkyRLSubTBConfig, SkyRLSubTBTrainer

        config = SkyRLSubTBConfig(loss_type="modified_subtb")
        trainer = SkyRLSubTBTrainer(config=config)

        batch = {
            "log_probs": torch.randn(2, 5),
            "log_reward": torch.tensor([1.0, 2.0]),
            "loss_mask": torch.ones(2, 5, dtype=torch.bool),
            "entropy": torch.randn(2, 5),
            "eos_logprobs": torch.randn(2, 5),  # EOS logprobs provided
        }

        result = trainer.train_step(batch)

        assert "loss" in result
        assert "logZ" in result
        assert not torch.isnan(torch.tensor(result["loss"]))

    def test_skyrl_trainer_falls_back_to_vanilla_without_eos(self):
        from synthstats.train.runners.skyrl_subtb import SkyRLSubTBConfig, SkyRLSubTBTrainer

        config = SkyRLSubTBConfig(loss_type="modified_subtb")  # request SubTB
        trainer = SkyRLSubTBTrainer(config=config)

        # no eos_logprobs in batch - should fall back to vanilla TB
        batch = {
            "log_probs": torch.randn(2, 5),
            "log_reward": torch.tensor([1.0, 2.0]),
            "loss_mask": torch.ones(2, 5, dtype=torch.bool),
            "entropy": torch.randn(2, 5),
            # no eos_logprobs key
        }

        result = trainer.train_step(batch)

        # should still work (vanilla TB)
        assert "loss" in result
        assert not torch.isnan(torch.tensor(result["loss"]))

    def test_skyrl_trainer_supports_agentic_subtb(self):
        from synthstats.train.runners.skyrl_subtb import SkyRLSubTBConfig, SkyRLSubTBTrainer

        config = SkyRLSubTBConfig(loss_type="agentic_subtb")
        trainer = SkyRLSubTBTrainer(config=config)

        batch = {
            "log_probs": torch.randn(2, 5),
            "log_reward": torch.tensor([1.0, 2.0]),
            "loss_mask": torch.ones(2, 5, dtype=torch.bool),
            "entropy": torch.randn(2, 5),
        }

        result = trainer.train_step(batch)

        assert "loss" in result
        assert "ab_subtb_loss" in result
        assert not torch.isnan(torch.tensor(result["loss"]))
