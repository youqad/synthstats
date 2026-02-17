import pytest
import torch
import torch.nn as nn

from synthstats.train.learners.subtb_torch import (
    SubTBTorchConfig,
    create_subtb_learner,
)


def _make_batch(batch_size: int = 2, seq_len: int = 4, device: str = "cpu") -> dict:
    return {
        "log_probs": torch.randn(batch_size, seq_len, device=device, requires_grad=True),
        "log_reward": torch.randn(batch_size, device=device),
        "loss_mask": torch.ones(batch_size, seq_len, device=device, dtype=torch.bool),
        "entropy": torch.randn(batch_size, device=device).abs(),
    }


class TestSubTBTorchLearnerInit:
    def test_optimizer_has_logz_and_policy_groups(self):
        policy = nn.Linear(4, 4)
        learner = create_subtb_learner(policy=policy)
        groups = learner.optimizer.param_groups
        assert len(groups) == 2
        assert groups[0]["lr"] == learner.config.logZ_lr
        assert groups[0]["weight_decay"] == 0.0

    def test_optimizer_without_policy(self):
        learner = create_subtb_learner(policy=None)
        assert len(learner.optimizer.param_groups) == 1

    def test_amp_scaler_created_when_enabled(self):
        config = SubTBTorchConfig(amp_enabled=True)
        learner = create_subtb_learner(learner_config=config)
        assert learner._scaler is not None

    def test_amp_scaler_none_when_disabled(self):
        learner = create_subtb_learner()
        assert learner._scaler is None


class TestSubTBTorchLearnerUpdate:
    def test_loss_decreases_over_steps(self):
        policy = nn.Linear(4, 4)
        learner = create_subtb_learner(policy=policy)

        losses = []
        for _ in range(20):
            batch = _make_batch()
            x = torch.randn(2, 4)
            out = policy(x)
            batch["log_probs"] = out
            batch["loss_mask"] = torch.ones(2, 4, dtype=torch.bool)
            metrics = learner.update(batch)
            losses.append(metrics["loss"])

        assert losses[0] > losses[-1] or losses[-1] < 1.0

    def test_gradient_flows_to_logz(self):
        learner = create_subtb_learner()
        initial_logz = learner.logZ

        batch = _make_batch()
        learner.update(batch)

        assert learner.logZ != pytest.approx(initial_logz, abs=1e-8)

    def test_gradient_flows_to_policy(self):
        policy = nn.Linear(4, 4)
        initial_weight = policy.weight.clone()
        learner = create_subtb_learner(policy=policy)

        x = torch.randn(2, 4)
        batch = _make_batch()
        batch["log_probs"] = policy(x)
        learner.update(batch)

        assert not torch.allclose(policy.weight, initial_weight)


class TestGradientClipping:
    def test_gradients_clipped_to_max_norm(self):
        max_norm = 0.5
        config = SubTBTorchConfig(max_grad_norm=max_norm)
        policy = nn.Linear(4, 4)
        learner = create_subtb_learner(policy=policy, learner_config=config)

        batch = _make_batch()
        batch["log_reward"] = torch.tensor([100.0, -100.0])
        x = torch.randn(2, 4)
        batch["log_probs"] = policy(x)

        # replicate update() internals to inspect grad norm before step
        learner.optimizer.zero_grad(set_to_none=True)
        loss, _ = learner.objective(batch)
        loss.backward()
        learner._clip_gradients()

        total_norm = 0.0
        for group in learner.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    total_norm += p.grad.norm().item() ** 2
        total_norm = total_norm**0.5
        assert total_norm <= max_norm + 0.01  # small tolerance


class TestStateDictRoundtrip:
    def test_save_load_preserves_logz(self):
        learner = create_subtb_learner()
        batch = _make_batch()
        learner.update(batch)
        expected_logz = learner.logZ

        state = learner.state_dict()

        learner2 = create_subtb_learner()
        learner2.load_state_dict(state)

        assert learner2.logZ == pytest.approx(expected_logz, abs=1e-6)

    def test_state_dict_contains_expected_keys(self):
        learner = create_subtb_learner()
        state = learner.state_dict()
        assert "objective" in state
        assert "optimizer" in state
