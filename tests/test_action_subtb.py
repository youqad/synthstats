"""Tests for action-boundary SubTB components."""

import torch

from synthstats.train.objectives.subtb import SubTBConfig, SubTBObjective


class TestActionSubTBCore:
    def test_zero_loss_when_flow_is_consistent(self):
        from synthstats.train.objectives.losses import compute_action_subtb_core

        # u is constructed so that each one-step delta is exactly zero:
        # u[i+1] = u[i] + log_pf[i]
        log_pf = torch.tensor([[-0.2, -0.3, -0.5]], dtype=torch.float32)
        u = torch.tensor([[0.0, -0.2, -0.5, -1.0]], dtype=torch.float32)
        mask = torch.tensor([[True, True, True]])

        loss = compute_action_subtb_core(log_pf=log_pf, u=u, mask=mask, subtb_lambda=0.9)

        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_respects_masked_transitions(self):
        from synthstats.train.objectives.losses import compute_action_subtb_core

        log_pf = torch.tensor([[-0.2, -0.3, -0.5]], dtype=torch.float32)
        u = torch.tensor([[0.0, -0.2, -0.5, -1.0]], dtype=torch.float32)
        mask = torch.tensor([[True, False, True]])

        loss = compute_action_subtb_core(log_pf=log_pf, u=u, mask=mask, subtb_lambda=0.9)

        # should remain finite and non-negative when some transitions are excluded
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0

    def test_zero_action_length_is_graph_connected_zero(self):
        from synthstats.train.objectives.losses import compute_action_subtb_core

        log_pf = torch.zeros(2, 0, requires_grad=True)
        u = torch.zeros(2, 1, requires_grad=True)

        loss = compute_action_subtb_core(log_pf=log_pf, u=u, mask=None, subtb_lambda=0.9)
        assert torch.isfinite(loss)
        assert loss.requires_grad

        loss.backward()
        assert u.grad is not None
        assert log_pf.grad is not None

    def test_lambda_extremes_are_stable(self):
        from synthstats.train.objectives.losses import compute_action_subtb_core

        log_pf = torch.tensor([[-0.2, -0.3, -0.4]], dtype=torch.float32)
        u = torch.tensor([[0.0, -0.15, -0.6, -1.1]], dtype=torch.float32)
        mask = torch.tensor([[True, True, True]])

        loss_l0 = compute_action_subtb_core(log_pf=log_pf, u=u, mask=mask, subtb_lambda=0.0)
        loss_l1 = compute_action_subtb_core(log_pf=log_pf, u=u, mask=mask, subtb_lambda=1.0)

        assert torch.isfinite(loss_l0)
        assert torch.isfinite(loss_l1)
        assert loss_l0.item() >= 0.0
        assert loss_l1.item() >= 0.0

    def test_all_transitions_masked_returns_finite_zero_like_loss(self):
        from synthstats.train.objectives.losses import compute_action_subtb_core

        log_pf = torch.tensor([[-0.2, -0.3, -0.4]], dtype=torch.float32, requires_grad=True)
        u = torch.tensor([[0.0, -0.2, -0.5, -1.0]], dtype=torch.float32, requires_grad=True)
        mask = torch.tensor([[False, False, False]])

        loss = compute_action_subtb_core(log_pf=log_pf, u=u, mask=mask, subtb_lambda=0.9)
        assert torch.isfinite(loss)
        assert loss.item() == 0.0

        loss.backward()
        assert log_pf.grad is not None
        assert u.grad is not None


class TestSubTBObjectiveABSubTB:
    def _build_batch(self) -> dict[str, torch.Tensor]:
        return {
            "log_probs": torch.tensor(
                [
                    [-0.2, -0.3, -0.4],
                    [-0.1, -0.4, 0.0],
                ],
                dtype=torch.float32,
            ),
            "loss_mask": torch.tensor(
                [
                    [True, True, True],
                    [True, True, False],
                ]
            ),
            "log_reward": torch.tensor([0.8, 0.4], dtype=torch.float32),
            "entropy": torch.zeros(2, 3, dtype=torch.float32),
        }

    def test_ab_subtb_loss_reports_metrics(self):
        objective = SubTBObjective(
            config=SubTBConfig(
                loss_type="ab_subtb",
                ab_subtb_alpha=0.25,
                use_boundary_critic=False,
                entropy_coef=0.0,
            ),
            device="cpu",
        )

        loss, metrics = objective(self._build_batch())

        assert torch.isfinite(loss)
        assert "ab_subtb_loss" in metrics
        assert "tb_loss" in metrics
        assert metrics["ab_subtb_loss"] >= 0.0

    def test_ab_subtb_with_critic_has_gradients(self):
        objective = SubTBObjective(
            config=SubTBConfig(
                loss_type="ab_subtb",
                ab_subtb_alpha=0.25,
                use_boundary_critic=True,
                boundary_critic_hidden_dim=16,
                boundary_critic_loss_coef=0.5,
                entropy_coef=0.0,
            ),
            device="cpu",
        )

        loss, metrics = objective(self._build_batch())
        loss.backward()

        assert torch.isfinite(loss)
        assert "ab_critic_loss" in metrics
        critic_grads = [
            p.grad
            for n, p in objective.named_parameters()
            if "boundary_critic" in n and p.grad is not None
        ]
        assert critic_grads, "boundary critic parameters should receive gradients"
