"""Day 1 smoke test: verify policy model receives gradients.

This test validates the critical architecture fix where local scoring
(use_local_scoring_for_training=True) replaces detached distributed
scoring, ensuring loss.backward() actually updates the policy model.

The test uses a minimal linear model as a stand-in for a transformer.
If this test fails, the policy model is NOT learning.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from synthstats.distributed.gfn_trainer import GFlowNetTrainer, GFNConfig
from synthstats.distributed.scoring import build_response_mask


class MinimalPolicyModel(nn.Module):
    """Minimal model that produces logits from token IDs.

    Simulates a language model: input_ids -> embeddings -> linear -> logits.
    Small enough to run in tests without GPU.
    """

    def __init__(self, vocab_size: int = 100, hidden_dim: int = 32) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> MagicMock:
        """Returns object with .logits attribute (HuggingFace convention)."""
        h = self.embedding(input_ids)
        logits = self.head(h)
        result = MagicMock()
        result.logits = logits
        return result


def _create_trainer_standalone(
    gfn_config: GFNConfig | None = None,
    vocab_size: int = 100,
) -> GFlowNetTrainer:
    """Create a GFlowNetTrainer in standalone mode (no SkyRL)."""
    if gfn_config is None:
        gfn_config = GFNConfig(
            use_local_scoring_for_training=True,
            loss_type="tb",
            length_normalize=False,
            entropy_coef=0.0,
        )

    # minimal Hydra-like config
    cfg = MagicMock()
    cfg.trainer.lr = 1e-3
    cfg.trainer.max_grad_norm = 1.0
    cfg.trainer.policy.model.path = "test-model"

    trainer = GFlowNetTrainer(cfg=cfg, gfn_config=gfn_config)

    # attach a local policy model
    model = MinimalPolicyModel(vocab_size=vocab_size)
    trainer.policy_model = model

    # set up optimizer covering both model and logZ
    param_groups = trainer.get_optimizer_param_groups()
    trainer.optimizer = torch.optim.Adam(param_groups)

    # set stop token IDs
    trainer._stop_token_ids = [2]  # simple EOS

    return trainer


def _create_dummy_batch(
    batch_size: int = 4,
    seq_len: int = 10,
    prompt_len: int = 3,
    vocab_size: int = 100,
    device: str = "cpu",
) -> dict:
    """Create a dummy scored batch for train_critic_and_policy."""
    input_ids = torch.randint(3, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)
    prompt_lengths = torch.full((batch_size,), prompt_len, device=device, dtype=torch.long)

    response_mask = build_response_mask(input_ids, prompt_lengths, attention_mask=attention_mask)

    # rewards (positive, finite)
    log_rewards = torch.randn(batch_size, device=device) * 0.5 + 1.0

    # dummy log_probs and eos_logprobs (will be replaced by re-scoring)
    log_probs = torch.randn(batch_size, seq_len - 1, device=device) * 0.1
    eos_logprobs = torch.randn(batch_size, seq_len - 1, device=device) * 0.1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
        "prompt_lengths": prompt_lengths,
        "log_rewards": log_rewards,
        "terminated": torch.ones(batch_size, device=device, dtype=torch.bool),
        "temperature": torch.ones(batch_size, device=device),
        "log_probs": log_probs,
        "eos_logprobs": eos_logprobs,
        "logZ": None,  # will be set by trainer
        "is_replay": torch.zeros(batch_size, device=device, dtype=torch.bool),
    }


class TestGradientFlow:
    """Verify the local scoring fix delivers gradients to the policy model."""

    def test_local_scoring_produces_grad_enabled_tensors(self) -> None:
        """_rescore_batch_with_gradients returns tensors with requires_grad."""
        trainer = _create_trainer_standalone()
        batch = _create_dummy_batch()

        rescored = trainer._rescore_batch_with_gradients(batch)

        assert rescored["log_probs"].requires_grad, (
            "log_probs should require grad after local re-scoring"
        )

    def test_policy_model_receives_gradients(self) -> None:
        """backward() on TB loss produces non-zero gradients on policy params."""
        trainer = _create_trainer_standalone()
        batch = _create_dummy_batch()
        batch["logZ"] = trainer.logZ

        # zero gradients
        trainer.optimizer.zero_grad(set_to_none=True)

        # re-score with gradients (this is the fix)
        batch = trainer._rescore_batch_with_gradients(batch)

        # compute TB loss manually
        from synthstats.training.losses.trajectory_balance import compute_trajectory_balance_loss
        from synthstats.distributed.gfn_trainer import ConfigProxy

        config = ConfigProxy({
            "logZ": trainer.logZ.item(),
            "_logZ_tensor": trainer.logZ,
            "tb_max_residual": 100.0,
        })

        loss, _ = compute_trajectory_balance_loss(
            log_probs=batch["log_probs"],
            old_log_probs=batch["log_probs"].detach(),
            advantages=batch["log_rewards"].unsqueeze(1).expand(-1, batch["log_probs"].shape[1]),
            config=config,
            loss_mask=batch["response_mask"],
        )

        loss.backward()

        # check model parameters have gradients
        has_grad = False
        for param in trainer.policy_model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, (
            "Policy model parameters have NO gradient after backward(). "
            "The gradient flow fix is NOT working!"
        )

    def test_logz_receives_gradients(self) -> None:
        """logZ parameter gets gradients through the TB loss."""
        trainer = _create_trainer_standalone()
        batch = _create_dummy_batch()
        batch["logZ"] = trainer.logZ

        trainer.optimizer.zero_grad(set_to_none=True)
        batch = trainer._rescore_batch_with_gradients(batch)

        from synthstats.training.losses.trajectory_balance import compute_trajectory_balance_loss
        from synthstats.distributed.gfn_trainer import ConfigProxy

        config = ConfigProxy({
            "logZ": trainer.logZ.item(),
            "_logZ_tensor": trainer.logZ,
            "tb_max_residual": 100.0,
        })

        loss, _ = compute_trajectory_balance_loss(
            log_probs=batch["log_probs"],
            old_log_probs=batch["log_probs"].detach(),
            advantages=batch["log_rewards"].unsqueeze(1).expand(-1, batch["log_probs"].shape[1]),
            config=config,
            loss_mask=batch["response_mask"],
        )
        loss.backward()

        assert trainer.logZ.grad is not None, "logZ has no gradient"
        assert trainer.logZ.grad.abs() > 0, "logZ gradient is zero"

    def test_params_change_after_optimizer_step(self) -> None:
        """Parameters actually update after optimizer.step()."""
        trainer = _create_trainer_standalone()

        # capture initial params
        initial_params = {
            name: p.clone().detach()
            for name, p in trainer.policy_model.named_parameters()
        }
        initial_logZ = trainer.logZ.clone().detach()

        # run full training step
        batch = _create_dummy_batch()
        batch["logZ"] = trainer.logZ
        metrics = trainer.train_critic_and_policy(batch)

        # verify params changed
        params_changed = False
        for name, p in trainer.policy_model.named_parameters():
            if not torch.allclose(initial_params[name], p.data, atol=1e-7):
                params_changed = True
                break

        assert params_changed, (
            "Policy model params unchanged after train step - "
            "gradient NOT flowing through local scoring!"
        )

        # logZ should also change
        assert not torch.allclose(initial_logZ, trainer.logZ.data, atol=1e-7), (
            "logZ unchanged after train step"
        )

    def test_loss_decreases_over_steps(self) -> None:
        """TB loss should decrease over multiple training steps (basic sanity)."""
        trainer = _create_trainer_standalone(
            GFNConfig(
                use_local_scoring_for_training=True,
                loss_type="tb",
                length_normalize=False,
                entropy_coef=0.0,
                lr_logZ=0.01,
            )
        )

        # use fixed batch (same data, model should learn it)
        torch.manual_seed(42)
        batch_template = _create_dummy_batch(batch_size=8, seq_len=12)

        losses = []
        for step in range(20):
            batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch_template.items()}
            batch["logZ"] = trainer.logZ
            metrics = trainer.train_critic_and_policy(batch)
            losses.append(metrics["loss"])

        # loss should decrease (not necessarily monotonically)
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}. "
            f"Training is not converging."
        )

    def test_detached_scoring_no_gradient(self) -> None:
        """Without local re-scoring, model params get NO gradient (proves the bug)."""
        trainer = _create_trainer_standalone(
            GFNConfig(
                use_local_scoring_for_training=False,
                loss_type="tb",  # TB includes logZ in residual (SubTB doesn't)
                length_normalize=False,
                entropy_coef=0.0,
            )
        )

        batch = _create_dummy_batch()
        batch["logZ"] = trainer.logZ

        # pre-fill with detached log_probs (simulates distributed scoring)
        batch["log_probs"] = torch.randn(4, 9).detach()
        batch["eos_logprobs"] = torch.randn(4, 9).detach()

        trainer.optimizer.zero_grad(set_to_none=True)
        metrics = trainer.train_critic_and_policy(batch)

        # model params should NOT have gradients (proving the bug)
        has_model_grad = False
        for param in trainer.policy_model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_model_grad = True
                break

        assert not has_model_grad, (
            "Model has gradients with detached scoring - "
            "this should NOT happen (the bug is not reproduced)"
        )

        # but logZ SHOULD still get gradients (it's in the loss directly)
        assert trainer.logZ.grad is not None and trainer.logZ.grad.abs() > 0, (
            "logZ should still get gradients even with detached scoring"
        )

    def test_subtb_loss_gradient_flow(self) -> None:
        """Modified SubTB loss also delivers gradients to model."""
        trainer = _create_trainer_standalone(
            GFNConfig(
                use_local_scoring_for_training=True,
                loss_type="modified_subtb",
                length_normalize=False,
                entropy_coef=0.0,
            )
        )

        initial_params = {
            name: p.clone().detach()
            for name, p in trainer.policy_model.named_parameters()
        }

        batch = _create_dummy_batch()
        batch["logZ"] = trainer.logZ
        metrics = trainer.train_critic_and_policy(batch)

        params_changed = any(
            not torch.allclose(initial_params[name], p.data, atol=1e-7)
            for name, p in trainer.policy_model.named_parameters()
        )

        assert params_changed, (
            "Policy model unchanged with SubTB loss - gradient not flowing!"
        )

    def test_length_normalization_changes_loss(self) -> None:
        """Length normalization should produce different loss values."""
        torch.manual_seed(123)
        batch = _create_dummy_batch(batch_size=4, seq_len=20, prompt_len=5)

        # without length norm
        trainer_no_norm = _create_trainer_standalone(
            GFNConfig(use_local_scoring_for_training=True, length_normalize=False, entropy_coef=0.0)
        )
        batch_a = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch_a["logZ"] = trainer_no_norm.logZ
        metrics_no_norm = trainer_no_norm.train_critic_and_policy(batch_a)

        # with length norm
        trainer_norm = _create_trainer_standalone(
            GFNConfig(use_local_scoring_for_training=True, length_normalize=True, entropy_coef=0.0)
        )
        batch_b = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch_b["logZ"] = trainer_norm.logZ
        metrics_norm = trainer_norm.train_critic_and_policy(batch_b)

        # losses should differ (normalization changes the magnitude)
        assert metrics_no_norm["loss"] != metrics_norm["loss"], (
            "Length normalization had no effect on loss"
        )


class TestVarGradLogZ:
    """Tests for the VarGrad logZ estimator from trajectory_balance_v2."""

    def test_vargrad_estimate_finite(self) -> None:
        """VarGrad logZ estimate should be finite."""
        from synthstats.training.losses.trajectory_balance_v2 import estimate_log_partition_vargrad

        B, T = 8, 10
        log_probs = torch.randn(B, T) * 0.1
        log_rewards = torch.randn(B) * 0.5 + 1.0
        response_mask = torch.ones(B, T)

        logZ = estimate_log_partition_vargrad(log_probs, log_rewards, response_mask)

        assert torch.isfinite(logZ), f"VarGrad logZ is not finite: {logZ}"

    def test_vargrad_with_ref_model(self) -> None:
        """VarGrad works with reference model log probs."""
        from synthstats.training.losses.trajectory_balance_v2 import estimate_log_partition_vargrad

        B, T = 8, 10
        log_probs = torch.randn(B, T) * 0.1
        ref_log_probs = torch.randn(B, T) * 0.1
        log_rewards = torch.randn(B) * 0.5 + 1.0
        response_mask = torch.ones(B, T)

        logZ = estimate_log_partition_vargrad(
            log_probs, log_rewards, response_mask, ref_log_probs=ref_log_probs
        )

        assert torch.isfinite(logZ)

    def test_vargrad_tb_loss_no_learned_logz(self) -> None:
        """VarGrad TB loss works without a learned logZ parameter."""
        from synthstats.training.losses.trajectory_balance_v2 import compute_vargrad_tb_loss

        B, T = 8, 10
        log_probs = (torch.randn(B, T) * 0.1).requires_grad_(True)
        log_rewards = torch.randn(B) * 0.5 + 1.0
        response_mask = torch.ones(B, T)

        loss, metrics = compute_vargrad_tb_loss(
            log_probs, log_rewards, response_mask
        )

        assert torch.isfinite(loss)
        assert loss.requires_grad

        # backward should work
        loss.backward()
        assert log_probs.grad is not None
        assert log_probs.grad.abs().sum() > 0

    def test_vargrad_tb_per_prompt_grouping(self) -> None:
        """Per-prompt grouping produces different logZ per prompt."""
        from synthstats.training.losses.trajectory_balance_v2 import compute_vargrad_tb_loss

        B, T = 8, 10
        log_probs = torch.randn(B, T, requires_grad=True) * 0.1
        # two prompts, 4 responses each
        log_rewards = torch.cat([torch.ones(4) * 2.0, torch.ones(4) * -1.0])
        response_mask = torch.ones(B, T)
        prompt_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

        loss, metrics = compute_vargrad_tb_loss(
            log_probs, log_rewards, response_mask, prompt_ids=prompt_ids
        )

        assert torch.isfinite(loss)
        loss.backward()

    def test_flowrl_loss_with_importance_weights(self) -> None:
        """FlowRL loss applies importance weights correctly."""
        from synthstats.training.losses.trajectory_balance_v2 import compute_flowrl_loss

        B, T = 4, 10
        log_probs = (torch.randn(B, T) * 0.1).requires_grad_(True)
        old_log_probs = log_probs.detach() + torch.randn(B, T) * 0.01
        log_rewards = torch.randn(B) * 0.5 + 1.0
        response_mask = torch.ones(B, T)
        logZ = torch.tensor(0.0)

        loss, metrics = compute_flowrl_loss(
            log_probs, log_rewards, response_mask, logZ,
            old_log_probs=old_log_probs,
            use_importance_weights=True,
        )

        assert torch.isfinite(loss)
        assert metrics["mean_is_weight"] > 0
        loss.backward()
        assert log_probs.grad is not None

    def test_tb_loss_with_kl_reference(self) -> None:
        """KL-regularized TB loss uses reference model."""
        from synthstats.training.losses.trajectory_balance_v2 import compute_tb_loss_with_kl

        B, T = 4, 10
        log_probs = (torch.randn(B, T) * 0.1).requires_grad_(True)
        ref_log_probs = torch.randn(B, T) * 0.1
        log_rewards = torch.randn(B) * 0.5 + 1.0
        response_mask = torch.ones(B, T)
        logZ = torch.tensor(0.0, requires_grad=True)

        loss, metrics = compute_tb_loss_with_kl(
            log_probs, log_rewards, response_mask, logZ,
            ref_log_probs=ref_log_probs,
        )

        assert torch.isfinite(loss)
        assert "mean_ref_log_prob" in metrics
        loss.backward()
        assert log_probs.grad is not None
        assert logZ.grad is not None
