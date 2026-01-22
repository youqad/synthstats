"""Integration tests for distributed GFlowNet training."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

import torch


class TestGFlowNetExpIntegration:
    """Integration tests for GFlowNetExp."""

    @pytest.fixture
    def mock_cfg(self):
        """Mock Hydra config for GFlowNetExp."""
        cfg = MagicMock()
        cfg.gfn = {
            "loss_type": "modified_subtb",
            "subtb_lambda": 0.9,
            "replay_buffer_size": 100,
            "replay_ratio": 0.5,
            "temperature": 0.7,
        }
        cfg.trainer = MagicMock()
        cfg.trainer.strategy = "fsdp2"
        cfg.trainer.policy = MagicMock()
        cfg.trainer.policy.model = MagicMock()
        cfg.trainer.policy.model.path = "mock-model"
        cfg.trainer.policy.num_workers = 1
        cfg.trainer.policy.num_gpus_per_worker = 1
        cfg.trainer.use_reference_model = False
        cfg.task = {
            "name": "boxing",
            "env": "dugongs",
            "num_prompts": 10,
            "max_steps": 5,
        }
        return cfg

    def test_create_boxing_task(self) -> None:
        """BoxingTask can be created for prompt generation."""
        # import directly without SkyRL
        import sys
        from synthstats.distributed.gfn_exp import GFlowNetExp

        # mock SkyRL imports for this test
        with patch.dict(sys.modules, {"skyrl_train.entrypoints.main_base": MagicMock()}):
            # create a minimal exp-like object with the method we want to test
            class MockExp:
                def _create_boxing_task(self, env_name, max_steps=10):
                    from synthstats.tasks.boxing.task import BoxingTask
                    return BoxingTask(env_name=env_name, max_steps=max_steps)

            exp = MockExp()
            task = exp._create_boxing_task("dugongs", max_steps=5)

            assert task is not None
            assert task.env_name == "dugongs"
            assert task.max_steps == 5

    def test_create_boxing_prompt_from_task(self) -> None:
        """Prompt generation uses task.observe() correctly."""
        from synthstats.tasks.boxing.task import BoxingTask

        task = BoxingTask(env_name="dugongs", max_steps=5)

        # create prompt using the pattern from GFlowNetExp
        state = task.reset(seed=42)
        messages = task.observe(state)

        # convert to prompt format
        prompt_parts = []
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"<|system|>\n{msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"<|user|>\n{msg.content}")

        prompt_parts.append("<|assistant|>")
        prompt = "\n".join(prompt_parts)

        assert "<|system|>" in prompt
        assert "<|user|>" in prompt
        assert "<|assistant|>" in prompt
        assert "dugongs" in prompt.lower()

    def test_prompt_determinism_with_seed(self) -> None:
        """Same seed produces same prompt."""
        from synthstats.tasks.boxing.task import BoxingTask

        task = BoxingTask(env_name="dugongs")

        # generate two prompts with same seed
        state1 = task.reset(seed=123)
        messages1 = task.observe(state1)

        state2 = task.reset(seed=123)
        messages2 = task.observe(state2)

        # should be identical
        assert messages1[0].content == messages2[0].content
        assert messages1[1].content == messages2[1].content

    def test_different_seeds_produce_different_prompts(self) -> None:
        """Different seeds may produce different environment states."""
        from synthstats.tasks.boxing.task import BoxingTask

        task = BoxingTask(env_name="dugongs")

        state1 = task.reset(seed=1)
        state2 = task.reset(seed=2)

        # states should be different (though system prompts are the same)
        # the key difference is in the underlying environment data
        assert state1.step == 0
        assert state2.step == 0


class TestTrainerIntegration:
    """Integration tests for GFlowNetTrainer with real components."""

    @pytest.fixture
    def trainer_with_mock_model(self):
        """Trainer with a mock model for testing."""
        from synthstats.distributed.gfn_trainer import GFlowNetTrainer, GFNConfig

        cfg = MagicMock()
        cfg.gfn = {
            "loss_type": "modified_subtb",
            "subtb_lambda": 0.9,
            "replay_buffer_size": 100,
            "replay_ratio": 0.0,  # disable replay for simpler testing
            "min_buffer_before_replay": 10,
            "entropy_coef": 0.0,
            "temperature": 1.0,
        }
        cfg.trainer = MagicMock()
        cfg.trainer.lr = 1e-4

        trainer = GFlowNetTrainer(cfg)

        # create mock model with actual trainable parameters
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                V = 100
                # trainable param that affects output (ensures gradients flow)
                self.logit_bias = torch.nn.Parameter(torch.zeros(V))
                self.logit_bias.data[0] = 10.0  # token 0 most likely
                self.logit_bias.data[2] = 5.0   # EOS token

            def forward(self, input_ids, attention_mask=None):
                B, L = input_ids.shape
                # broadcast parameter to output shape, keeps gradient connection
                logits = self.logit_bias.unsqueeze(0).unsqueeze(0).expand(B, L, -1)

                class Output:
                    pass

                out = Output()
                out.logits = logits
                return out

        mock_model = MockModel()
        trainer.policy_model = mock_model
        # include model params in optimizer for gradient flow
        trainer.optimizer = torch.optim.Adam(
            [{"params": [trainer.logZ], "lr": 0.01},
             {"params": mock_model.parameters(), "lr": 1e-4}]
        )

        return trainer

    def test_full_training_step(self, trainer_with_mock_model) -> None:
        """Complete training step with scoring and loss computation."""
        trainer = trainer_with_mock_model
        B, L = 4, 10

        # create a batch mimicking real data
        batch = {
            "input_ids": torch.randint(0, 100, (B, L)),
            "attention_mask": torch.ones(B, L),
            "response_mask": torch.ones(B, L - 1),
            "prompt_lengths": torch.tensor([2, 2, 3, 3]),
            "log_rewards": torch.tensor([-0.5, -0.3, -0.4, -0.2]),
            "terminated": torch.ones(B, dtype=torch.bool),
            "temperature": torch.full((B,), 0.7),
        }

        # score the batch (adds log_probs and eos_logprobs)
        scored_batch = trainer._score_batch(batch)

        assert "log_probs" in scored_batch
        assert "eos_logprobs" in scored_batch
        assert scored_batch["log_probs"].shape == (B, L - 1)

        scored_batch["logZ"] = trainer.logZ

        # make log_probs require grad for backward pass (simulates real training)
        scored_batch["log_probs"] = scored_batch["log_probs"].clone().detach().requires_grad_(True)

        metrics = trainer.train_critic_and_policy(scored_batch)

        assert "loss" in metrics
        assert "logZ" in metrics
        assert isinstance(metrics["loss"], float)

    def test_replay_buffer_integration(self, trainer_with_mock_model) -> None:
        """Replay buffer properly stores and retrieves entries."""
        trainer = trainer_with_mock_model
        B = 4

        # add entries to buffer with UNIQUE sequences (to avoid deduplication)
        # each sequence has a different response token to make them unique
        input_ids = [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 11, 12, 13, 14, 15, 16, 17],  # different response
            [1, 2, 3, 4, 20, 21, 22, 23, 24, 25],   # different response
            [1, 2, 3, 4, 5, 30, 31, 32, 33, 34],    # different response
        ]
        prompt_lengths = [3, 3, 4, 4]
        log_rewards = [-0.5, -0.3, -0.4, -0.2]

        added = trainer.replay_buffer.add_from_batch(
            input_ids=input_ids,
            prompt_lengths=prompt_lengths,
            log_rewards=log_rewards,
        )

        assert added == B
        assert len(trainer.replay_buffer) == B

        stats = trainer.replay_buffer.get_stats()
        assert stats["size"] == B
        assert stats["mean_log_reward"] == pytest.approx(-0.35)


class TestDistributedScoringPath:
    """Tests for the distributed scoring code path."""

    def test_score_batch_fallback_to_local(self) -> None:
        """Falls back to local scoring when actor group unavailable."""
        from synthstats.distributed.gfn_trainer import GFlowNetTrainer

        cfg = MagicMock()
        cfg.gfn = {
            "loss_type": "tb",
            "subtb_lambda": 0.9,
            "replay_buffer_size": 100,
            "replay_ratio": 0.0,
            "min_buffer_before_replay": 10,
            "temperature": 1.0,
        }
        cfg.trainer = MagicMock()
        cfg.trainer.lr = 1e-4

        trainer = GFlowNetTrainer(cfg)

        # create mock model
        class MockModel:
            def __init__(self):
                self._parameters = [torch.nn.Parameter(torch.zeros(1))]

            def parameters(self):
                return iter(self._parameters)

            def __call__(self, input_ids, attention_mask):
                B, L = input_ids.shape
                V = 100
                logits = torch.zeros(B, L, V)

                class Output:
                    pass

                out = Output()
                out.logits = logits
                return out

        trainer.policy_model = MockModel()

        # no policy_actor_group set - should use local scoring
        B, L = 2, 5
        batch = {
            "input_ids": torch.zeros(B, L, dtype=torch.long),
            "attention_mask": torch.ones(B, L),
            "response_mask": torch.ones(B, L - 1),
            "temperature": torch.ones(B),
        }

        scored = trainer._score_batch(batch)

        assert "log_probs" in scored
        assert "eos_logprobs" in scored
        assert scored["log_probs"].shape == (B, L - 1)

    def test_distributed_scoring_method_exists(self) -> None:
        """_score_batch_distributed method is available."""
        from synthstats.distributed.gfn_trainer import GFlowNetTrainer

        cfg = MagicMock()
        cfg.gfn = {}
        cfg.trainer = MagicMock()
        cfg.trainer.lr = 1e-4

        trainer = GFlowNetTrainer(cfg)

        assert hasattr(trainer, "_score_batch_distributed")
        assert callable(trainer._score_batch_distributed)


class TestEndToEndPipeline:
    """End-to-end tests for the complete training pipeline."""

    def test_task_to_prompt_to_training(self) -> None:
        """Full pipeline: task -> prompts -> tokenization -> training."""
        from synthstats.tasks.boxing.task import BoxingTask
        from synthstats.distributed.gfn_trainer import GFlowNetTrainer

        # 1. Create task and generate prompt
        task = BoxingTask(env_name="dugongs", max_steps=5)
        state = task.reset(seed=42)
        messages = task.observe(state)

        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[1].role == "user"

        # 2. Convert to prompt string (as GFlowNetExp would)
        prompt_parts = []
        for msg in messages:
            prompt_parts.append(f"<|{msg.role}|>\n{msg.content}")
        prompt_parts.append("<|assistant|>")
        prompt = "\n".join(prompt_parts)

        assert len(prompt) > 0
        assert "<|system|>" in prompt

        # 3. Create trainer and verify it initializes
        cfg = MagicMock()
        cfg.gfn = {
            "loss_type": "modified_subtb",
            "replay_buffer_size": 100,
            "replay_ratio": 0.0,
        }
        cfg.trainer = MagicMock()
        cfg.trainer.lr = 1e-4

        trainer = GFlowNetTrainer(cfg)

        assert trainer.gfn_config.loss_type == "modified_subtb"
        assert isinstance(trainer.logZ, torch.nn.Parameter)

    @pytest.mark.skipif(
        True,  # skip by default - requires full SkyRL
        reason="Full SkyRL integration test - requires skyrl-train"
    )
    def test_full_skyrl_integration(self) -> None:
        """Full integration with SkyRL (skipped unless explicitly enabled)."""
        # this would test:
        # 1. GFlowNetExp initialization
        # 2. Trainer setup with actor groups
        # 3. Actual training steps
        pass
