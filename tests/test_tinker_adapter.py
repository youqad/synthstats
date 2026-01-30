"""Tests for Tinker API adapter.

Tests the Tinker integration layer using mock clients (no API access needed).
"""

import pytest
import torch
import torch.nn as nn

# skip all tests in this module if Tinker SDK is not installed
try:
    import tinker  # noqa: F401

    TINKER_AVAILABLE = True
except ImportError:
    TINKER_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not TINKER_AVAILABLE,
    reason="Tinker SDK not installed",
)

# --- Shared Test Fixtures ---


class CharLevelMockTokenizer:
    """Character-level mock tokenizer for testing turn mask logic."""

    def encode_plus(self, text, **kwargs):
        """Return offset_mapping treating each character as a token."""
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}

    def encode(self, text, add_special_tokens=True):
        """Return word-level token IDs (for prompt offset calculation)."""
        return list(range(len(text.split())))


@pytest.fixture
def mock_tokenizer():
    """Provide a character-level mock tokenizer for mask tests."""
    return CharLevelMockTokenizer()


# --- Test Classes ---


class TestImportSafety:
    """Test that tinker module is import-safe without Tinker SDK."""

    def test_import_without_tinker(self):
        """Module should import even without tinker SDK."""
        from synthstats.integrations import tinker

        assert hasattr(tinker, "is_tinker_available")
        assert hasattr(tinker, "TinkerPolicy")
        assert hasattr(tinker, "TinkerTrainer")

    def test_is_tinker_available(self):
        """is_tinker_available returns bool without crashing."""
        from synthstats.integrations.tinker import is_tinker_available

        result = is_tinker_available()
        assert isinstance(result, bool)


class TestTinkerConfig:
    """Test TinkerConfig dataclass."""

    def test_default_config(self):
        """Config with defaults should work."""
        from synthstats.integrations.tinker import TinkerConfig

        config = TinkerConfig()
        assert config.model == "Qwen/Qwen3-4B"
        assert config.max_tokens == 256
        assert config.temperature == 0.7
        assert config.lora_rank == 32

    def test_custom_config(self):
        """Config with custom values should work."""
        from synthstats.integrations.tinker import TinkerConfig

        config = TinkerConfig(
            model="meta-llama/Llama-3.1-8B",
            api_key="test-key",
            max_tokens=512,
            temperature=0.9,
        )
        assert config.model == "meta-llama/Llama-3.1-8B"
        assert config.get_api_key() == "test-key"

    def test_get_api_key_from_env(self, monkeypatch):
        """Config should read API key from environment."""
        from synthstats.integrations.tinker import TinkerConfig

        monkeypatch.setenv("TINKER_API_KEY", "env-key-123")
        config = TinkerConfig()
        assert config.get_api_key() == "env-key-123"

    def test_get_api_key_missing_raises(self, monkeypatch):
        """Config should raise if no API key available."""
        from synthstats.integrations.tinker import TinkerConfig

        monkeypatch.delenv("TINKER_API_KEY", raising=False)
        config = TinkerConfig()
        with pytest.raises(ValueError, match="API key required"):
            config.get_api_key()


class TestMockTinkerClient:
    """Test MockTinkerClient for testing without API."""

    def test_mock_sample(self):
        """Mock client sample() returns valid structure."""
        from synthstats.integrations.tinker import MockTinkerClient

        client = MockTinkerClient()
        result = client.sample(prompt="test prompt")

        assert hasattr(result, "text")
        assert hasattr(result, "logprobs")
        assert isinstance(result.text, str)
        assert isinstance(result.logprobs, list)
        assert all(isinstance(lp, float) for lp in result.logprobs)

    def test_mock_logprobs(self):
        """Mock client logprobs() returns valid structure."""
        from synthstats.integrations.tinker import MockTinkerClient

        client = MockTinkerClient()
        result = client.logprobs(prompt="test", completion="the answer is 42")

        assert hasattr(result, "logprobs")
        assert isinstance(result.logprobs, list)


class TestTinkerPolicyWithMock:
    """Test TinkerPolicy using mock client."""

    @pytest.fixture
    def mock_policy(self, monkeypatch):
        """Create policy with mock sampling client injected."""
        from dataclasses import dataclass, field

        from synthstats.integrations import tinker
        from synthstats.integrations.tinker import (
            TinkerConfig,
            TinkerPolicy,
        )

        # patch is_tinker_available to return False for mock mode
        monkeypatch.setattr(tinker, "is_tinker_available", lambda: False)

        # create mock sampling client that matches Tinker's interface
        @dataclass
        class MockSampledSequence:
            tokens: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
            logprobs: list[float] = field(default_factory=lambda: [-0.1, -0.2, -0.15, -0.1, -0.2])

        @dataclass
        class MockSampleResult:
            sequences: list[MockSampledSequence] = field(
                default_factory=lambda: [MockSampledSequence()]
            )

        @dataclass
        class MockLogprobsResult:
            logprobs: list[float] = field(default_factory=lambda: [-0.1] * 20)

        class MockTokenizer:
            def encode(self, text):
                return list(range(len(text.split())))

            def decode(self, tokens, skip_special_tokens=False):
                return '{"type": "answer", "payload": "42"}'

        class MockSamplingClient:
            def sample(self, prompt, num_samples, sampling_params):
                return MockSampleResult()

            def compute_logprobs(self, model_input):
                return MockLogprobsResult()

            def get_tokenizer(self):
                return MockTokenizer()

        config = TinkerConfig(api_key="test-key")
        policy = TinkerPolicy(config=config)

        # inject mocks directly into private attributes
        policy._sampling_client = MockSamplingClient()
        policy._tokenizer = MockTokenizer()

        return policy

    def test_call_returns_action_logp_entropy(self, mock_policy):
        """Policy call should return (action, logp, entropy) tuple."""
        action, logp, entropy = mock_policy("What is 2+2?")

        assert isinstance(action, dict)
        assert "type" in action or "payload" in action
        assert isinstance(logp, float)
        assert isinstance(entropy, float)

    def test_call_with_temperature(self, mock_policy):
        """Policy should accept temperature parameter."""
        action, logp, entropy = mock_policy("test", temperature=0.5)
        assert isinstance(action, dict)

    def test_score_action(self, mock_policy):
        """score_action should return tensor tuple."""
        action = {"type": "answer", "payload": "42"}
        logp, entropy = mock_policy.score_action("What is 6*7?", action)

        assert isinstance(logp, torch.Tensor)
        assert isinstance(entropy, torch.Tensor)

    def test_require_grad_logp_raises(self):
        """require_grad_logp=True should raise NotImplementedError."""
        from synthstats.integrations.tinker import TinkerConfig, TinkerPolicy

        config = TinkerConfig(api_key="test")
        with pytest.raises(NotImplementedError):
            TinkerPolicy(config=config, require_grad_logp=True)


class TestMockTinkerTrainingClient:
    """Test MockTinkerTrainingClient for testing without API."""

    def test_forward_backward_custom(self):
        """Mock training client should call loss function."""
        from synthstats.integrations.tinker import MockTinkerTrainingClient

        client = MockTinkerTrainingClient()

        data = [{"prompt": "test", "completion": "answer"}]
        call_count = [0]

        def loss_fn(data, logprobs_list):
            call_count[0] += 1
            loss = torch.tensor(0.5)
            return loss, {"test_metric": 0.5}

        future = client.forward_backward_custom(data, loss_fn)
        result = future.result()

        assert call_count[0] == 1
        assert hasattr(result, "loss")
        assert hasattr(result, "metrics")
        assert result.loss == 0.5

    def test_optim_step(self):
        """Mock training client optim_step should increment counter."""
        from synthstats.integrations.tinker import MockTinkerTrainingClient

        client = MockTinkerTrainingClient()
        assert client._step_count == 0
        client.optim_step()
        assert client._step_count == 1


class TestTinkerTrainerWithMock:
    """Test TinkerTrainer using mock training client."""

    @pytest.fixture
    def mock_trainer(self, monkeypatch):
        """Create trainer with mock client injected."""
        from synthstats.integrations.tinker import (
            MockTinkerTrainingClient,
            TinkerConfig,
            TinkerTrainer,
        )

        config = TinkerConfig(api_key="test-key")
        trainer = TinkerTrainer(config=config, logZ_init=0.5)
        # inject mock client
        trainer._training_client = MockTinkerTrainingClient()
        return trainer

    def test_logZ_parameter(self, mock_trainer):
        """Trainer should have logZ as nn.Parameter."""
        assert isinstance(mock_trainer.logZ, nn.Parameter)
        assert mock_trainer.logZ.item() == 0.5

    def test_train_step_returns_metrics(self, mock_trainer):
        """train_step should return dict with loss and logZ."""
        batch = {
            "log_probs": torch.tensor([[-0.5, -0.3, -0.2]]),
            "loss_mask": torch.ones(1, 3, dtype=torch.bool),
            "log_reward": torch.tensor([0.0]),
            "prompts": ["What is 2+2?"],
            "completions": ["The answer is 4"],
        }

        result = mock_trainer.train_step(batch)

        assert isinstance(result, dict)
        assert "loss" in result
        assert "logZ" in result
        assert isinstance(result["loss"], float)
        assert isinstance(result["logZ"], float)

    def test_train_step_calls_subtb_loss(self, mock_trainer):
        """train_step should use SubTB loss."""
        batch = {
            "log_probs": torch.tensor([[-0.5, -0.3]]),
            "loss_mask": torch.ones(1, 2, dtype=torch.bool),
            "log_reward": torch.tensor([1.0]),
            "prompts": ["test"],
            "completions": ["answer"],
        }

        result = mock_trainer.train_step(batch)

        # loss metrics should be present
        assert "loss" in result
        assert "logZ" in result
        assert "mean_log_reward" in result

    def test_parameters(self, mock_trainer):
        """parameters() should return list containing logZ."""
        params = mock_trainer.parameters()
        assert isinstance(params, list)
        assert len(params) == 1
        assert params[0] is mock_trainer.logZ


class TestTinkerEnvProtocol:
    """Test TinkerEnvProtocol for compatibility checks."""

    def test_protocol_check(self):
        """Protocol should be runtime checkable."""
        from synthstats.integrations.tinker import TinkerEnvProtocol

        class FakeEnv:
            def initial_observation(self) -> str:
                return "test"

            async def step(self, action):
                return None

        env = FakeEnv()
        assert isinstance(env, TinkerEnvProtocol)


class TestIntegrationWithSubTBLoss:
    """Integration tests verifying SubTB loss computation."""

    def test_subtb_loss_gradient_flow(self):
        """Verify SubTB loss computation works with trainer."""
        from synthstats.integrations.tinker import (
            MockTinkerTrainingClient,
            TinkerConfig,
            TinkerTrainer,
        )

        config = TinkerConfig(api_key="test")
        trainer = TinkerTrainer(config=config, logZ_init=1.0)
        trainer._training_client = MockTinkerTrainingClient()

        batch = {
            "log_probs": torch.tensor([[-0.5, -0.3], [-0.4, -0.2]]),
            "loss_mask": torch.ones(2, 2, dtype=torch.bool),
            "log_reward": torch.tensor([0.5, 0.8]),
            "prompts": ["q1", "q2"],
            "completions": ["a1", "a2"],
        }

        # should not raise
        result = trainer.train_step(batch)
        assert result["loss"] >= 0


class TestTrajectoriesToTinkerBatch:
    """Tests for trajectories_to_tinker_batch converter function."""

    def _create_trajectory(
        self,
        messages: list[tuple[str, str]],
        reward: float = 0.5,
    ):
        """Helper to create a Trajectory for testing."""
        from synthstats.core.types import Message, Reward, Trajectory

        return Trajectory(
            messages=[Message(role=role, content=content) for role, content in messages],
            token_ids=[[1, 2, 3]],
            token_logprobs=[[-0.1, -0.2, -0.15]],
            loss_mask=[[True, True, True]],
            reward=Reward(total=reward, components={"likelihood": reward}, info={}),
        )

    def test_basic_conversion(self):
        """Basic trajectory should convert correctly."""
        from synthstats.integrations.tinker import trajectories_to_tinker_batch

        traj = self._create_trajectory(
            [
                ("system", "You are helpful."),
                ("user", "What is 2+2?"),
                ("assistant", "The answer is 4."),
            ],
            reward=0.8,
        )

        batch = trajectories_to_tinker_batch([traj])

        assert len(batch["prompts"]) == 1
        assert len(batch["completions"]) == 1
        assert batch["log_reward"].shape == (1,)

        # prompt should contain system + user
        assert "You are helpful." in batch["prompts"][0]
        assert "What is 2+2?" in batch["prompts"][0]

        # CRITICAL: prompt should end with newline to prevent token fusion
        assert batch["prompts"][0].endswith("\n"), "Prompt must end with newline delimiter"

        # completion should contain assistant
        assert batch["completions"][0] == "The answer is 4."

        # log_reward should be log(0.8)
        import math

        assert abs(batch["log_reward"][0].item() - math.log(0.8)) < 1e-5

    def test_prompt_ends_with_newline_delimiter(self):
        """Prompt must end with newline to prevent token fusion with completion."""
        from synthstats.integrations.tinker import trajectories_to_tinker_batch

        # test various prompt endings
        test_cases = [
            [("user", "No trailing newline"), ("assistant", "Response")],
            [("user", "Has trailing newline\n"), ("assistant", "Response")],
            [("system", "System"), ("user", "User"), ("assistant", "Response")],
        ]

        for messages in test_cases:
            traj = self._create_trajectory(messages)
            batch = trajectories_to_tinker_batch([traj])

            # all prompts should end with newline
            assert batch["prompts"][0].endswith("\n"), (
                f"Prompt '{batch['prompts'][0]!r}' should end with newline"
            )

    def test_multiple_trajectories(self):
        """Multiple trajectories should batch correctly."""
        from synthstats.integrations.tinker import trajectories_to_tinker_batch

        trajs = [
            self._create_trajectory(
                [
                    ("user", "Q1"),
                    ("assistant", "A1"),
                ],
                reward=0.5,
            ),
            self._create_trajectory(
                [
                    ("user", "Q2"),
                    ("assistant", "A2"),
                ],
                reward=0.9,
            ),
        ]

        batch = trajectories_to_tinker_batch(trajs)

        assert len(batch["prompts"]) == 2
        assert len(batch["completions"]) == 2
        assert batch["log_reward"].shape == (2,)

        assert batch["completions"][0] == "A1"
        assert batch["completions"][1] == "A2"

    def test_tool_messages_in_prompt(self):
        """Tool messages should be included in prompt."""
        from synthstats.integrations.tinker import trajectories_to_tinker_batch

        traj = self._create_trajectory(
            [
                ("user", "Run this tool."),
                ("tool", "Tool output: 42"),
                ("assistant", "The result is 42."),
            ]
        )

        batch = trajectories_to_tinker_batch([traj])

        # tool output should be in prompt
        assert "[Tool Result]:" in batch["prompts"][0]
        assert "42" in batch["prompts"][0]

    def test_multiple_assistant_messages_rejected_by_default(self):
        """Multiple assistant messages should be rejected (single-turn only)."""
        from synthstats.integrations.tinker import trajectories_to_tinker_batch

        traj = self._create_trajectory(
            [
                ("user", "Think step by step."),
                ("assistant", "Step 1: thinking..."),
                ("user", "Continue."),
                ("assistant", "Step 2: done!"),
            ]
        )

        # should raise by default
        with pytest.raises(ValueError, match="multi_turn=True"):
            trajectories_to_tinker_batch([traj])

    def test_multiple_assistant_with_override(self):
        """With strict_single_turn=False, multi-turn is allowed (but lossy)."""
        from synthstats.integrations.tinker import trajectories_to_tinker_batch

        traj = self._create_trajectory(
            [
                ("user", "Think step by step."),
                ("assistant", "Step 1: thinking..."),
                ("user", "Continue."),
                ("assistant", "Step 2: done!"),
            ]
        )

        # override allows it but with warning in docstring
        batch = trajectories_to_tinker_batch([traj], strict_single_turn=False)

        # both assistant messages should be in completion
        assert "Step 1" in batch["completions"][0]
        assert "Step 2" in batch["completions"][0]

    def test_reward_floor_prevents_log_zero(self):
        """Reward floor should prevent log(0) = -inf."""
        import math

        from synthstats.integrations.tinker import trajectories_to_tinker_batch

        traj = self._create_trajectory(
            [
                ("user", "Test"),
                ("assistant", "Answer"),
            ],
            reward=0.0,
        )  # zero reward

        batch = trajectories_to_tinker_batch([traj], reward_floor=1e-10)

        # should not be -inf
        assert not math.isinf(batch["log_reward"][0].item())
        # use tolerance for float32 vs float64 precision
        assert abs(batch["log_reward"][0].item() - math.log(1e-10)) < 1e-5

    def test_negative_reward_uses_floor(self):
        """Negative rewards should use floor (shouldn't happen but be safe)."""
        import math

        from synthstats.integrations.tinker import trajectories_to_tinker_batch

        traj = self._create_trajectory(
            [
                ("user", "Test"),
                ("assistant", "Answer"),
            ],
            reward=-1.0,
        )  # negative reward

        batch = trajectories_to_tinker_batch([traj], reward_floor=1e-4)

        # should use floor, not log of negative
        # use tolerance for float32 vs float64 precision
        assert abs(batch["log_reward"][0].item() - math.log(1e-4)) < 1e-5

    def test_device_parameter(self):
        """Device parameter should work."""
        from synthstats.integrations.tinker import trajectories_to_tinker_batch

        traj = self._create_trajectory(
            [
                ("user", "Test"),
                ("assistant", "Answer"),
            ]
        )

        batch = trajectories_to_tinker_batch([traj], device="cpu")
        assert batch["log_reward"].device.type == "cpu"

    def test_empty_trajectories_raises(self):
        """Empty trajectory list should raise or return empty."""
        from synthstats.integrations.tinker import trajectories_to_tinker_batch

        batch = trajectories_to_tinker_batch([])

        assert len(batch["prompts"]) == 0
        assert len(batch["completions"]) == 0
        assert batch["log_reward"].shape == (0,)

    def test_loss_mask_passthrough(self):
        """Trajectory with loss_mask should have loss_mask in batch."""
        from synthstats.core.types import Message, Reward, Trajectory
        from synthstats.integrations.tinker import trajectories_to_tinker_batch

        # trajectory with explicit loss_mask (some tokens masked)
        traj = Trajectory(
            messages=[
                Message(role="user", content="Think step by step."),
                Message(role="assistant", content="<think>hidden</think>answer"),
            ],
            token_ids=[[1, 2, 3, 4, 5]],
            token_logprobs=[[-0.1, -0.2, -0.15, -0.1, -0.2]],
            loss_mask=[[False, False, False, True, True]],  # mask out <think> tokens
            reward=Reward(total=0.5, components={}, info={}),
        )

        batch = trajectories_to_tinker_batch([traj])

        assert "loss_mask" in batch
        assert batch["loss_mask"].shape == (1, 5)
        assert batch["loss_mask"][0].tolist() == [False, False, False, True, True]

    def test_loss_mask_not_present_if_all_empty(self):
        """Trajectory without loss_mask should not have loss_mask in batch."""
        from synthstats.core.types import Message, Reward, Trajectory
        from synthstats.integrations.tinker import trajectories_to_tinker_batch

        # trajectory with empty loss_mask
        traj = Trajectory(
            messages=[
                Message(role="user", content="Question"),
                Message(role="assistant", content="Answer"),
            ],
            token_ids=[[1, 2, 3]],
            token_logprobs=[[-0.1, -0.2, -0.15]],
            loss_mask=[[]],  # empty = no mask
            reward=Reward(total=0.5, components={}, info={}),
        )

        batch = trajectories_to_tinker_batch([traj])

        # no loss_mask key when no actual masks
        assert "loss_mask" not in batch

    def test_loss_mask_padding_mixed_lengths(self):
        """Multiple trajectories with different mask lengths should pad correctly."""
        from synthstats.core.types import Message, Reward, Trajectory
        from synthstats.integrations.tinker import trajectories_to_tinker_batch

        traj1 = Trajectory(
            messages=[
                Message(role="user", content="Q1"),
                Message(role="assistant", content="A1"),
            ],
            token_ids=[[1, 2, 3]],
            token_logprobs=[[-0.1, -0.2, -0.15]],
            loss_mask=[[True, True, True]],  # 3 tokens
            reward=Reward(total=0.5, components={}, info={}),
        )

        traj2 = Trajectory(
            messages=[
                Message(role="user", content="Q2"),
                Message(role="assistant", content="A2 longer response"),
            ],
            token_ids=[[1, 2, 3, 4, 5]],
            token_logprobs=[[-0.1, -0.2, -0.15, -0.1, -0.2]],
            loss_mask=[[True, False, True, False, True]],  # 5 tokens, mixed mask
            reward=Reward(total=0.8, components={}, info={}),
        )

        batch = trajectories_to_tinker_batch([traj1, traj2])

        assert "loss_mask" in batch
        assert batch["loss_mask"].shape == (2, 5)  # padded to max length
        # traj1's mask is padded with True (default)
        assert batch["loss_mask"][0].tolist() == [True, True, True, True, True]
        assert batch["loss_mask"][1].tolist() == [True, False, True, False, True]

    def test_loss_mask_some_trajectories_have_mask(self):
        """Mixed batch: some trajectories have mask, some don't."""
        from synthstats.core.types import Message, Reward, Trajectory
        from synthstats.integrations.tinker import trajectories_to_tinker_batch

        traj_with_mask = Trajectory(
            messages=[
                Message(role="user", content="Q1"),
                Message(role="assistant", content="A1"),
            ],
            token_ids=[[1, 2, 3]],
            token_logprobs=[[-0.1, -0.2, -0.15]],
            loss_mask=[[True, False, True]],
            reward=Reward(total=0.5, components={}, info={}),
        )

        traj_without_mask = Trajectory(
            messages=[
                Message(role="user", content="Q2"),
                Message(role="assistant", content="A2"),
            ],
            token_ids=[[1, 2]],
            token_logprobs=[[-0.1, -0.2]],
            loss_mask=[[]],  # no mask
            reward=Reward(total=0.8, components={}, info={}),
        )

        batch = trajectories_to_tinker_batch([traj_with_mask, traj_without_mask])

        assert "loss_mask" in batch
        assert batch["loss_mask"].shape == (2, 3)  # padded to max length
        # traj1 has explicit mask
        assert batch["loss_mask"][0].tolist() == [True, False, True]
        # traj2 has no mask, defaults to all True
        assert batch["loss_mask"][1].tolist() == [True, True, True]


class TestMultiTurnSupport:
    """Tests for multi-turn trajectory handling."""

    def _create_multi_turn_trajectory(
        self,
        turns: int = 3,
        reward: float = 0.5,
    ):
        """Helper to create multi-turn trajectories."""
        from synthstats.core.types import Message, Reward, Trajectory

        messages = [
            Message(role="system", content="You are a scientist."),
            Message(role="user", content="Analyze this data."),
        ]
        token_ids = []
        token_logprobs = []
        loss_mask = []

        for i in range(turns):
            messages.append(
                Message(
                    role="assistant",
                    content=f"Turn {i}: Analysis step {i}",
                )
            )
            token_ids.append([i * 10 + j for j in range(5)])
            token_logprobs.append([-0.1] * 5)
            loss_mask.append([True, True, False, True, True])

            if i < turns - 1:
                messages.append(
                    Message(
                        role="user",
                        content=f"Continue with step {i + 1}.",
                    )
                )

        return Trajectory(
            messages=messages,
            token_ids=token_ids,
            token_logprobs=token_logprobs,
            loss_mask=loss_mask,
            reward=Reward(total=reward, components={}, info={}),
        )

    def test_multi_turn_conversion_enabled(self):
        """Multi-turn mode should extract turn boundaries."""
        from synthstats.integrations.tinker import trajectories_to_tinker_batch

        traj = self._create_multi_turn_trajectory(turns=3)
        batch = trajectories_to_tinker_batch([traj], multi_turn=True)

        assert batch["is_multi_turn"] is True
        assert "turn_boundaries" in batch
        assert len(batch["turn_boundaries"]) == 1

        boundaries = batch["turn_boundaries"][0]
        assistant_turns = [b for b in boundaries if b.role == "assistant"]
        assert len(assistant_turns) == 3

    def test_multi_turn_boundaries_have_correct_indices(self):
        """Turn boundaries should map to correct generation indices."""
        from synthstats.integrations.tinker import trajectories_to_tinker_batch

        traj = self._create_multi_turn_trajectory(turns=2)
        batch = trajectories_to_tinker_batch([traj], multi_turn=True)

        boundaries = batch["turn_boundaries"][0]
        assistant_boundaries = [b for b in boundaries if b.role == "assistant"]

        assert assistant_boundaries[0].generation_idx == 0
        assert assistant_boundaries[1].generation_idx == 1

    def test_multi_turn_final_turn_has_reward(self):
        """Only final assistant turn should have has_reward=True."""
        from synthstats.integrations.tinker import trajectories_to_tinker_batch

        traj = self._create_multi_turn_trajectory(turns=3)
        batch = trajectories_to_tinker_batch([traj], multi_turn=True)

        boundaries = batch["turn_boundaries"][0]
        assistant_boundaries = [b for b in boundaries if b.role == "assistant"]

        assert assistant_boundaries[0].has_reward is False
        assert assistant_boundaries[1].has_reward is False
        assert assistant_boundaries[2].has_reward is True

    def test_single_turn_backward_compat_in_multi_mode(self):
        """Single-turn trajectories should work in multi_turn mode."""
        from synthstats.core.types import Message, Reward, Trajectory
        from synthstats.integrations.tinker import trajectories_to_tinker_batch

        traj = Trajectory(
            messages=[
                Message(role="user", content="Question"),
                Message(role="assistant", content="Answer"),
            ],
            token_ids=[[1, 2, 3]],
            token_logprobs=[[-0.1, -0.2, -0.3]],
            loss_mask=[[True, True, True]],
            reward=Reward(total=0.5, components={}, info={}),
        )

        # both modes should work
        batch_single = trajectories_to_tinker_batch([traj], multi_turn=False)
        batch_multi = trajectories_to_tinker_batch([traj], multi_turn=True)

        assert batch_single["completions"][0] == "Answer"
        assert batch_multi["completions"][0] == "Answer"
        # single-turn in multi mode has empty boundaries, so is_multi_turn=False
        assert batch_multi["is_multi_turn"] is False

    def test_multi_turn_completion_preserves_structure(self):
        """Multi-turn completion should preserve all messages."""
        from synthstats.core.types import Message, Reward, Trajectory
        from synthstats.integrations.tinker import trajectories_to_tinker_batch

        traj = Trajectory(
            messages=[
                Message(role="user", content="Start"),
                Message(role="assistant", content="Response 1"),
                Message(role="user", content="Continue"),
                Message(role="assistant", content="Response 2"),
            ],
            token_ids=[[1, 2], [3, 4]],
            token_logprobs=[[-0.1, -0.2], [-0.1, -0.2]],
            loss_mask=[[True, True], [True, True]],
            reward=Reward(total=0.5, components={}, info={}),
        )

        batch = trajectories_to_tinker_batch([traj], multi_turn=True)

        # completion should contain all post-first-assistant messages
        completion = batch["completions"][0]
        assert "Response 1" in completion
        assert "Continue" in completion
        assert "Response 2" in completion

    def test_multi_turn_user_turns_in_boundaries(self):
        """User turns between assistant turns should be in boundaries."""
        from synthstats.core.types import Message, Reward, Trajectory
        from synthstats.integrations.tinker import trajectories_to_tinker_batch

        traj = Trajectory(
            messages=[
                Message(role="user", content="Start"),
                Message(role="assistant", content="A1"),
                Message(role="user", content="U2"),
                Message(role="assistant", content="A2"),
            ],
            token_ids=[[1], [2]],
            token_logprobs=[[-0.1], [-0.1]],
            loss_mask=[[True], [True]],
            reward=Reward(total=0.5, components={}, info={}),
        )

        batch = trajectories_to_tinker_batch([traj], multi_turn=True)
        boundaries = batch["turn_boundaries"][0]

        roles = [b.role for b in boundaries]
        assert roles == ["assistant", "user", "assistant"]

    def test_zero_assistant_messages_returns_empty(self):
        """Trajectory with 0 assistant messages should return empty completion."""
        from synthstats.core.types import Message, Reward, Trajectory
        from synthstats.integrations.tinker import trajectories_to_tinker_batch

        traj = Trajectory(
            messages=[
                Message(role="user", content="Question with no answer"),
            ],
            token_ids=[],
            token_logprobs=[],
            loss_mask=[],
            reward=Reward(total=0.5, components={}, info={}),
        )

        batch = trajectories_to_tinker_batch([traj], multi_turn=True)

        assert batch["completions"][0] == ""
        assert batch["is_multi_turn"] is False
        # should have empty boundaries list
        assert len(batch.get("turn_boundaries", [[]])[0]) == 0

    def test_empty_assistant_content_creates_zero_width_boundary(self):
        """Empty assistant content should create zero-width boundary."""
        from synthstats.core.types import Message, Reward, Trajectory
        from synthstats.integrations.tinker import trajectories_to_tinker_batch

        traj = Trajectory(
            messages=[
                Message(role="user", content="Question"),
                Message(role="assistant", content=""),  # empty!
                Message(role="user", content="Follow-up"),
                Message(role="assistant", content="Answer"),
            ],
            token_ids=[[], [1, 2]],
            token_logprobs=[[], [-0.1, -0.2]],
            loss_mask=[[], [True, True]],
            reward=Reward(total=0.5, components={}, info={}),
        )

        batch = trajectories_to_tinker_batch([traj], multi_turn=True)
        boundaries = batch["turn_boundaries"][0]

        # find the empty assistant boundary (generation_idx=0)
        empty_boundary = [b for b in boundaries if b.generation_idx == 0][0]
        assert empty_boundary.start_char == empty_boundary.end_char, (
            "Empty content should create zero-width boundary"
        )


class TestBuildTurnMask:
    """Tests for _build_turn_mask helper function."""

    def test_assistant_only_included(self, mock_tokenizer):
        """Only assistant turn tokens should be True in mask."""
        from synthstats.integrations.tinker import TurnBoundary, _build_turn_mask

        completion = "Assistant response\nUser question\nAssistant again"
        boundaries = [
            TurnBoundary(0, 18, "assistant", 0, False),
            TurnBoundary(19, 32, "user", -1, False),
            TurnBoundary(33, 48, "assistant", 1, True),
        ]

        mask = _build_turn_mask(completion, boundaries, mock_tokenizer, len(completion), "cpu")

        # assistant regions should be True
        assert mask[0:18].all()  # first assistant
        assert mask[33:48].all()  # second assistant

        # user region should be False
        assert not mask[19:32].any()

    def test_empty_boundaries_all_false(self, mock_tokenizer):
        """Empty boundaries should result in all-False mask."""
        from synthstats.integrations.tinker import _build_turn_mask

        mask = _build_turn_mask("some text", [], mock_tokenizer, 9, "cpu")
        assert not mask.any()

    def test_raises_without_offset_mapping(self):
        """Should raise ValueError if tokenizer lacks offset_mapping support."""
        from synthstats.integrations.tinker import TurnBoundary, _build_turn_mask

        completion = "Test"
        boundaries = [TurnBoundary(0, 4, "assistant", 0, True)]

        class SimpleTokenizer:
            def encode_plus(self, text, **kwargs):
                raise TypeError("No offset_mapping")

        with pytest.raises(ValueError, match="does not support offset_mapping"):
            _build_turn_mask(completion, boundaries, SimpleTokenizer(), 4, "cpu")

    def test_prompt_offset_shifts_mask_positions(self, mock_tokenizer):
        """Prompt offset should shift mask to correct positions.

        When Tinker tokenizes prompt+completion, logprobs include prompt tokens.
        The mask must be offset so it marks completion positions, not prompt ones.
        """
        from synthstats.integrations.tinker import TurnBoundary, _build_turn_mask

        # completion = "AB" (2 chars = 2 tokens with our mock)
        # boundary marks entire completion as assistant
        completion = "AB"
        boundaries = [TurnBoundary(0, 2, "assistant", 0, True)]

        # seq_len = 7 (simulating prompt of 5 tokens + completion of 2)
        # without offset: mask[0:2] would be True (WRONG - these are prompt tokens)
        # with offset=5: mask[5:7] should be True (CORRECT - these are completion tokens)
        mask = _build_turn_mask(completion, boundaries, mock_tokenizer, 7, "cpu", prompt_offset=5)

        # prompt positions should be False
        assert not mask[0:5].any(), "Prompt tokens should be masked out"
        # completion positions should be True
        assert mask[5:7].all(), "Completion tokens should be included"

    def test_prompt_offset_with_multi_turn(self, mock_tokenizer):
        """Multi-turn with prompt offset should mark only assistant turns."""
        from synthstats.integrations.tinker import TurnBoundary, _build_turn_mask

        # completion = "A1\nQ2\nA2" = "A1" (assistant) + "\n" + "Q2" (user)
        # + "\n" + "A2" (assistant)
        # chars: A=0, 1=1, \n=2, Q=3, 2=4, \n=5, A=6, 2=7
        completion = "A1\nQ2\nA2"
        boundaries = [
            TurnBoundary(0, 2, "assistant", 0, False),  # "A1" at chars 0-2
            TurnBoundary(3, 5, "user", -1, False),  # "Q2" at chars 3-5
            TurnBoundary(6, 8, "assistant", 1, True),  # "A2" at chars 6-8
        ]

        # prompt has 4 tokens, so offset = 4-1 = 3 (accounting for shift)
        # seq_len = 11 (prompt 4 + completion 8 - 1 for shift)
        mask = _build_turn_mask(completion, boundaries, mock_tokenizer, 11, "cpu", prompt_offset=3)

        # first assistant "A1" at chars 0-2 -> tokens 0-2 -> offset -> positions 3-5
        assert mask[3:5].all(), "First assistant turn should be True"
        # user "Q2" at chars 3-5 -> tokens 3-5 -> offset -> positions 6-8
        assert not mask[6:8].any(), "User turn should be False"
        # second assistant "A2" at chars 6-8 -> tokens 6-8 -> offset -> positions 9-11
        assert mask[9:11].all(), "Second assistant turn should be True"
        # prompt region (positions 0-2) should remain False
        assert not mask[0:3].any(), "Prompt tokens should be False"

    def test_prompt_offset_default_zero(self, mock_tokenizer):
        """Default prompt_offset=0 should not shift mask positions."""
        from synthstats.integrations.tinker import TurnBoundary, _build_turn_mask

        completion = "ABC"
        boundaries = [TurnBoundary(0, 3, "assistant", 0, True)]

        # seq_len = 3 (no prompt), offset = 0 (default - omitted)
        mask = _build_turn_mask(completion, boundaries, mock_tokenizer, 3, "cpu")

        # positions 0-3 should be True (no shift)
        assert mask[0:3].all(), "Default offset=0 should mark positions 0-2"

    def test_prompt_offset_exceeds_seq_len(self, mock_tokenizer):
        """Offset >= seq_len should result in all-False mask (silent edge case)."""
        from synthstats.integrations.tinker import TurnBoundary, _build_turn_mask

        completion = "AB"
        boundaries = [TurnBoundary(0, 2, "assistant", 0, True)]

        # seq_len = 3, but offset = 10 (way too large)
        mask = _build_turn_mask(completion, boundaries, mock_tokenizer, 3, "cpu", prompt_offset=10)

        # all False because adjusted_idx >= seq_len immediately
        assert not mask.any(), "Offset >= seq_len should produce all-False mask"


class TestTinkerTrainerMultiTurn:
    """Integration tests for multi-turn training step."""

    @pytest.fixture
    def mock_trainer(self, monkeypatch):
        """Create trainer with mock client for multi-turn testing."""
        from synthstats.integrations.tinker import (
            MockTinkerTrainingClient,
            TinkerConfig,
            TinkerTrainer,
        )

        config = TinkerConfig(api_key="test-key")
        trainer = TinkerTrainer(config=config, logZ_init=0.5)
        trainer._training_client = MockTinkerTrainingClient()
        return trainer

    def test_train_step_single_turn_unchanged(self, mock_trainer):
        """Single-turn batches should work as before."""
        batch = {
            "prompts": ["Question?"],
            "completions": ["Answer."],
            "log_reward": torch.tensor([0.5]),
            "is_multi_turn": False,
        }

        result = mock_trainer.train_step(batch)

        assert "loss" in result
        assert "logZ" in result
        assert result.get("is_multi_turn", 0.0) == 0.0

    def test_train_step_multi_turn_with_boundaries(self, mock_trainer):
        """Multi-turn batches should use turn boundaries for masking."""
        from synthstats.integrations.tinker import TurnBoundary

        batch = {
            "prompts": ["System\nUser"],
            "completions": ["Asst1\nUser2\nAsst2"],
            "log_reward": torch.tensor([0.5]),
            "is_multi_turn": True,
            "turn_boundaries": [
                [
                    TurnBoundary(0, 5, "assistant", 0, False),
                    TurnBoundary(6, 11, "user", -1, False),
                    TurnBoundary(12, 17, "assistant", 1, True),
                ]
            ],
        }

        result = mock_trainer.train_step(batch)

        assert "loss" in result
        assert result.get("is_multi_turn", 0.0) == 1.0
