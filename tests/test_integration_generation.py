"""Integration tests for policy generation with logprobs.

Tests the end-to-end flow of:
- Policy generates text with aligned token_ids and logprobs
- logprobs() method returns valid logprobs for given tokens
- Multiple generations are independent
"""

import pytest


class TestGenerationIntegration:
    """Test policy generation produces valid outputs."""

    def test_mock_policy_generates_with_logprobs(self):
        """MockPolicy generates text with aligned token_ids and logprobs."""
        from synthstats.core.policy import GenConfig
        from synthstats.core.types import Message
        from synthstats.policies.hf_policy import MockPolicy

        policy = MockPolicy(
            fixed_text="Hello world",
            fixed_token_ids=[1, 2, 3],
            fixed_token_logprobs=[-0.5, -0.3, -0.2],
        )

        messages = [Message(role="user", content="Say hello")]
        gen = policy.generate(messages, gen=GenConfig())

        assert gen.text == "Hello world"
        assert len(gen.token_ids) == len(gen.token_logprobs)
        assert all(isinstance(lp, float) for lp in gen.token_logprobs)

    def test_generation_token_alignment(self):
        """Token IDs align with logprobs - same length."""
        from synthstats.core.policy import GenConfig
        from synthstats.core.types import Message
        from synthstats.policies.hf_policy import MockPolicy

        # test with various lengths
        for length in [1, 5, 10, 50]:
            token_ids = list(range(100, 100 + length))
            logprobs = [-0.1 * (i + 1) for i in range(length)]

            policy = MockPolicy(
                fixed_text="test",
                fixed_token_ids=token_ids,
                fixed_token_logprobs=logprobs,
            )

            messages = [Message(role="user", content="test")]
            gen = policy.generate(messages, gen=GenConfig())

            assert len(gen.token_ids) == length
            assert len(gen.token_logprobs) == length
            assert gen.token_ids == token_ids
            assert gen.token_logprobs == logprobs

    def test_generation_logprobs_are_negative(self):
        """Log probabilities should be non-positive (log of probability)."""
        from synthstats.core.policy import GenConfig
        from synthstats.core.types import Message
        from synthstats.policies.hf_policy import MockPolicy

        policy = MockPolicy(
            fixed_text="test output",
            fixed_token_ids=[1, 2, 3, 4, 5],
            fixed_token_logprobs=[-0.5, -0.3, -0.2, -0.7, -0.1],
        )

        messages = [Message(role="user", content="test")]
        gen = policy.generate(messages, gen=GenConfig())

        # log probabilities should be <= 0
        assert all(lp <= 0.0 for lp in gen.token_logprobs)

    def test_multiple_generations_independent(self):
        """Multiple generate calls produce independent results."""
        from synthstats.core.policy import GenConfig
        from synthstats.core.types import Message
        from synthstats.policies.hf_policy import MockPolicy

        policy = MockPolicy(
            fixed_text="response",
            fixed_token_ids=[10, 20, 30],
            fixed_token_logprobs=[-0.5, -0.3, -0.2],
        )

        messages = [Message(role="user", content="test")]

        gen1 = policy.generate(messages, gen=GenConfig())
        gen2 = policy.generate(messages, gen=GenConfig())

        # results should be equal (deterministic mock)
        assert gen1.text == gen2.text
        assert gen1.token_ids == gen2.token_ids
        assert gen1.token_logprobs == gen2.token_logprobs

        # but should be separate objects (not shared references)
        gen1.token_ids[0] = 999
        assert gen2.token_ids[0] != 999

    def test_generation_finish_reason(self):
        """Generation includes finish_reason field."""
        from synthstats.core.policy import GenConfig
        from synthstats.core.types import Message
        from synthstats.policies.hf_policy import MockPolicy

        policy = MockPolicy(
            fixed_text="done",
            fixed_token_ids=[1, 2],
            fixed_token_logprobs=[-0.1, -0.2],
            fixed_finish_reason="stop",
        )

        messages = [Message(role="user", content="test")]
        gen = policy.generate(messages, gen=GenConfig())

        assert gen.finish_reason == "stop"

    def test_generation_with_empty_messages(self):
        """Policy handles empty message list."""
        from synthstats.core.policy import GenConfig
        from synthstats.policies.hf_policy import MockPolicy

        policy = MockPolicy(
            fixed_text="default",
            fixed_token_ids=[1],
            fixed_token_logprobs=[-0.1],
        )

        # empty messages list (edge case)
        gen = policy.generate([], gen=GenConfig())

        assert gen.text == "default"
        assert len(gen.token_ids) == 1

    def test_mock_policy_rejects_misaligned_tokens(self):
        """MockPolicy raises error if token_ids and logprobs lengths differ."""
        from synthstats.policies.hf_policy import MockPolicy

        with pytest.raises(ValueError, match="token_ids length"):
            MockPolicy(
                fixed_text="test",
                fixed_token_ids=[1, 2, 3],
                fixed_token_logprobs=[-0.1, -0.2],  # mismatched length
            )


class TestLogprobsIntegration:
    """Test logprobs() method for computing token logprobs."""

    def test_logprobs_for_known_tokens(self):
        """logprobs() method returns logprobs for given tokens."""
        from synthstats.core.types import Message
        from synthstats.policies.hf_policy import MockPolicy

        policy = MockPolicy()
        messages = [Message(role="user", content="test")]
        tokens = [1, 2, 3, 4, 5]

        result = policy.logprobs(messages, tokens)

        assert len(result.token_ids) == len(tokens)
        assert len(result.logprobs) == len(tokens)
        assert result.token_ids == tokens

    def test_logprobs_returns_negative_values(self):
        """logprobs() returns non-positive values."""
        from synthstats.core.types import Message
        from synthstats.policies.hf_policy import MockPolicy

        policy = MockPolicy()
        messages = [Message(role="user", content="test")]
        tokens = [100, 200, 300]

        result = policy.logprobs(messages, tokens)

        assert all(lp <= 0.0 for lp in result.logprobs)

    def test_logprobs_empty_tokens(self):
        """logprobs() handles empty token list."""
        from synthstats.core.types import Message
        from synthstats.policies.hf_policy import MockPolicy

        policy = MockPolicy()
        messages = [Message(role="user", content="test")]

        result = policy.logprobs(messages, [])

        assert len(result.token_ids) == 0
        assert len(result.logprobs) == 0

    def test_logprobs_single_token(self):
        """logprobs() works for single token."""
        from synthstats.core.types import Message
        from synthstats.policies.hf_policy import MockPolicy

        policy = MockPolicy()
        messages = [Message(role="user", content="test")]

        result = policy.logprobs(messages, [42])

        assert len(result.token_ids) == 1
        assert len(result.logprobs) == 1
        assert result.token_ids[0] == 42

    def test_logprobs_large_sequence(self):
        """logprobs() handles large token sequences."""
        from synthstats.core.types import Message
        from synthstats.policies.hf_policy import MockPolicy

        policy = MockPolicy()
        messages = [Message(role="user", content="test")]
        tokens = list(range(1000))

        result = policy.logprobs(messages, tokens)

        assert len(result.token_ids) == 1000
        assert len(result.logprobs) == 1000


class TestGenerationConfigIntegration:
    """Test generation respects configuration."""

    def test_gen_config_defaults(self):
        """GenConfig has sensible defaults."""
        from synthstats.core.policy import GenConfig

        config = GenConfig()

        assert config.max_tokens > 0
        assert config.temperature >= 0.0
        assert config.top_p > 0.0

    def test_gen_config_custom_values(self):
        """GenConfig accepts custom values."""
        from synthstats.core.policy import GenConfig

        config = GenConfig(
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            stop_sequences=["STOP", "END"],
        )

        assert config.max_tokens == 512
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.stop_sequences == ["STOP", "END"]


class TestGenerationDataclassIntegration:
    """Test Generation dataclass structure."""

    def test_generation_dataclass_fields(self):
        """Generation has all required fields."""
        from synthstats.core.policy import Generation

        gen = Generation(
            text="hello",
            token_ids=[1, 2],
            token_logprobs=[-0.1, -0.2],
            finish_reason="stop",
        )

        assert gen.text == "hello"
        assert gen.token_ids == [1, 2]
        assert gen.token_logprobs == [-0.1, -0.2]
        assert gen.finish_reason == "stop"

    def test_token_logprobs_dataclass(self):
        """TokenLogProbs has required fields."""
        from synthstats.core.policy import TokenLogProbs

        result = TokenLogProbs(
            token_ids=[1, 2, 3],
            logprobs=[-0.1, -0.2, -0.3],
        )

        assert result.token_ids == [1, 2, 3]
        assert result.logprobs == [-0.1, -0.2, -0.3]
