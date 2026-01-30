"""Tests for scoring utilities."""

from __future__ import annotations

import pytest
import torch

from synthstats.distributed.scoring import (
    STOP_TOKEN_IDS,
    build_response_mask,
    compute_log_probs_with_eos,
    get_stop_token_ids,
)


class TestGetStopTokenIds:
    """Tests for get_stop_token_ids."""

    def test_returns_default_without_tokenizer(self) -> None:
        """Falls back to default tokens."""

        class MockTokenizer:
            eos_token_id = None
            additional_special_tokens_ids = []

        ids = get_stop_token_ids(MockTokenizer())
        assert ids == STOP_TOKEN_IDS["default"]

    def test_uses_tokenizer_eos(self) -> None:
        """Uses tokenizer eos_token_id."""

        class MockTokenizer:
            eos_token_id = 12345
            additional_special_tokens_ids = []

        ids = get_stop_token_ids(MockTokenizer())
        assert 12345 in ids

    def test_ignores_additional_special_tokens(self) -> None:
        """Does not treat role/control tokens as stop tokens."""

        class MockTokenizer:
            eos_token_id = 2
            additional_special_tokens_ids = [99, 100]
            name_or_path = "meta-llama/Llama-3.1-8B"

        ids = get_stop_token_ids(MockTokenizer())
        assert 2 in ids
        assert 99 not in ids
        assert 100 not in ids

    def test_model_name_detection(self) -> None:
        """Detects model family from name."""

        class MockTokenizer:
            eos_token_id = None
            additional_special_tokens_ids = []

        # Qwen model
        ids = get_stop_token_ids(MockTokenizer(), model_name="Qwen/Qwen3-0.6B")
        assert any(tok in ids for tok in STOP_TOKEN_IDS["qwen3"])

        # Llama model
        ids = get_stop_token_ids(MockTokenizer(), model_name="meta-llama/Llama-3.1-8B")
        assert any(tok in ids for tok in STOP_TOKEN_IDS["llama"])


class TestBuildResponseMask:
    """Tests for build_response_mask."""

    def test_basic_mask(self) -> None:
        """Masks prompt, unmasks response."""
        # Sequence: [P, P, R, R, R] where P=prompt, R=response
        # Tokens:    0  1  2  3  4
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        prompt_lengths = torch.tensor([2])

        mask = build_response_mask(input_ids, prompt_lengths)

        # mask[t] = 1 if position t predicts a response token (t+1 >= prompt_length)
        # prompt_length=2, so response starts at token index 2
        # position 0 predicts token 1 (prompt) -> mask=0
        # position 1 predicts token 2 (response) -> mask=1
        # position 2 predicts token 3 (response) -> mask=1
        # position 3 predicts token 4 (response) -> mask=1
        assert mask.shape == (1, 4)
        expected = torch.tensor([[0.0, 1.0, 1.0, 1.0]])
        assert torch.allclose(mask, expected), f"Expected {expected}, got {mask}"

    def test_batch_with_different_prompt_lengths(self) -> None:
        """Handles varying prompt lengths."""
        input_ids = torch.tensor(
            [
                [1, 2, 3, 4, 5, 0],  # prompt=2, response=3, pad=1
                [1, 2, 3, 4, 5, 6],  # prompt=3, response=3
            ]
        )
        prompt_lengths = torch.tensor([2, 3])

        mask = build_response_mask(input_ids, prompt_lengths, pad_token_id=0)

        assert mask.shape == (2, 5)  # [B, L-1]

    def test_with_padding(self) -> None:
        """Padding tokens masked out."""
        input_ids = torch.tensor(
            [
                [1, 2, 3, 0, 0],  # real tokens + padding
            ]
        )
        prompt_lengths = torch.tensor([1])

        mask = build_response_mask(input_ids, prompt_lengths, pad_token_id=0)

        # padding positions should be 0
        assert mask[0, -1].item() == 0  # last position (predicting pad)
        assert mask[0, -2].item() == 0  # second to last (predicting pad)


class TestComputeLogProbsWithEos:
    """Tests for compute_log_probs_with_eos."""

    @pytest.fixture
    def mock_model(self):
        """Model returning predictable logits."""

        class MockModel:
            def __init__(self):
                self._parameters = [torch.nn.Parameter(torch.zeros(1))]

            def parameters(self):
                return iter(self._parameters)

            def __call__(self, input_ids, attention_mask):
                B, L = input_ids.shape
                V = 100  # vocab size

                # create predictable logits
                # make token 0 always most likely, token 2 second
                logits = torch.zeros(B, L, V)
                logits[..., 0] = 10.0  # token 0 has highest prob
                logits[..., 2] = 5.0  # EOS token

                class Output:
                    pass

                out = Output()
                out.logits = logits
                return out

        return MockModel()

    def test_output_shapes(self, mock_model) -> None:
        """Returns [B, L-1] tensors."""
        B, L = 2, 10
        input_ids = torch.randint(0, 100, (B, L))
        attention_mask = torch.ones(B, L)
        response_mask = torch.ones(B, L - 1)
        stop_token_ids = [2]

        log_probs, eos_logprobs = compute_log_probs_with_eos(
            model=mock_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            response_mask=response_mask,
            stop_token_ids=stop_token_ids,
        )

        assert log_probs.shape == (B, L - 1)
        assert eos_logprobs.shape == (B, L - 1)

    def test_temperature_scaling(self, mock_model) -> None:
        """Temperature sharpens/flattens distribution."""
        B, L = 1, 5
        input_ids = torch.zeros(B, L, dtype=torch.long)
        attention_mask = torch.ones(B, L)
        response_mask = torch.ones(B, L - 1)
        stop_token_ids = [2]

        # low temperature -> sharper distribution (higher log prob for top token)
        log_probs_low, _ = compute_log_probs_with_eos(
            model=mock_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            response_mask=response_mask,
            stop_token_ids=stop_token_ids,
            temperature=0.1,
        )

        # high temperature -> flatter distribution (lower log prob for top token)
        log_probs_high, _ = compute_log_probs_with_eos(
            model=mock_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            response_mask=response_mask,
            stop_token_ids=stop_token_ids,
            temperature=2.0,
        )

        # lower temperature should give higher (less negative) log probs for token 0
        # since mock_model gives token 0 highest logit (10.0)
        assert (log_probs_low > log_probs_high).all(), (
            f"Low temp should have higher log probs: low={log_probs_low}, high={log_probs_high}"
        )

    def test_multi_eos_tokens(self, mock_model) -> None:
        """Multiple EOS tokens use logsumexp."""
        B, L = 1, 5
        input_ids = torch.zeros(B, L, dtype=torch.long)
        attention_mask = torch.ones(B, L)
        response_mask = torch.ones(B, L - 1)

        # single EOS
        _, eos_single = compute_log_probs_with_eos(
            model=mock_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            response_mask=response_mask,
            stop_token_ids=[2],
        )

        # multiple EOS (should be higher due to logsumexp)
        _, eos_multi = compute_log_probs_with_eos(
            model=mock_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            response_mask=response_mask,
            stop_token_ids=[2, 3, 4],
        )

        # logsumexp over more tokens should give higher probability
        assert (eos_multi >= eos_single - 1e-5).all()

    def test_response_mask_applied(self, mock_model) -> None:
        """Mask zeros out prompt positions."""
        B, L = 1, 5
        input_ids = torch.zeros(B, L, dtype=torch.long)
        attention_mask = torch.ones(B, L)

        # mask first 2 positions (prompt)
        response_mask = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        stop_token_ids = [2]

        log_probs, eos_logprobs = compute_log_probs_with_eos(
            model=mock_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            response_mask=response_mask,
            stop_token_ids=stop_token_ids,
        )

        # masked positions should be 0
        assert log_probs[0, 0].item() == 0.0
        assert log_probs[0, 1].item() == 0.0
        assert eos_logprobs[0, 0].item() == 0.0
        assert eos_logprobs[0, 1].item() == 0.0

        # unmasked positions should be non-zero
        assert log_probs[0, 2].item() != 0.0
        assert log_probs[0, 3].item() != 0.0

    def test_empty_sequence_edge_case(self, mock_model) -> None:
        """Single-token sequences return empty tensors."""
        # single token: nothing to predict
        input_ids = torch.tensor([[42]])
        attention_mask = torch.ones(1, 1)
        response_mask = torch.zeros(1, 0)  # [B, L-1] = [1, 0]
        stop_token_ids = [2]

        log_probs, eos_logprobs = compute_log_probs_with_eos(
            model=mock_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            response_mask=response_mask,
            stop_token_ids=stop_token_ids,
        )

        # should return empty tensors without crashing
        assert log_probs.shape == (1, 0)
        assert eos_logprobs.shape == (1, 0)
