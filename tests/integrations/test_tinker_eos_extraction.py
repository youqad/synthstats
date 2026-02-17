import torch

from synthstats.integrations.tinker.eos_extraction import (
    extract_eos_from_topk,
    get_default_eos_token_ids,
)


class TestExtractEOSFromTopK:
    def test_eos_in_topk(self):
        # [B=1, T=3, k=5]: pos 0 and 2 have EOS token 2, pos 1 does not
        topk_token_ids = torch.tensor([[[2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [1, 2, 3, 4, 5]]])
        topk_logprobs = torch.tensor(
            [
                [
                    [-0.5, -1.0, -1.5, -2.0, -2.5],
                    [-0.5, -1.0, -1.5, -2.0, -2.5],
                    [-0.5, -0.8, -1.0, -1.5, -2.0],
                ]
            ]
        )
        eos_token_ids = [2]

        eos_logprob, eos_available = extract_eos_from_topk(
            topk_token_ids, topk_logprobs, eos_token_ids
        )

        assert eos_available.shape == (1, 3)
        assert eos_available[0, 0].item() is True  # EOS at position 0
        assert eos_available[0, 1].item() is False  # no EOS at position 1
        assert eos_available[0, 2].item() is True  # EOS at position 2

        assert eos_logprob[0, 0] == -0.5  # first in top-k
        assert eos_logprob[0, 2] == -0.8  # second in top-k

    def test_no_eos_in_topk(self):
        topk_token_ids = torch.tensor([[[3, 4, 5], [6, 7, 8]]])
        topk_logprobs = torch.tensor([[[-0.5, -1.0, -1.5], [-0.5, -1.0, -1.5]]])
        eos_token_ids = [2]

        eos_logprob, eos_available = extract_eos_from_topk(
            topk_token_ids, topk_logprobs, eos_token_ids
        )

        assert not eos_available.any()
        assert (eos_logprob == -1e6).all()

    def test_multiple_eos_tokens(self):
        # both EOS tokens (2 and 3) present at position 0
        topk_token_ids = torch.tensor([[[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]]])
        topk_logprobs = torch.tensor(
            [[[-1.0, -2.0, -3.0, -4.0, -5.0], [-1.0, -2.0, -3.0, -4.0, -5.0]]]
        )
        eos_token_ids = [2, 3]

        eos_logprob, eos_available = extract_eos_from_topk(
            topk_token_ids, topk_logprobs, eos_token_ids
        )

        assert eos_available[0, 0].item() is True
        # logsumexp of -1.0 and -2.0
        expected = torch.logsumexp(torch.tensor([-1.0, -2.0]), dim=0)
        assert abs(eos_logprob[0, 0].item() - expected.item()) < 1e-5

    def test_batch_processing(self):
        B, T, k = 2, 2, 3
        topk_token_ids = torch.tensor(
            [
                [[2, 3, 4], [5, 6, 7]],  # batch 0: EOS at pos 0
                [[3, 4, 5], [2, 3, 4]],  # batch 1: EOS at pos 1
            ]
        )
        topk_logprobs = torch.full((B, T, k), -1.0)
        eos_token_ids = [2]

        eos_logprob, eos_available = extract_eos_from_topk(
            topk_token_ids, topk_logprobs, eos_token_ids
        )

        assert eos_available[0, 0].item() is True
        assert eos_available[0, 1].item() is False
        assert eos_available[1, 0].item() is False
        assert eos_available[1, 1].item() is True


class TestGetDefaultEOSTokenIDs:
    def test_qwen_model(self):
        eos_ids = get_default_eos_token_ids("Qwen/Qwen3-4B")
        assert isinstance(eos_ids, list)
        assert len(eos_ids) > 0

    def test_llama_model(self):
        eos_ids = get_default_eos_token_ids("meta-llama/Llama-3.1-8B")
        assert isinstance(eos_ids, list)
        assert 128001 in eos_ids or 128009 in eos_ids

    def test_mistral_model(self):
        eos_ids = get_default_eos_token_ids("mistralai/Ministral-3-3B")
        assert isinstance(eos_ids, list)
        assert 2 in eos_ids

    def test_unknown_model(self):
        eos_ids = get_default_eos_token_ids("unknown-model")
        assert isinstance(eos_ids, list)
        assert len(eos_ids) > 0
