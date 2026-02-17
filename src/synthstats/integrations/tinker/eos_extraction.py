"""EOS logprob extraction from Tinker's top-k sampling."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from synthstats.core.constants import STOP_TOKEN_IDS


def extract_eos_from_topk(
    topk_token_ids: Tensor,
    topk_logprobs: Tensor,
    eos_token_ids: list[int],
) -> tuple[Tensor, Tensor]:
    """Extract EOS logprob from top-k candidates via logsumexp."""
    if isinstance(topk_token_ids, list):
        topk_token_ids = _nested_to_tensor(topk_token_ids, dtype=torch.long)
    if isinstance(topk_logprobs, list):
        topk_logprobs = _nested_to_tensor(topk_logprobs, dtype=torch.float32)

    B, T, k = topk_token_ids.shape
    device = topk_token_ids.device

    eos_mask = torch.zeros(B, T, k, dtype=torch.bool, device=device)
    for eos_id in eos_token_ids:
        eos_mask |= topk_token_ids == eos_id

    eos_available = eos_mask.any(dim=-1)
    eos_logprob = torch.full((B, T), -1e6, device=device, dtype=topk_logprobs.dtype)

    masked_logprobs = torch.where(
        eos_mask, topk_logprobs, torch.full_like(topk_logprobs, -float("inf"))
    )
    eos_logprob_all = torch.logsumexp(masked_logprobs, dim=-1)
    eos_logprob = torch.where(eos_available, eos_logprob_all, eos_logprob)

    return eos_logprob, eos_available


def extract_eos_from_tinker_result(
    sample_result: Any,
    eos_token_ids: list[int],
    device: str | torch.device = "cpu",
) -> tuple[Tensor, Tensor]:
    """Extract EOS logprob from Tinker SampleResult."""
    if hasattr(sample_result, "sequences"):
        sequences = sample_result.sequences
    elif hasattr(sample_result, "topk_prompt_logprobs"):
        return _process_topk_list(sample_result.topk_prompt_logprobs, eos_token_ids, device)
    else:
        raise ValueError("sample_result must have 'sequences' or 'topk_prompt_logprobs' attribute")

    all_eos_logprobs = []
    all_eos_available = []

    for seq in sequences:
        if not hasattr(seq, "topk_prompt_logprobs") or seq.topk_prompt_logprobs is None:
            raise ValueError("Sequence missing topk_prompt_logprobs")

        eos_lp, eos_avail = _process_topk_list(seq.topk_prompt_logprobs, eos_token_ids, device)
        all_eos_logprobs.append(eos_lp)
        all_eos_available.append(eos_avail)

    eos_logprob = torch.stack(all_eos_logprobs, dim=0)
    eos_available = torch.stack(all_eos_available, dim=0)

    return eos_logprob, eos_available


def _process_topk_list(
    topk_list: list[list[tuple[int, float]]],
    eos_token_ids: list[int],
    device: str | torch.device,
) -> tuple[Tensor, Tensor]:
    """Process Tinker's top-k format: list of [(token_id, logprob), ...]."""
    T = len(topk_list)
    eos_token_set = set(eos_token_ids)

    eos_logprob = torch.full((T,), -1e6, device=device, dtype=torch.float32)
    eos_available = torch.zeros(T, dtype=torch.bool, device=device)

    for t, topk_at_pos in enumerate(topk_list):
        eos_logprobs_at_pos = [lp for (tok_id, lp) in topk_at_pos if tok_id in eos_token_set]

        if eos_logprobs_at_pos:
            eos_available[t] = True
            if len(eos_logprobs_at_pos) == 1:
                eos_logprob[t] = eos_logprobs_at_pos[0]
            else:
                eos_logprob[t] = torch.logsumexp(
                    torch.tensor(eos_logprobs_at_pos, device=device), dim=0
                )

    return eos_logprob, eos_available


def _nested_to_tensor(nested: list, dtype: torch.dtype) -> Tensor:
    import numpy as np

    arr = np.array(nested)
    return torch.tensor(arr, dtype=dtype)


def pad_eos_for_subtb(
    eos_logprob: Tensor,
    eos_available: Tensor,
) -> tuple[Tensor, Tensor]:
    """Pad EOS tensors from [B, T] to [B, T+1] for SubTB endpoint loss.

    SubTB needs T+1 positions to form subtrajectories. The appended
    terminal position assumes P(stop)=1 (log=0), which holds when the
    sequence was truncated or ended with an explicit EOS. For variable-length
    stopping where the model could continue, this introduces small bias.
    """
    B, T = eos_logprob.shape
    device = eos_logprob.device
    dtype = eos_logprob.dtype

    terminal_logprob = torch.zeros(B, 1, device=device, dtype=dtype)
    terminal_available = torch.ones(B, 1, device=device, dtype=torch.bool)

    eos_logprob_padded = torch.cat([eos_logprob, terminal_logprob], dim=-1)
    eos_available_padded = torch.cat([eos_available, terminal_available], dim=-1)

    return eos_logprob_padded, eos_available_padded


def get_default_eos_token_ids(model_name: str) -> list[int]:
    """Return EOS token IDs for a model family."""
    model_lower = model_name.lower()

    for family, ids in STOP_TOKEN_IDS.items():
        if family == "default":
            continue
        if family in model_lower:
            return list(ids)

    return list(STOP_TOKEN_IDS["default"])
