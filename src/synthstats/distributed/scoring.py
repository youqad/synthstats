"""Actor scoring for distributed GFlowNet training.

Computes log probabilities and EOS log probabilities via teacher-forcing.
vLLM only returns logprobs for sampled tokens, but modified SubTB needs
eos_logprobs at every step. So we use the FSDP actor model for scoring
and vLLM only for generation.

Note on token alignment: logits[:, t] predicts token t+1 (shift required).
Multi-EOS models (Qwen3, GLM) need logsumexp over all stop tokens.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# common stop tokens by model family
STOP_TOKEN_IDS: dict[str, list[int]] = {
    # Qwen3 models use multiple stop tokens
    "qwen3": [151643, 151645],  # <|endoftext|>, <|im_end|>
    "qwen2": [151643, 151645],
    # GLM models
    "glm": [151329, 151336, 151338],  # <|endoftext|>, <|user|>, <|observation|>
    # Llama/Mistral models
    "llama": [2],  # </s>
    "mistral": [2],
    # default fallback
    "default": [2],
}


def get_stop_token_ids(
    tokenizer: Any,
    model_name: str | None = None,
) -> list[int]:
    """Get stop token IDs from tokenizer, with model family fallback.

    Uses eos_token_id(s) plus known model-family stop tokens. Avoids treating
    all additional special tokens as stop tokens because many are role/control
    markers for chat models.
    """
    stop_ids: list[int] = []

    def _add_id(tok_id: Any) -> None:
        if tok_id is None:
            return
        if isinstance(tok_id, (list, tuple, set)):
            for item in tok_id:
                _add_id(item)
            return
        try:
            tok_int = int(tok_id)
        except (TypeError, ValueError):
            return
        if tok_int not in stop_ids:
            stop_ids.append(tok_int)

    # try to get from tokenizer first
    if hasattr(tokenizer, "eos_token_id"):
        _add_id(tokenizer.eos_token_id)
    if hasattr(tokenizer, "eos_token_ids"):
        _add_id(tokenizer.eos_token_ids)

    # infer model name from tokenizer if not provided
    if model_name is None and hasattr(tokenizer, "name_or_path"):
        model_name = tokenizer.name_or_path

    # if model name provided, use known stop tokens
    if model_name:
        name_lower = str(model_name).lower()
        for family, ids in STOP_TOKEN_IDS.items():
            if family == "default":
                continue
            if family in name_lower:
                for tok_id in ids:
                    _add_id(tok_id)
                break

    # fallback to default if nothing found
    if not stop_ids:
        stop_ids = STOP_TOKEN_IDS["default"]

    return stop_ids


def compute_log_probs_with_eos(
    model: Any,
    input_ids: Tensor,
    attention_mask: Tensor,
    response_mask: Tensor,
    stop_token_ids: list[int],
    temperature: float | Tensor = 1.0,
    chunk_size: int | None = None,
    no_grad: bool = False,
) -> tuple[Tensor, Tensor]:
    """Compute log probabilities and EOS log probabilities via teacher-forcing.

    Args:
        model: HuggingFace model (or FSDP-wrapped)
        input_ids: [B, L] prompt + response tokens
        attention_mask: [B, L] attention mask
        response_mask: [B, L-1] mask for trainable positions (after shift)
        stop_token_ids: List of valid termination token IDs
        temperature: Scalar or [B] tensor. Per-sample temperatures for on-policy scoring.
        chunk_size: If set, process sequence in chunks (for large models)
        no_grad: If True, use inference_mode (for replay scoring). Default False for training.

    Returns:
        Tuple of (log_probs, eos_logprobs), both [B, L-1]
    """
    B, L = input_ids.shape
    device = input_ids.device
    dtype = next(model.parameters()).dtype

    # edge case: sequences with 0 or 1 tokens have no positions to score
    if L <= 1:
        empty = torch.zeros(B, max(0, L - 1), device=device, dtype=dtype)
        return empty, empty

    # normalize temperature to [B, 1, 1] for broadcasting
    if isinstance(temperature, Tensor):
        temp = temperature.view(B, 1, 1).to(device=device, dtype=dtype)
    else:
        temp = torch.full((B, 1, 1), temperature, device=device, dtype=dtype)

    if chunk_size is not None and L > chunk_size:
        if not no_grad:
            logger.warning(
                "Chunked scoring with gradients not recommended (memory). "
                "Consider smaller batch or no_grad=True for replay."
            )
        return _compute_log_probs_chunked(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            response_mask=response_mask,
            stop_token_ids=stop_token_ids,
            temperature=temp,
            chunk_size=chunk_size,
            no_grad=no_grad,
        )

    # training needs gradients; replay uses inference_mode for memory
    context = torch.inference_mode() if no_grad else torch.enable_grad()
    with context:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, L, V]

    # shift: logits[t] predicts token t+1
    next_logits = logits[:, :-1, :] / temp  # [B, L-1, V]
    targets = input_ids[:, 1:]  # [B, L-1]

    log_Z = torch.logsumexp(next_logits, dim=-1)  # [B, L-1]
    target_logits = next_logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
    log_probs = target_logits - log_Z  # [B, L-1]

    # multi-EOS: logsumexp over all valid stop tokens
    stop_ids_tensor = torch.tensor(stop_token_ids, device=device, dtype=torch.long)
    eos_logits = next_logits[..., stop_ids_tensor]  # [B, L-1, num_stop_tokens]
    eos_logprobs = torch.logsumexp(eos_logits, dim=-1) - log_Z  # [B, L-1]

    # mask prompt positions
    if response_mask is not None:
        response_mask_f = response_mask.to(dtype=dtype)
        log_probs = log_probs * response_mask_f
        eos_logprobs = eos_logprobs * response_mask_f

    return log_probs, eos_logprobs


def _compute_log_probs_chunked(
    model: Any,
    input_ids: Tensor,
    attention_mask: Tensor,
    response_mask: Tensor,
    stop_token_ids: list[int],
    temperature: Tensor,
    chunk_size: int,
    no_grad: bool = False,
) -> tuple[Tensor, Tensor]:
    """Chunked scoring with context overlap for large models (30B+)."""
    B, L = input_ids.shape
    device = input_ids.device
    dtype = next(model.parameters()).dtype

    temp = temperature
    context_size = min(chunk_size, L - 1)

    all_log_probs: list[Tensor] = []
    all_eos_logprobs: list[Tensor] = []

    stop_ids_tensor = torch.tensor(stop_token_ids, device=device, dtype=torch.long)

    pos = 0
    context = torch.inference_mode() if no_grad else torch.enable_grad()
    while pos < L - 1:
        # determine context start (go back context_size, but not before 0)
        context_start = max(0, pos - context_size)
        # determine chunk end (advance by chunk_size, cap at L)
        chunk_end = min(pos + chunk_size + 1, L)  # +1 for target token

        # slice with context
        chunk_input_ids = input_ids[:, context_start:chunk_end]
        chunk_attention = attention_mask[:, context_start:chunk_end]

        with context:
            outputs = model(input_ids=chunk_input_ids, attention_mask=chunk_attention)
            logits = outputs.logits  # [B, chunk_len, V]

        next_logits = logits[:, :-1, :] / temp
        targets = chunk_input_ids[:, 1:]

        log_Z = torch.logsumexp(next_logits, dim=-1)
        target_logits = next_logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        chunk_log_probs = target_logits - log_Z

        eos_logits = next_logits[..., stop_ids_tensor]
        chunk_eos_logprobs = torch.logsumexp(eos_logits, dim=-1) - log_Z

        # keep only new positions
        keep_start = pos - context_start
        keep_end = min(chunk_end - context_start - 1, chunk_log_probs.shape[1])

        chunk_lp = chunk_log_probs[:, keep_start:keep_end]
        chunk_eos = chunk_eos_logprobs[:, keep_start:keep_end]

        # detach to save memory
        if no_grad:
            chunk_lp = chunk_lp.detach()
            chunk_eos = chunk_eos.detach()

        all_log_probs.append(chunk_lp)
        all_eos_logprobs.append(chunk_eos)

        # advance position by actual new positions processed
        pos += keep_end - keep_start

    log_probs = torch.cat(all_log_probs, dim=1)  # [B, L-1]
    eos_logprobs = torch.cat(all_eos_logprobs, dim=1)  # [B, L-1]

    if response_mask is not None:
        response_mask_f = response_mask.to(dtype=dtype)
        log_probs = log_probs * response_mask_f
        eos_logprobs = eos_logprobs * response_mask_f

    return log_probs, eos_logprobs


def build_response_mask(
    input_ids: Tensor,
    prompt_lengths: Tensor,
    pad_token_id: int | None = None,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Returns [B, L-1] mask where 1 = response position, 0 = prompt/pad.

    Args:
        input_ids: [B, L] token IDs
        prompt_lengths: [B] length of prompt for each sequence
        pad_token_id: DEPRECATED - use attention_mask instead. Kept for compatibility.
        attention_mask: [B, L] attention mask (1=real token, 0=padding). Preferred
            over pad_token_id because many models use pad_token_id == eos_token_id,
            which would incorrectly mask legitimate EOS tokens.

    Returns:
        [B, L-1] mask where 1 = trainable response position
    """
    B, L = input_ids.shape
    device = input_ids.device

    # create position indices [0, 1, ..., L-2]
    positions = torch.arange(L - 1, device=device).unsqueeze(0).expand(B, -1)

    # position t corresponds to predicting token t+1
    # mask positions where t+1 is part of response (t >= prompt_length - 1)
    response_start = prompt_lengths.unsqueeze(1)
    mask = (positions >= response_start - 1).float()

    # mask out padding - prefer attention_mask over pad_token_id
    # attention_mask correctly handles pad==eos case
    if attention_mask is not None:
        # attention_mask[:, 1:] corresponds to target positions
        target_attention = attention_mask[:, 1:].float()
        mask = mask * target_attention
    elif pad_token_id is not None:
        # fallback: token ID comparison (can incorrectly mask EOS when pad==eos)
        targets = input_ids[:, 1:]
        pad_mask = (targets != pad_token_id).float()
        mask = mask * pad_mask

    return mask
