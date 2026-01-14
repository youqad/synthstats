"""Modeling module - LLM policy wrappers.

Provides Policy implementations for different backends:
- HFPolicy: HuggingFace Transformers models
- MockPolicy: Deterministic mock for testing

Also provides utilities for processing LLM outputs:
- ThinkBlockSpan: Dataclass for <think> block spans
- find_think_blocks: Detect <think>...</think> blocks in text
- create_think_mask: Token-level masking for SubTB loss exclusion
- create_think_mask_simple: Text-level masking without tokenizer

NOTE: HFPolicy and MockPolicy are now canonically defined in synthstats.policies.
This module re-exports them for backward compatibility.
"""

from synthstats.modeling.thought_mask import (
    ThinkBlockSpan,
    create_think_mask,
    create_think_mask_simple,
    find_think_blocks,
)
from synthstats.policies.hf_policy import HFPolicy, MockPolicy

__all__ = [
    "HFPolicy",
    "MockPolicy",
    "ThinkBlockSpan",
    "create_think_mask",
    "create_think_mask_simple",
    "find_think_blocks",
]
