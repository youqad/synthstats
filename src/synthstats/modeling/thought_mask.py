"""Thought mask module for detecting and masking <think> blocks.

Detects <think>...</think> blocks in LLM outputs and produces loss masks
that exclude thinking tokens from SubTB loss computation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

# regex for matching <think>...</think> blocks
# - matches opening <think> tag
# - captures content non-greedily
# - matches closing </think> tag
# flags: DOTALL to match across newlines
_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# regex for unclosed <think> block (extends to end of text)
_UNCLOSED_THINK_PATTERN = re.compile(r"<think>(.*)$", re.DOTALL)


@dataclass
class ThinkBlockSpan:
    """A span of text inside <think> tags.

    Attributes:
        start_char: Character offset where <think> tag begins
        end_char: Character offset where </think> ends (or end of text if unclosed)
        content: The text content between the tags (excluding tags themselves)
    """

    start_char: int
    end_char: int
    content: str


def find_think_blocks(text: str) -> list[ThinkBlockSpan]:
    """Find all <think>...</think> blocks in text.

    Returns spans for each block found. Handles:
    - Multiple blocks
    - Unclosed blocks (extends to end of text)
    - Empty blocks

    Args:
        text: The text to search for <think> blocks

    Returns:
        List of ThinkBlockSpan objects for each block found, ordered by position
    """
    if not text:
        return []

    blocks: list[ThinkBlockSpan] = []
    search_start = 0

    while search_start < len(text):
        # try to find a complete <think>...</think> block
        match = _THINK_PATTERN.search(text, search_start)

        if match:
            blocks.append(
                ThinkBlockSpan(
                    start_char=match.start(),
                    end_char=match.end(),
                    content=match.group(1),
                )
            )
            search_start = match.end()
        else:
            # no more complete blocks, check for unclosed <think>
            unclosed = _UNCLOSED_THINK_PATTERN.search(text, search_start)
            if unclosed:
                blocks.append(
                    ThinkBlockSpan(
                        start_char=unclosed.start(),
                        end_char=len(text),
                        content=unclosed.group(1),
                    )
                )
            break

    return blocks


def create_think_mask_simple(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Simple text-level masking without tokenizer.

    Removes all <think>...</think> blocks from text and returns the cleaned
    text along with the original character spans that were removed.

    Args:
        text: The text to clean

    Returns:
        Tuple of:
        - cleaned_text: Text with all <think> blocks removed
        - spans: List of (start, end) tuples marking removed regions in original text
    """
    blocks = find_think_blocks(text)

    if not blocks:
        return text, []

    spans: list[tuple[int, int]] = []
    cleaned_parts: list[str] = []
    last_end = 0

    for block in blocks:
        # add text before this block
        cleaned_parts.append(text[last_end : block.start_char])
        # record the span we're removing
        spans.append((block.start_char, block.end_char))
        last_end = block.end_char

    # add remaining text after last block
    cleaned_parts.append(text[last_end:])

    return "".join(cleaned_parts), spans


def _token_overlaps_any_block(
    token_start: int,
    token_end: int,
    blocks: list[ThinkBlockSpan],
) -> bool:
    """Check if token range overlaps any think block using interval logic.

    Two intervals [a, b) and [c, d) overlap iff NOT (b <= c OR a >= d).
    Equivalently: a < d AND b > c.

    This is O(B) where B = number of blocks (typically 1-2), compared to
    the previous O(n) set membership check.
    """
    for block in blocks:
        # intervals overlap if token starts before block ends AND token ends after block starts
        if token_start < block.end_char and token_end > block.start_char:
            return True
    return False


def create_think_mask(
    text: str,
    token_ids: list[int],
    tokenizer: Any,
) -> list[bool]:
    """Create loss mask excluding <think> tokens.

    Returns a boolean mask where True means "include in loss" and False means
    "exclude from loss" (token is inside a <think> block).

    The approach:
    1. Find all <think> block spans in the text
    2. Decode each token to get its text
    3. Track cumulative character position
    4. For each token, check if its character range overlaps any <think> span

    Args:
        text: The full text that was tokenized
        token_ids: The token IDs from tokenization
        tokenizer: HuggingFace-compatible tokenizer (must support decode())

    Returns:
        List of bools, True = include in loss, False = exclude (inside <think>)

    Warning:
        This implementation uses iterative token decoding to track character
        positions. For BPE/SentencePiece tokenizers, decoded text may not
        exactly match the original (due to normalization, space handling).
        This can cause position drift on long sequences. For production use,
        consider using tokenizer's offset_mapping (return_offsets_mapping=True)
        if available.
    """
    if not token_ids:
        return []

    blocks = find_think_blocks(text)

    if not blocks:
        # no think blocks, all tokens included
        return [True] * len(token_ids)

    # decode tokens incrementally and track positions
    mask: list[bool] = []
    char_pos = 0

    for token_id in token_ids:
        # decode this single token
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        token_len = len(token_text)

        if token_len == 0:
            # special tokens (BOS, EOS, etc.) have no text representation
            # always include in loss - they're structural, not content
            mask.append(True)
        else:
            # check if this token's range overlaps any <think> block
            token_start = char_pos
            token_end = char_pos + token_len

            is_inside = _token_overlaps_any_block(token_start, token_end, blocks)
            mask.append(not is_inside)

            char_pos += token_len

    return mask
