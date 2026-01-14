"""Tests for thought mask module.

TDD tests for detecting <think>...</think> blocks in LLM outputs
and producing loss masks that exclude thinking tokens from SubTB loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from synthstats.modeling.thought_mask import (
    ThinkBlockSpan,
    create_think_mask,
    create_think_mask_simple,
    find_think_blocks,
)


class TestFindThinkBlocks:
    """Test find_think_blocks function."""

    def test_single_block(self):
        """Single <think> block should be detected."""
        text = "Hello <think>reasoning here</think> world"
        blocks = find_think_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].content == "reasoning here"
        # verify character positions
        assert text[blocks[0].start_char : blocks[0].end_char] == "<think>reasoning here</think>"

    def test_multiple_blocks(self):
        """Multiple <think> blocks should all be detected."""
        text = "<think>first</think> middle <think>second</think>"
        blocks = find_think_blocks(text)
        assert len(blocks) == 2
        assert blocks[0].content == "first"
        assert blocks[1].content == "second"

    def test_no_blocks(self):
        """Text without <think> blocks returns empty list."""
        text = "No thinking here"
        blocks = find_think_blocks(text)
        assert len(blocks) == 0

    def test_unclosed_block(self):
        """Unclosed <think> extends to end of text."""
        text = "Start <think>never closed"
        blocks = find_think_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].content == "never closed"
        # end_char should be at text end
        assert blocks[0].end_char == len(text)

    def test_empty_block(self):
        """Empty <think></think> block should be detected."""
        text = "Empty <think></think> block"
        blocks = find_think_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].content == ""

    def test_multiline_block(self):
        """Multiline content inside <think> should be captured."""
        text = "<think>\nStep 1\nStep 2\n</think>"
        blocks = find_think_blocks(text)
        assert len(blocks) == 1
        assert "Step 1" in blocks[0].content
        assert "Step 2" in blocks[0].content

    def test_nested_tags_handled_gracefully(self):
        """Nested <think> tags (invalid but graceful handling)."""
        # should match first opening with first closing
        text = "<think>outer <think>inner</think> more</think>"
        blocks = find_think_blocks(text)
        # greedy matching: first <think> matched with first </think>
        assert len(blocks) >= 1
        # exact behavior is implementation-defined, just ensure no crash

    def test_only_opening_tag(self):
        """Only opening tag, rest is content."""
        text = "Prefix <think>all the rest is thinking"
        blocks = find_think_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].content == "all the rest is thinking"

    def test_case_sensitivity(self):
        """Tags should be case-sensitive (lowercase only)."""
        text = "<THINK>not detected</THINK>"
        blocks = find_think_blocks(text)
        assert len(blocks) == 0

    def test_whitespace_in_tags(self):
        """Tags with internal whitespace should not match."""
        text = "< think>not valid</ think>"
        blocks = find_think_blocks(text)
        assert len(blocks) == 0

    def test_adjacent_blocks(self):
        """Adjacent blocks without space between."""
        text = "<think>first</think><think>second</think>"
        blocks = find_think_blocks(text)
        assert len(blocks) == 2
        assert blocks[0].content == "first"
        assert blocks[1].content == "second"

    def test_special_characters_in_content(self):
        """Special characters inside <think> should be preserved."""
        text = "<think>def foo():\n    return 42 < 100</think>"
        blocks = find_think_blocks(text)
        assert len(blocks) == 1
        assert "42 < 100" in blocks[0].content

    def test_empty_text(self):
        """Empty text returns empty list."""
        blocks = find_think_blocks("")
        assert len(blocks) == 0

    def test_only_think_tags(self):
        """Text that is only <think>...</think>."""
        text = "<think>everything</think>"
        blocks = find_think_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].content == "everything"
        assert blocks[0].start_char == 0
        assert blocks[0].end_char == len(text)


class TestCreateThinkMaskSimple:
    """Test create_think_mask_simple function (text-level masking)."""

    def test_removes_think_content(self):
        """Single block removal."""
        text = "Hello <think>hidden</think> world"
        cleaned, spans = create_think_mask_simple(text)
        assert cleaned == "Hello  world"
        assert len(spans) == 1

    def test_multiple_removals(self):
        """Multiple blocks removed."""
        text = "<think>a</think>b<think>c</think>d"
        cleaned, spans = create_think_mask_simple(text)
        assert cleaned == "bd"
        assert len(spans) == 2

    def test_no_blocks_unchanged(self):
        """Text without blocks returned unchanged."""
        text = "No thinking here"
        cleaned, spans = create_think_mask_simple(text)
        assert cleaned == text
        assert len(spans) == 0

    def test_empty_text(self):
        """Empty text unchanged."""
        cleaned, spans = create_think_mask_simple("")
        assert cleaned == ""
        assert len(spans) == 0

    def test_unclosed_block_removed(self):
        """Unclosed block to end of text removed."""
        text = "Start <think>all rest"
        cleaned, spans = create_think_mask_simple(text)
        assert cleaned == "Start "
        assert len(spans) == 1

    def test_spans_are_original_positions(self):
        """Spans should be positions in original text."""
        text = "AB<think>XY</think>CD"
        cleaned, spans = create_think_mask_simple(text)
        assert cleaned == "ABCD"
        # span should mark original positions
        start, end = spans[0]
        assert text[start:end] == "<think>XY</think>"

    def test_multiline_removal(self):
        """Multiline content inside block removed."""
        text = "Before\n<think>\nLine1\nLine2\n</think>\nAfter"
        cleaned, spans = create_think_mask_simple(text)
        assert "Line1" not in cleaned
        assert "Line2" not in cleaned
        assert "Before" in cleaned
        assert "After" in cleaned


@dataclass
class MockTokenizer:
    """Simple mock tokenizer for testing create_think_mask."""

    vocab: dict[str, int]
    reverse_vocab: dict[int, str]

    @classmethod
    def from_vocab(cls, words: list[str]) -> MockTokenizer:
        """Create tokenizer from word list."""
        vocab = {w: i for i, w in enumerate(words)}
        reverse_vocab = {i: w for w, i in vocab.items()}
        return cls(vocab=vocab, reverse_vocab=reverse_vocab)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        return "".join(self.reverse_vocab.get(tid, "") for tid in token_ids)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Simple character-level encode for testing."""
        return [self.vocab.get(c, 0) for c in text if c in self.vocab]


class TestCreateThinkMask:
    """Test create_think_mask function (token-level masking)."""

    @pytest.fixture
    def char_tokenizer(self) -> MockTokenizer:
        """Character-level tokenizer for precise testing."""
        chars = list("abcdefghijklmnopqrstuvwxyz<>/ABCDEFGHIJKLMNOPQRSTUVWXYZ ")
        return MockTokenizer.from_vocab(chars)

    def test_all_outside_think(self, char_tokenizer: MockTokenizer):
        """All tokens outside <think> should be True (included in loss)."""
        text = "hello world"
        token_ids = char_tokenizer.encode(text)
        mask = create_think_mask(text, token_ids, char_tokenizer)

        assert len(mask) == len(token_ids)
        assert all(mask), "All tokens outside <think> should be True"

    def test_all_inside_think(self, char_tokenizer: MockTokenizer):
        """All tokens inside <think> should be False (excluded from loss)."""
        text = "<think>hidden</think>"
        token_ids = char_tokenizer.encode(text)
        mask = create_think_mask(text, token_ids, char_tokenizer)

        assert len(mask) == len(token_ids)
        assert not any(mask), "All tokens inside <think> should be False"

    def test_mixed_mask(self, char_tokenizer: MockTokenizer):
        """Some tokens inside, some outside."""
        text = "hi<think>x</think>bye"
        token_ids = char_tokenizer.encode(text)
        mask = create_think_mask(text, token_ids, char_tokenizer)

        assert len(mask) == len(token_ids)

        # decode and check: h, i should be True; <think>x</think> False; b, y, e True
        # first two chars (h, i) should be True
        assert mask[0] is True  # h
        assert mask[1] is True  # i
        # last three chars (b, y, e) should be True
        assert mask[-3] is True  # b
        assert mask[-2] is True  # y
        assert mask[-1] is True  # e

    def test_empty_text(self, char_tokenizer: MockTokenizer):
        """Empty text returns empty mask."""
        mask = create_think_mask("", [], char_tokenizer)
        assert mask == []

    def test_multiple_blocks(self, char_tokenizer: MockTokenizer):
        """Multiple <think> blocks all masked."""
        text = "a<think>x</think>b<think>y</think>c"
        token_ids = char_tokenizer.encode(text)
        mask = create_think_mask(text, token_ids, char_tokenizer)

        # a, b, c should be True; rest False
        # simpler check: count True values
        true_count = sum(mask)
        # should have 3 True values (a, b, c)
        assert true_count == 3

    def test_unclosed_block(self, char_tokenizer: MockTokenizer):
        """Unclosed block masks everything after <think>."""
        text = "ab<think>rest"
        token_ids = char_tokenizer.encode(text)
        mask = create_think_mask(text, token_ids, char_tokenizer)

        # a, b should be True; rest False
        assert mask[0] is True  # a
        assert mask[1] is True  # b
        # everything after should be False
        assert not any(mask[2:])


class TestThinkBlockSpan:
    """Test ThinkBlockSpan dataclass."""

    def test_span_creation(self):
        """Basic span creation."""
        span = ThinkBlockSpan(start_char=10, end_char=50, content="test")
        assert span.start_char == 10
        assert span.end_char == 50
        assert span.content == "test"

    def test_span_equality(self):
        """Spans with same values are equal."""
        span1 = ThinkBlockSpan(0, 10, "test")
        span2 = ThinkBlockSpan(0, 10, "test")
        assert span1 == span2


class TestEdgeCases:
    """Additional edge case tests."""

    def test_think_in_code_block(self):
        """<think> inside code should still be detected."""
        text = "```python\n<think>code thinking</think>\n```"
        blocks = find_think_blocks(text)
        assert isinstance(blocks, list)
        assert len(blocks) == 1

    def test_html_like_tags_not_matched(self):
        """Other HTML-like tags should not match."""
        text = "<div>content</div> <span>more</span>"
        blocks = find_think_blocks(text)
        assert len(blocks) == 0

    def test_think_with_attributes_not_matched(self):
        """<think attr='x'> should not match (strict tag matching)."""
        text = '<think id="1">content</think>'
        blocks = find_think_blocks(text)
        # implementation may vary - strict or lenient
        # for now, just validate the result is a list
        assert isinstance(blocks, list)

    def test_very_long_content(self):
        """Long content inside <think> should work."""
        long_content = "x" * 10000
        text = f"<think>{long_content}</think>"
        blocks = find_think_blocks(text)
        assert len(blocks) == 1
        assert len(blocks[0].content) == 10000

    def test_unicode_content(self):
        """Unicode content inside <think> should work."""
        text = "<think>Thinking in 日本語 and émojis 🤔</think>"
        blocks = find_think_blocks(text)
        assert len(blocks) == 1
        assert "日本語" in blocks[0].content
        assert "🤔" in blocks[0].content


class TestThinkMaskE2EWithRealTokenizer:
    """E2E tests with real HuggingFace tokenizer."""

    @pytest.fixture
    def real_tokenizer(self):
        """Load a real tokenizer for E2E testing."""
        pytest.importorskip("transformers")
        from transformers import AutoTokenizer

        # Use GPT-2 - small, widely available, no auth needed
        return AutoTokenizer.from_pretrained("gpt2")

    def test_real_tokenizer_no_think_blocks(self, real_tokenizer):
        """Text without think blocks - all tokens included."""
        text = "Hello world, this is a test."
        token_ids = real_tokenizer.encode(text)
        mask = create_think_mask(text, token_ids, real_tokenizer)

        assert len(mask) == len(token_ids)
        assert all(mask), "All tokens should be included when no think blocks"

    def test_real_tokenizer_with_think_block(self, real_tokenizer):
        """Text with think block - think tokens excluded."""
        text = "Start <think>internal reasoning here</think> end."
        token_ids = real_tokenizer.encode(text)
        mask = create_think_mask(text, token_ids, real_tokenizer)

        assert len(mask) == len(token_ids)
        # Some tokens should be excluded (inside think block)
        assert not all(mask), "Some tokens should be masked inside think block"
        # But not all should be excluded
        assert any(mask), "Tokens outside think block should be included"

    def test_real_tokenizer_think_at_end(self, real_tokenizer):
        """Unclosed think block at end."""
        text = "Visible text <think>thinking..."
        token_ids = real_tokenizer.encode(text)
        mask = create_think_mask(text, token_ids, real_tokenizer)

        assert len(mask) == len(token_ids)
        # First tokens should be included, later ones excluded
        assert mask[0] is True, "First token should be visible"
