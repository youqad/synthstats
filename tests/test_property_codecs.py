"""Property-based tests for codec roundtrip invariants."""

import json

from hypothesis import given, settings
from hypothesis import strategies as st

from tests.strategies import st_final_answer, st_program, st_tool_call


class TestJSONCodecRoundtrip:
    """Property tests for JSONToolCodec roundtrip."""

    @given(action=st_final_answer())
    @settings(max_examples=100)
    def test_final_answer_roundtrip(self, action):
        """Property: parse(render(FinalAnswer)) == FinalAnswer."""
        from synthstats.core.types import FinalAnswer
        from synthstats.runtime.codecs import JSONToolCodec

        codec = JSONToolCodec()

        rendered = codec.render(action)
        parsed = codec.parse(rendered)

        assert isinstance(parsed, FinalAnswer)
        assert parsed.text == action.text

    @given(action=st_tool_call())
    @settings(max_examples=100)
    def test_tool_call_roundtrip(self, action):
        """Property: parse(render(ToolCall)) preserves name and input."""
        from synthstats.core.types import ToolCall
        from synthstats.runtime.codecs import JSONToolCodec

        codec = JSONToolCodec()

        rendered = codec.render(action)
        parsed = codec.parse(rendered)

        assert isinstance(parsed, ToolCall)
        assert parsed.name == action.name
        assert parsed.input == action.input

    @given(action=st_program())
    @settings(max_examples=100)
    def test_program_roundtrip(self, action):
        """Property: parse(render(Program)) == Program."""
        from synthstats.core.types import Program
        from synthstats.runtime.codecs import JSONToolCodec

        codec = JSONToolCodec()

        rendered = codec.render(action)
        parsed = codec.parse(rendered)

        assert isinstance(parsed, Program)
        assert parsed.code == action.code
        assert parsed.language == action.language

    @given(action=st_tool_call())
    @settings(max_examples=50)
    def test_render_produces_valid_json(self, action):
        """Property: render() always produces valid JSON."""
        from synthstats.runtime.codecs import JSONToolCodec

        codec = JSONToolCodec()

        rendered = codec.render(action)

        # should not raise
        parsed_json = json.loads(rendered)
        assert "tool" in parsed_json
        assert "input" in parsed_json


class TestXMLCodecRoundtrip:
    """Property tests for XMLToolCodec roundtrip.

    XMLToolCodec preserves whitespace in content (important for Python code).
    """

    @given(
        answer_text=st.text(
            min_size=1, max_size=100, alphabet="abcdefghijklmnopqrstuvwxyz0123456789 \n\t"
        ),
    )
    @settings(max_examples=100)
    def test_final_answer_roundtrip(self, answer_text):
        """Property: parse(render(FinalAnswer)) == FinalAnswer."""
        from synthstats.core.types import FinalAnswer
        from synthstats.runtime.codecs import XMLToolCodec

        codec = XMLToolCodec()
        action = FinalAnswer(text=answer_text)

        rendered = codec.render(action)
        parsed = codec.parse(rendered)

        assert isinstance(parsed, FinalAnswer)
        assert parsed.text == action.text

    @given(
        tool_name=st.text(min_size=1, max_size=30, alphabet="abcdefghijklmnopqrstuvwxyz_"),
    )
    @settings(max_examples=50)
    def test_tool_call_name_preserved(self, tool_name):
        """Property: Tool name survives XML roundtrip."""
        from synthstats.core.types import ToolCall
        from synthstats.runtime.codecs import XMLToolCodec

        codec = XMLToolCodec()
        action = ToolCall(name=tool_name, input={}, raw="")

        rendered = codec.render(action)
        parsed = codec.parse(rendered)

        assert isinstance(parsed, ToolCall)
        assert parsed.name == action.name

    @given(
        code=st.text(
            min_size=1, max_size=200, alphabet="abcdefghijklmnopqrstuvwxyz0123456789=() \n\t"
        ),
        language=st.sampled_from(["pymc", "numpyro", "lazyppl"]),
    )
    @settings(max_examples=100)
    def test_program_roundtrip(self, code, language):
        """Property: parse(render(Program)) == Program, including whitespace."""
        from synthstats.core.types import Program
        from synthstats.runtime.codecs import XMLToolCodec

        codec = XMLToolCodec()
        action = Program(code=code, language=language)

        rendered = codec.render(action)
        parsed = codec.parse(rendered)

        assert isinstance(parsed, Program)
        assert parsed.code == action.code
        assert parsed.language == action.language
