"""BoxingGym action codec."""

from __future__ import annotations

import json
import re

from synthstats.core.types import Action, Program, ToolCall
from synthstats.runtime.codecs import ParseError


class BoxingCodec:
    """Codec for parsing and formatting BoxingGym actions."""

    def parse(self, text: str) -> Action:
        """Parse raw text into a structured Action.

        Args:
            text: Raw text containing action markup.

        Returns:
            Parsed Action (ToolCall or Program).

        Raises:
            ParseError: If no valid boxing action found.
        """
        # try tool_call format
        tool_match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
        if tool_match:
            try:
                data = json.loads(tool_match.group(1))
                return ToolCall(
                    name=data["name"],
                    input=data.get("input", {}),
                    raw=tool_match.group(0),
                )
            except (json.JSONDecodeError, KeyError) as exc:
                raise ParseError(f"Invalid <tool_call> payload: {exc}") from exc

        # try submit_program format
        program_match = re.search(r"<submit_program>(.*?)</submit_program>", text, re.DOTALL)
        if program_match:
            code = program_match.group(1).strip()
            return Program(code=code, language="pymc")

        raise ParseError(f"No valid boxing action found in text: {text[:200]}")

    def render(self, action: Action) -> str:
        """Format a structured Action as text.

        Args:
            action: Action to format.

        Returns:
            Text representation of the action.
        """
        if isinstance(action, ToolCall):
            data = {"name": action.name, "input": action.input}
            return f"<tool_call>{json.dumps(data)}</tool_call>"
        elif isinstance(action, Program):
            return f"<submit_program>{action.code}</submit_program>"
        else:
            return str(action)
