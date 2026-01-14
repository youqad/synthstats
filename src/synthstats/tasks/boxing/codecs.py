"""BoxingCodec - parses query/submit actions for BoxingGym.

Handles two action formats:
1. Tool calls: <tool_call>{"name": "query", "input": {"query": "..."}}</tool_call>
2. Program submission: <submit_program>code</submit_program>
"""

import json
import re

from synthstats.core.types import Action, Program, ToolCall


class BoxingCodec:
    """Codec for parsing and formatting BoxingGym actions."""

    def parse(self, text: str) -> Action | None:
        """Parse raw text into a structured Action.

        Args:
            text: Raw text containing action markup.

        Returns:
            Parsed Action (ToolCall or Program), or None if unparseable.
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
            except (json.JSONDecodeError, KeyError):
                pass

        # try submit_program format
        program_match = re.search(
            r"<submit_program>(.*?)</submit_program>", text, re.DOTALL
        )
        if program_match:
            code = program_match.group(1).strip()
            return Program(code=code, language="pymc")

        return None

    def format(self, action: Action) -> str:
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
