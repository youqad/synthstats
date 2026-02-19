"""ActionCodec protocol and implementations (JSON and XML)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from synthstats.core.types import Action, FinalAnswer, Program, ToolCall


class ParseError(Exception):
    pass


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@runtime_checkable
class ActionCodec(Protocol):
    def format_action_spec(self, tools: list[ToolSpec]) -> str: ...

    def parse(self, assistant_text: str) -> Action: ...

    def render(self, action: Action) -> str: ...


class JSONToolCodec:
    _CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL | re.IGNORECASE)

    @staticmethod
    def _find_json_objects(text: str) -> list[str]:
        """Extract JSON objects via brace counting, handling string literals."""
        objects = []
        depth = 0
        start = None
        in_string = False
        escape_next = False

        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if char == "\\" and in_string:
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    objects.append(text[start : i + 1])
                    start = None
        return objects

    def format_action_spec(self, tools: list[ToolSpec]) -> str:
        lines = [
            "You can respond with JSON in a code block to take actions:",
            "",
            "To call a tool:",
            "```json",
            '{"tool": "<tool_name>", "input": {<parameters>}}',
            "```",
            "",
            "To submit a final answer:",
            "```json",
            '{"answer": "<your_answer>"}',
            "```",
            "",
            "To submit a program:",
            "```json",
            '{"program": "<code>", "language": "pymc"}',
            "```",
            "",
            "Available tools:",
        ]
        for tool in tools:
            lines.append(f"- {tool.name}: {tool.description}")
            if tool.parameters:
                lines.append(f"  Parameters: {json.dumps(tool.parameters)}")
        return "\n".join(lines)

    def parse(self, assistant_text: str) -> Action:
        code_blocks = self._CODE_BLOCK_RE.findall(assistant_text)
        for block in code_blocks:
            try:
                return self._parse_json_object(block.strip())
            except (json.JSONDecodeError, KeyError):
                continue

        inline_matches = self._find_json_objects(assistant_text)
        for match in inline_matches:
            try:
                return self._parse_json_object(match)
            except (json.JSONDecodeError, KeyError):
                continue

        raise ParseError(f"No valid JSON action found in text: {assistant_text[:200]}")

    def _parse_json_object(self, text: str) -> Action:
        data = json.loads(text)

        if "tool" in data:
            return ToolCall(
                name=data["tool"],
                input=data.get("input", {}),
                raw=text,
            )
        elif "answer" in data:
            return FinalAnswer(text=data["answer"])
        elif "program" in data:
            return Program(
                code=data["program"],
                language=data.get("language", "pymc"),
            )
        else:
            raise KeyError("JSON must have 'tool', 'answer', or 'program' key")

    def render(self, action: Action) -> str:
        if isinstance(action, ToolCall):
            return json.dumps({"tool": action.name, "input": action.input})
        elif isinstance(action, FinalAnswer):
            return json.dumps({"answer": action.text})
        elif isinstance(action, Program):
            return json.dumps({"program": action.code, "language": action.language})
        else:
            raise ValueError(f"Unknown action type: {type(action)}")


class XMLToolCodec:
    _TOOL_RE = re.compile(
        r'<tool\s+name=["\']([^"\']+)["\']\s*>(.*?)</tool>',
        re.DOTALL | re.IGNORECASE,
    )
    _ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
    _PROGRAM_RE = re.compile(
        r'<program(?:\s+language=["\']([^"\']+)["\'])?\s*>(.*?)</program>',
        re.DOTALL | re.IGNORECASE,
    )

    def format_action_spec(self, tools: list[ToolSpec]) -> str:
        lines = [
            "You can use XML tags to take actions:",
            "",
            "To call a tool:",
            '<tool name="tool_name">{"param": "value"}</tool>',
            "",
            "To submit a final answer:",
            "<answer>Your answer here</answer>",
            "",
            "To submit a program:",
            '<program language="pymc">',
            "your code here",
            "</program>",
            "",
            "Available tools:",
        ]
        for tool in tools:
            lines.append(f"- {tool.name}: {tool.description}")
            if tool.parameters:
                lines.append(f"  Parameters: {json.dumps(tool.parameters)}")
        return "\n".join(lines)

    def parse(self, assistant_text: str) -> Action:
        tool_match = self._TOOL_RE.search(assistant_text)
        if tool_match:
            name = tool_match.group(1)
            params_text = tool_match.group(2).strip()
            try:
                params = json.loads(params_text) if params_text else {}
            except json.JSONDecodeError:
                # preserve raw text for debugging; executor can attempt to use it
                params = {"_parse_error": True, "_raw": params_text}
            return ToolCall(name=name, input=params, raw=tool_match.group(0))

        answer_match = self._ANSWER_RE.search(assistant_text)
        if answer_match:
            return FinalAnswer(text=answer_match.group(1))

        program_match = self._PROGRAM_RE.search(assistant_text)
        if program_match:
            language = program_match.group(1) or "pymc"
            code = program_match.group(2)  # preserve whitespace (important for Python)
            return Program(code=code, language=language)

        raise ParseError(f"No valid XML action found in text: {assistant_text[:200]}")

    def render(self, action: Action) -> str:
        if isinstance(action, ToolCall):
            params = json.dumps(action.input) if action.input else ""
            return f'<tool name="{action.name}">{params}</tool>'
        elif isinstance(action, FinalAnswer):
            return f"<answer>{action.text}</answer>"
        elif isinstance(action, Program):
            return f'<program language="{action.language}">{action.code}</program>'
        else:
            raise ValueError(f"Unknown action type: {type(action)}")
