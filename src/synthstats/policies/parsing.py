"""Shared action parsing for policy implementations.

All policies (HF, RunPod, Tinker) parse LLM output through the same cascade:
strip <think> → extract <tool_call> → extract <submit_program> → raw JSON → fallback.
"""

from __future__ import annotations

import json
import re
from typing import Any


def parse_action(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    tc_match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if tc_match:
        try:
            return json.loads(tc_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    sp_match = re.search(r"<submit_program>(.*?)</submit_program>", text, re.DOTALL)
    if sp_match:
        return {"type": "submit_program", "payload": sp_match.group(1).strip()}

    try:
        if "{" in text:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
    except (json.JSONDecodeError, ValueError):
        pass

    return {"type": "answer", "payload": text}


def render_action(action: dict[str, Any]) -> str:
    return json.dumps(action)


def estimate_entropy(logprobs: list[float]) -> float:
    """Mean negative log-prob of sampled tokens (NLL proxy, not true entropy)."""
    if not logprobs:
        return 0.0
    return -sum(logprobs) / len(logprobs)


def build_prompt(obs: str) -> str:
    return (
        "You are an agent that responds to observations.\n"
        f"Observation: {obs}\n"
        "Respond with a JSON action: "
    )
