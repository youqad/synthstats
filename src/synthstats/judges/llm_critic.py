"""LLM-as-critic judge for process reward modeling."""

from __future__ import annotations

import re
from typing import Any

from synthstats.core.types import Reward, Trajectory

DEFAULT_PROMPT_TEMPLATE = (
    "You are an expert statistician and probabilistic programmer reviewing PyMC code.\n"
    "\n"
    "TASK: {task_name}\n"
    "\n"
    "PROGRAM TO REVIEW:\n"
    "```python\n"
    "{program}\n"
    "```\n"
    "\n"
    "Rate this program on three dimensions (0.0 to 1.0):\n"
    "\n"
    "1. **code_quality**: Is the code well-structured, readable, and follows PyMC best practices?\n"
    "   - 1.0: Excellent structure, clear naming, proper indentation\n"
    "   - 0.5: Acceptable but could be improved\n"
    "   - 0.0: Poorly structured, unreadable, or has syntax errors\n"
    "\n"
    "2. **statistical_validity**: Is the statistical model sound and appropriate?\n"
    "   - 1.0: Well-specified priors, appropriate likelihood, valid model structure\n"
    "   - 0.5: Minor issues with priors or model specification\n"
    "   - 0.0: Invalid model, inappropriate distributions, or logical errors\n"
    "\n"
    "3. **problem_alignment**: Does the model address the stated problem effectively?\n"
    "   - 1.0: Directly models the phenomenon, captures key relationships\n"
    "   - 0.5: Partially addresses the problem\n"
    "   - 0.0: Does not address the problem or misses key aspects\n"
    "\n"
    "Respond ONLY in this exact format:\n"
    "SCORES:\n"
    "code_quality: <score>\n"
    "statistical_validity: <score>\n"
    "problem_alignment: <score>\n"
    "RATIONALE: <brief one-line explanation>"
)


class LLMCriticJudge:
    """Uses an LLM via LiteLLM to critique programs on code quality,
    statistical validity, and problem alignment."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.3,
        num_samples: int = 1,
        prompt_template: str | None = None,
        default_score: float = 0.5,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.num_samples = max(1, num_samples)
        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        self.default_score = default_score

    def _build_prompt(self, program: str, task_name: str) -> str:
        return self.prompt_template.format(task_name=task_name, program=program)

    def _parse_critique(self, response: str) -> dict[str, float]:
        result = {
            "code_quality": self.default_score,
            "statistical_validity": self.default_score,
            "problem_alignment": self.default_score,
        }

        scores_match = re.search(
            r"SCORES:\s*(.*?)(?:RATIONALE:|$)", response, re.DOTALL | re.IGNORECASE
        )
        if not scores_match:
            return result

        scores_text = scores_match.group(1)

        for key in result:
            pattern = rf"{key}:\s*(-?[\d.]+)"
            match = re.search(pattern, scores_text, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    result[key] = max(0.0, min(1.0, score))  # clamp to [0, 1]
                except ValueError:
                    pass

        return result

    def _call_llm(self, prompt: str) -> str | None:
        try:
            import litellm
        except ImportError:
            return None

        try:
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=256,
            )
            return response.choices[0].message.content
        except Exception:
            # API errors, missing keys, rate limits, etc.
            return None

    def score(self, *, task_name: str, trajectory: Trajectory, artifacts: dict[str, Any]) -> Reward:
        program = artifacts.get("program", "")

        if not program.strip():
            return Reward(
                total=0.0,
                components={
                    "code_quality": 0.0,
                    "statistical_validity": 0.0,
                    "problem_alignment": 0.0,
                },
                info={"error": "empty_program"},
            )

        prompt = self._build_prompt(program, task_name)

        all_scores: list[dict[str, float]] = []
        responses: list[str] = []

        for _ in range(self.num_samples):
            response = self._call_llm(prompt)
            if response:
                responses.append(response)
                scores = self._parse_critique(response)
                all_scores.append(scores)

        if not all_scores:
            return Reward(
                total=self.default_score,
                components={
                    "code_quality": self.default_score,
                    "statistical_validity": self.default_score,
                    "problem_alignment": self.default_score,
                },
                info={"error": "llm_call_failed"},
            )

        avg_scores = {
            "code_quality": sum(s["code_quality"] for s in all_scores) / len(all_scores),
            "statistical_validity": sum(s["statistical_validity"] for s in all_scores)
            / len(all_scores),
            "problem_alignment": sum(s["problem_alignment"] for s in all_scores) / len(all_scores),
        }

        total = sum(avg_scores.values()) / len(avg_scores)

        info: dict[str, Any] = {"num_samples": len(all_scores)}
        if len(responses) == 1:
            info["response"] = responses[0]

        return Reward(total=total, components=avg_scores, info=info)
