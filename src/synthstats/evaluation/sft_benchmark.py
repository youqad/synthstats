"""SFT benchmark evaluation for GFlowNet policies."""

from __future__ import annotations

import ast
import logging
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from synthstats.data.sft_loader import SFTExample

logger = logging.getLogger(__name__)

CODE_FENCE_PATTERN = re.compile(r"```python\s*(.*?)```", re.DOTALL)
SUBMIT_PROGRAM_PATTERN = re.compile(r"<submit_program>(.*?)</submit_program>", re.DOTALL)


@dataclass
class BenchmarkResult:
    num_examples: int
    exact_match: float
    program_match: float
    mean_reward: float
    reward_std: float
    generation_success: float
    per_example: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, float]:
        return {
            "eval/num_examples": float(self.num_examples),
            "eval/exact_match": self.exact_match,
            "eval/program_match": self.program_match,
            "eval/mean_reward": self.mean_reward,
            "eval/reward_std": self.reward_std,
            "eval/generation_success": self.generation_success,
        }


@dataclass
class _ExampleEvalState:
    best_reward: float = -float("inf")
    is_exact: bool = False
    is_program_match: bool = False
    generated_program: str | None = None


def normalize_code(code: str) -> str:
    """Strip whitespace and blank lines for comparison."""
    lines = code.strip().split("\n")
    return "\n".join(line.rstrip() for line in lines if line.strip())


def extract_program(text: str) -> str | None:
    # prefer BoxingGym submit tags when present
    submit_match = SUBMIT_PROGRAM_PATTERN.search(text)
    if submit_match:
        return submit_match.group(1).strip()

    code_match = CODE_FENCE_PATTERN.search(text)
    if code_match:
        return code_match.group(1).strip()

    return None


def _iter_examples(examples: list[SFTExample], *, show_progress: bool) -> Any:
    if not show_progress:
        return examples

    try:
        from tqdm import tqdm

        return tqdm(examples, desc="Evaluating SFT benchmark")
    except ImportError:
        return examples


def _action_to_text(action: Any) -> str:
    if isinstance(action, dict):
        payload = action.get("payload")
        if isinstance(payload, str):
            return payload
        if payload is not None:
            return str(payload)
        return str(action)
    return str(action)


def _extract_program_from_action(action: Any) -> str | None:
    if not isinstance(action, dict):
        return None

    if action.get("type") != "submit_program":
        return None

    payload = action.get("payload")
    if not isinstance(payload, str):
        return None

    payload = payload.strip()
    if not payload:
        return None

    extracted = extract_program(payload)
    if extracted is not None:
        return extracted

    try:
        parsed = ast.parse(payload)
    except SyntaxError:
        return None

    if len(parsed.body) == 1 and isinstance(parsed.body[0], ast.Expr):
        expr = parsed.body[0].value
        if isinstance(expr, (ast.Name, ast.Constant)):
            return None

    return payload


def _score_program(judge: Any | None, program: str) -> float:
    if judge is None:
        return 0.0

    try:
        reward_obj = judge.score(
            task_name="sft_eval",
            trajectory=None,
            artifacts={"program": program},
        )
        if hasattr(reward_obj, "total"):
            return float(reward_obj.total)
        return float(reward_obj)
    except Exception as e:
        logger.debug("Judge scoring failed: %s", e)
        return 0.0


def _update_example_state(
    *,
    state: _ExampleEvalState,
    example: SFTExample,
    generated_text: str,
    generated_program: str,
    reward: float,
) -> None:
    if reward > state.best_reward:
        state.best_reward = reward
        state.generated_program = generated_program

    if generated_text.strip() == example.completion.strip():
        state.is_exact = True

    if example.program and normalize_code(generated_program) == normalize_code(example.program):
        state.is_program_match = True


def _evaluate_example(
    *,
    policy: Any,
    example: SFTExample,
    judge: Any | None,
    num_samples: int,
    temperature: float,
) -> _ExampleEvalState:
    state = _ExampleEvalState()

    for _ in range(num_samples):
        try:
            action, _logp, _entropy = policy(example.prompt, temperature=temperature)
        except Exception as e:
            logger.debug("Generation failed: %s", e)
            continue

        generated_text = _action_to_text(action)
        generated_program = _extract_program_from_action(action)
        if generated_program is None:
            generated_program = extract_program(generated_text)
        if generated_program is None:
            continue

        reward = _score_program(judge, generated_program)
        _update_example_state(
            state=state,
            example=example,
            generated_text=generated_text,
            generated_program=generated_program,
            reward=reward,
        )

    return state


def _effective_reward(best_reward: float) -> float:
    if best_reward > -float("inf"):
        return best_reward
    return 0.0


def _reward_stats(rewards: list[float]) -> tuple[float, float]:
    if not rewards:
        return 0.0, 0.0

    mean_reward = sum(rewards) / len(rewards)
    variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
    return mean_reward, math.sqrt(variance)


def _per_example_row(
    example: SFTExample,
    state: _ExampleEvalState,
    reward: float,
) -> dict[str, Any]:
    return {
        "prompt_preview": example.prompt[:100],
        "reference_program": example.program[:200] if example.program else None,
        "generated_program": state.generated_program[:200] if state.generated_program else None,
        "reward": reward,
        "is_exact": state.is_exact,
        "is_program_match": state.is_program_match,
    }


def evaluate_on_sft(
    policy: Any,
    examples: list[SFTExample],
    judge: Any | None = None,
    *,
    num_samples: int = 1,
    temperature: float = 0.7,
    store_per_example: bool = False,
    max_examples: int | None = None,
    show_progress: bool = True,
) -> BenchmarkResult:
    """Evaluate policy on SFT benchmark, keeping best-of-n by reward."""
    if max_examples is not None:
        examples = examples[:max_examples]

    iterator = _iter_examples(examples, show_progress=show_progress)

    exact_matches = 0
    program_matches = 0
    rewards: list[float] = []
    generation_successes = 0
    per_example_results: list[dict[str, Any]] = []

    for example in iterator:
        state = _evaluate_example(
            policy=policy,
            example=example,
            judge=judge,
            num_samples=num_samples,
            temperature=temperature,
        )
        reward = _effective_reward(state.best_reward)

        if state.generated_program is not None:
            generation_successes += 1

        if state.is_exact:
            exact_matches += 1

        if state.is_program_match:
            program_matches += 1

        rewards.append(reward)

        if store_per_example:
            per_example_results.append(_per_example_row(example, state, reward))

    n = len(examples)
    mean_reward, std_reward = _reward_stats(rewards)

    return BenchmarkResult(
        num_examples=n,
        exact_match=exact_matches / n if n > 0 else 0.0,
        program_match=program_matches / n if n > 0 else 0.0,
        mean_reward=mean_reward,
        reward_std=std_reward,
        generation_success=generation_successes / n if n > 0 else 0.0,
        per_example=per_example_results if store_per_example else None,
    )
