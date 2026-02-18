"""SFT data loading for GFlowNet warm-start."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from synthstats.core.constants import REWARD_FLOOR_DEFAULT

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
CODE_FENCE_PATTERN = re.compile(r"```python\s*(.*?)```", re.DOTALL)
SUBMIT_PROGRAM_PATTERN = re.compile(r"<submit_program>(.*?)</submit_program>", re.DOTALL)


@dataclass
class SFTExample:

    prompt: str
    completion: str
    thinking: str | None
    program: str
    source_file: str | None = None
    line_number: int | None = None

    @property
    def has_thinking(self) -> bool:
        return self.thinking is not None and len(self.thinking.strip()) > 0


def parse_completion(completion: str) -> tuple[str | None, str]:
    """Extract (thinking, program) from a completion string."""
    think_match = THINK_PATTERN.search(completion)
    thinking = think_match.group(1).strip() if think_match else None

    # prefer submit_program tags (BoxingGym format) over code fences
    program = ""
    submit_match = SUBMIT_PROGRAM_PATTERN.search(completion)
    if submit_match:
        program = submit_match.group(1).strip()
    else:
        code_match = CODE_FENCE_PATTERN.search(completion)
        program = code_match.group(1).strip() if code_match else ""

    return thinking, program


def load_sft_jsonl(
    path: Path,
    *,
    require_program: bool = True,
    max_examples: int | None = None,
) -> list[SFTExample]:
    """Load SFT examples from a JSONL file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SFT data file not found: {path}")

    examples: list[SFTExample] = []
    skipped = 0

    with open(path, encoding="utf-8-sig") as f:
        for line_num, line in enumerate(f, start=1):
            if max_examples is not None and len(examples) >= max_examples:
                break

            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON at line {line_num}: {e}")
                skipped += 1
                continue

            if "prompt" not in data or "completion" not in data:
                logger.warning(f"Missing keys at line {line_num}")
                skipped += 1
                continue

            prompt = data["prompt"]
            completion = data["completion"]
            thinking, program = parse_completion(completion)

            if require_program and not program:
                logger.debug(f"No program found at line {line_num}, skipping")
                skipped += 1
                continue

            examples.append(
                SFTExample(
                    prompt=prompt,
                    completion=completion,
                    thinking=thinking,
                    program=program,
                    source_file=str(path),
                    line_number=line_num,
                )
            )

    if skipped > 0:
        logger.info(f"Loaded {len(examples)} examples, skipped {skipped}")
    else:
        logger.info(f"Loaded {len(examples)} examples from {path}")

    return examples


def sft_to_buffer_entry(
    example: SFTExample,
    *,
    policy_version: int = 0,
    log_reward: float,
    strip_thinking: bool = False,
) -> Any:
    """Convert an SFT example to a BufferEntry."""
    if log_reward is None:
        raise TypeError(
            "log_reward must be a float, got None. "
            "Compute real ELPD reward using compute_sft_rewards() before calling."
        )

    from synthstats.train.data.replay import BufferEntry

    action_payload = example.program if strip_thinking else example.completion

    actions = [{"type": "submit_program", "payload": action_payload}]
    observations = [example.prompt]

    return BufferEntry(
        actions=actions,
        log_reward=log_reward,
        observations=observations,
        policy_version=policy_version,
        temperature=1.0,  # SFT data has no temperature
    )


def compute_sft_rewards(
    examples: list[SFTExample],
    reward_fn: Any,
    *,
    log_clamp: tuple[float, float] = (-50.0, 50.0),
    show_progress: bool = True,
) -> list[float]:
    """Compute clamped log rewards for SFT examples."""
    import math

    rewards = []
    iterator = examples
    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(examples, desc="Computing SFT rewards")
        except ImportError:
            pass

    for ex in iterator:
        try:
            reward = reward_fn(ex.program)
            log_r = math.log(max(reward, REWARD_FLOOR_DEFAULT))
            log_r = max(log_clamp[0], min(log_clamp[1], log_r))
        except Exception as e:
            logger.warning(f"Reward computation failed for {ex.source_file}:{ex.line_number}: {e}")
            log_r = log_clamp[0]  # use floor on failure

        rewards.append(log_r)

    return rewards
