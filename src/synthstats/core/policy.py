"""Policy protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from synthstats.core.types import Message


@dataclass
class GenConfig:
    """Configuration for generation."""

    max_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)


@dataclass
class Generation:
    """Result of a generation call."""

    text: str
    token_ids: list[int]
    token_logprobs: list[float]
    finish_reason: str  # "stop" | "length" | "tool_call"
    # SubTB flow matching: log p(EOS) at each generation step
    # Needed for computing per-step termination flow in SubTB loss
    eos_logprobs: list[float] = field(default_factory=list)


@dataclass
class TokenLogProbs:
    """Log probabilities for a sequence of tokens."""

    token_ids: list[int]
    logprobs: list[float]


@runtime_checkable
class Policy(Protocol):
    """LLM wrapper interface.

    Must support:
    - Sampling completions
    - Returning per-token logprobs
    - Optional thought masking
    - Optional flow head (for SubTB variants)
    """

    def generate(self, messages: list[Message], *, gen: GenConfig) -> Generation:
        """Generate a completion for the given messages.

        Args:
            messages: Conversation history.
            gen: Generation configuration.

        Returns:
            Generation result with text, tokens, and logprobs.
        """
        ...

    def logprobs(self, messages: list[Message], tokens: list[int]) -> TokenLogProbs:
        """Compute log probabilities for a token sequence.

        Args:
            messages: Conversation context.
            tokens: Token IDs to compute logprobs for.

        Returns:
            TokenLogProbs with per-token log probabilities.
        """
        ...
