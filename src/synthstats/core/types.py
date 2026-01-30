"""Core types for SynthStats.

These dataclasses define the shared vocabulary across all components:
- Message: LLM conversation messages
- Action: Structured actions (FinalAnswer, ToolCall, Program)
- StepResult: Result of a task step
- Reward: Multi-component reward signal
- Trajectory: Complete episode data for training
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    """A message in the conversation."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {"role": self.role, "content": self.content, "tool_call_id": self.tool_call_id}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Deserialize from dict."""
        return cls(
            role=data["role"], content=data["content"], tool_call_id=data.get("tool_call_id")
        )


@dataclass
class Action:
    """Base class for structured actions."""

    pass


@dataclass
class FinalAnswer(Action):
    """Terminal action with final text output."""

    text: str


@dataclass
class ToolCall(Action):
    """Action to invoke a tool."""

    name: str
    input: dict[str, Any]
    raw: str  # original text representation


@dataclass
class Program(Action):
    """Action to submit a probabilistic program."""

    code: str
    language: str = "pymc"  # "pymc" | "numpyro" | "lazyppl"


@dataclass
class StepResult:
    """Result of a single task step."""

    next_state: Any
    done: bool
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass
class Reward:
    """Multi-component reward signal."""

    total: float
    components: dict[str, float]
    info: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {"total": self.total, "components": self.components, "info": self.info}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Reward":
        """Deserialize from dict."""
        return cls(total=data["total"], components=data["components"], info=data["info"])


@dataclass
class Trajectory:
    """Complete episode data for training.

    Note on alignment:
    - messages: ALL messages in the conversation (system, user, assistant, tool)
    - token_ids/token_logprobs/loss_mask/eos_logprobs: ONLY assistant generations
      (one entry per generate() call)

    This asymmetry is intentional: we only have token-level data for what the policy generated.
    Use len(token_ids) for number of generations, len(messages) for full conversation length.

    Fields:
    - token_ids[i]: token IDs for assistant generation i
    - token_logprobs[i]: log probabilities for assistant generation i
    - loss_mask[i]: which tokens to include in loss (False = mask out, e.g., <think>)
    - eos_logprobs[i]: log p(EOS) at each step for SubTB flow matching
    """

    messages: list[Message]
    token_ids: list[list[int]]  # one entry per assistant generation
    token_logprobs: list[list[float]]  # one entry per assistant generation
    loss_mask: list[list[bool]]  # one entry per assistant generation
    reward: Reward
    # SubTB: EOS log probabilities for flow matching at sub-trajectory endpoints
    # Optional for backward compatibility with vanilla TB
    eos_logprobs: list[list[float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "messages": [m.to_dict() for m in self.messages],
            "token_ids": self.token_ids,
            "token_logprobs": self.token_logprobs,
            "loss_mask": self.loss_mask,
            "reward": self.reward.to_dict(),
            "eos_logprobs": self.eos_logprobs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Trajectory":
        """Deserialize from dict."""
        return cls(
            messages=[Message.from_dict(m) for m in data["messages"]],
            token_ids=data["token_ids"],
            token_logprobs=data["token_logprobs"],
            loss_mask=data["loss_mask"],
            reward=Reward.from_dict(data["reward"]),
            eos_logprobs=data.get("eos_logprobs", []),
        )
