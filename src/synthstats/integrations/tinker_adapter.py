"""Tinker API integration for distributed LLM training.

Tinker (https://thinkingmachines.ai/tinker/) is a training API from Thinking
Machines Lab that abstracts distributed LLM fine-tuning. This module provides
adapters to use Tinker with SynthStats' GFlowNet training.

This module is import-safe: works without Tinker installed for local testing.
Use `is_tinker_available()` before depending on Tinker-specific functionality.

Key components:
- TinkerPolicy: Wraps Tinker.sample() to match HFPolicy interface
- TinkerTrainer: Uses forward_backward_custom() with TB (Trajectory Balance) loss
- TinkerClient: Thin wrapper for Tinker training client
- trajectories_to_tinker_batch: Convert Trajectory objects to TinkerTrainer batch format

Usage:
    # check availability
    if is_tinker_available():
        policy = TinkerPolicy(model="Qwen/Qwen3-4B", api_key="...")
        action, logp, ent = policy("observation text")

    # convert trajectories for training
    from synthstats.integrations.tinker_adapter import trajectories_to_tinker_batch
    batch = trajectories_to_tinker_batch(trajectories, device="cuda")
    metrics = trainer.train_step(batch)
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch
    from torch import Tensor

    from synthstats.core.types import Message, Trajectory

logger = logging.getLogger(__name__)


class TinkerOptionalDependencyError(RuntimeError):
    """Raised when Tinker integration is requested but Tinker is not installed."""


def is_tinker_available() -> bool:
    """Return True if Tinker SDK is installed."""
    try:
        import tinker  # noqa: F401

        return True
    except ImportError:
        return False


def require_tinker() -> Any:
    """Import and return the tinker module, or raise helpful error."""
    try:
        import tinker

        return tinker
    except ImportError as e:
        raise TinkerOptionalDependencyError(
            "Tinker SDK is not installed. Install with:\n"
            "  pip install tinker\n"
            "Get API access at: https://thinkingmachines.ai/tinker/"
        ) from e


@dataclass
class TinkerConfig:
    """Configuration for Tinker integration.

    Attributes:
        model: Model name (e.g., "Qwen/Qwen3-4B", "meta-llama/Llama-3.1-8B")
        api_key: Tinker API key (or set TINKER_API_KEY env var)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        lora_rank: LoRA rank for fine-tuning (0 = no LoRA)
        learning_rate: Learning rate for training
    """

    model: str = "Qwen/Qwen3-4B"
    api_key: str | None = None
    max_tokens: int = 256
    temperature: float = 0.7
    lora_rank: int = 32
    learning_rate: float = 1e-5

    def get_api_key(self) -> str:
        """Get API key from config or environment."""
        key = self.api_key or os.environ.get("TINKER_API_KEY")
        if not key:
            raise ValueError(
                "Tinker API key required. Set TINKER_API_KEY env var or pass api_key."
            )
        return key


# type alias matching HFPolicy
PolicyOutput = tuple[dict[str, Any], float, float]


@dataclass
class TurnBoundary:
    """Character-level turn boundary in completion text.

    Used for multi-turn trajectory processing. Character positions are used
    instead of token positions because Tinker retokenizes the text, so original
    token boundaries may not align.

    Attributes:
        start_char: Start character index in completion text
        end_char: End character index in completion text
        role: Message role ("assistant", "user", or "tool")
        generation_idx: Index into Trajectory.token_ids/loss_mask for assistant
            turns (-1 for non-assistant turns)
        has_reward: True only for the final assistant turn (where terminal
            reward is placed)
    """

    start_char: int
    end_char: int
    role: str
    generation_idx: int
    has_reward: bool = False


def _extract_prompt(messages: list[Any]) -> str:
    """Extract prompt from messages before first assistant.

    Args:
        messages: List of Message objects

    Returns:
        Prompt text with newline delimiter at end
    """
    prompt_parts = []
    for msg in messages:
        if msg.role == "assistant":
            break
        if msg.role in ("system", "user"):
            prompt_parts.append(msg.content)
        elif msg.role == "tool":
            prompt_parts.append(f"[Tool Result]: {msg.content}")

    prompt_text = "\n".join(prompt_parts)
    if prompt_text and not prompt_text.endswith("\n"):
        prompt_text += "\n"
    return prompt_text


def _extract_single_turn_completion(messages: list[Any]) -> str:
    """Extract completion from assistant messages only (single-turn mode)."""
    parts = [m.content for m in messages if m.role == "assistant"]
    return "\n".join(parts)


def _extract_multi_turn_completion(
    messages: list[Any],
) -> tuple[str, list[TurnBoundary]]:
    """Extract multi-turn completion preserving full conversation structure.

    Builds completion text containing all messages after the first assistant,
    with turn boundaries tracking where each turn starts/ends.

    Args:
        messages: List of Message objects

    Returns:
        Tuple of (completion_text, turn_boundaries)
    """
    parts: list[str] = []
    boundaries: list[TurnBoundary] = []
    char_pos = 0
    assistant_idx = 0
    past_first_assistant = False
    last_assistant_idx: int | None = None

    for msg in messages:
        if msg.role == "assistant":
            past_first_assistant = True

        if not past_first_assistant:
            continue

        content = msg.content

        # add delimiter between turns for tokenization boundary
        if parts:
            parts.append("\n")
            char_pos += 1

        parts.append(content)

        boundaries.append(TurnBoundary(
            start_char=char_pos,
            end_char=char_pos + len(content),
            role=msg.role,
            generation_idx=assistant_idx if msg.role == "assistant" else -1,
            has_reward=False,
        ))

        char_pos += len(content)

        if msg.role == "assistant":
            last_assistant_idx = len(boundaries) - 1  # track last assistant
            assistant_idx += 1

    # mark final assistant turn as having reward
    if last_assistant_idx is not None:
        boundaries[last_assistant_idx].has_reward = True

    return "".join(parts), boundaries


def _build_turn_mask(
    completion: str,
    boundaries: list[TurnBoundary],
    tokenizer: Any,
    seq_len: int,
    device: Any,
    prompt_offset: int = 0,
) -> Any:
    """Build loss mask from turn boundaries.

    Maps character-level turn boundaries to token positions and creates a mask
    where only assistant turns are included (True). User and tool turns are
    masked out (False).

    Args:
        completion: The completion text
        boundaries: List of TurnBoundary objects with character positions
        tokenizer: Tokenizer with encode_plus method (for offset_mapping)
        seq_len: Length of the full token sequence (prompt + completion)
        device: Device for the output tensor
        prompt_offset: Number of prompt tokens to offset mask positions by.
            When Tinker tokenizes prompt+completion together, the logprobs
            include prompt tokens. This offset ensures we mark the correct
            positions (completion tokens, not prompt tokens).

    Returns:
        Boolean tensor of shape [seq_len] with True for assistant turn tokens
    """
    import torch

    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)

    # get token-to-character mapping for completion
    try:
        encoding = tokenizer.encode_plus(
            completion,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        offsets = encoding.get("offset_mapping", [])
    except (TypeError, AttributeError) as e:
        raise ValueError(
            f"Tokenizer {type(tokenizer).__name__} does not support offset_mapping. "
            "Cannot build turn mask without character-to-token mapping."
        ) from e

    # for each assistant turn, mark corresponding tokens
    for boundary in boundaries:
        if boundary.role != "assistant":
            continue

        # find token range for this turn's character range
        tok_start, tok_end = None, None
        for tok_idx, (char_start, char_end) in enumerate(offsets):
            # offset by prompt tokens, since logprobs include full text
            adjusted_idx = tok_idx + prompt_offset
            if adjusted_idx >= seq_len:
                break
            # token overlaps boundary if: token_end > boundary_start AND token_start < boundary_end
            if char_end > boundary.start_char and char_start < boundary.end_char:
                if tok_start is None:
                    tok_start = adjusted_idx
                tok_end = adjusted_idx + 1

        if tok_start is not None and tok_end is not None:
            mask[tok_start:tok_end] = True

    return mask


def trajectories_to_tinker_batch(
    trajectories: list[Any],  # list[Trajectory]
    device: str = "cpu",
    reward_floor: float = 1e-10,
    strict_single_turn: bool = True,
    multi_turn: bool = False,
) -> dict[str, Any]:
    """Convert SynthStats Trajectories to TinkerTrainer batch format.

    Supports both single-turn and multi-turn trajectories:

    Single-turn mode (default):
        Extracts prompt (pre-first-assistant) and completion (all assistant messages).
        Rejects multi-turn trajectories by default to prevent semantic corruption.

    Multi-turn mode (multi_turn=True):
        Preserves full conversation structure in completion text.
        Includes turn boundaries for per-turn loss computation in train_step.
        Only assistant turns contribute to the loss (user/tool turns are masked).

    Args:
        trajectories: List of Trajectory objects from SimpleCollector
        device: Device for tensors ("cpu" or "cuda")
        reward_floor: Minimum reward value to prevent log(0)
        strict_single_turn: If True (default), reject multi-turn when multi_turn=False.
            Ignored when multi_turn=True.
        multi_turn: Enable multi-turn mode with turn boundary extraction.

    Returns:
        Batch dict with keys:
        - prompts: list[str] - context before first assistant (ends with newline)
        - completions: list[str] - completion text (structure depends on mode)
        - log_reward: Tensor[B] - log of trajectory rewards
        - mask: Tensor[B, T] - loss mask (single-turn only, if any trajectory has mask)
        - is_multi_turn: bool - whether batch is in multi-turn mode
        - turn_boundaries: list[list[TurnBoundary]] - per-trajectory turn info (multi-turn only)

    Raises:
        ValueError: If strict_single_turn=True and trajectory has multiple assistant
            messages (when multi_turn=False)

    Example:
        >>> # Single-turn (default)
        >>> batch = trajectories_to_tinker_batch(trajectories, device="cuda")
        >>>
        >>> # Multi-turn
        >>> batch = trajectories_to_tinker_batch(trajectories, multi_turn=True)
        >>> assert batch["is_multi_turn"]
    """
    import torch

    prompts: list[str] = []
    completions: list[str] = []
    log_rewards: list[float] = []
    loss_masks: list[list[bool]] = []
    all_turn_boundaries: list[list[TurnBoundary]] = []
    has_any_mask = False

    for i, traj in enumerate(trajectories):
        # count assistant messages for validation
        assistant_count = sum(1 for m in traj.messages if m.role == "assistant")

        # validation: reject multi-turn in single-turn mode
        if not multi_turn and strict_single_turn and assistant_count > 1:
            raise ValueError(
                f"Trajectory {i} has {assistant_count} assistant messages. "
                "Use multi_turn=True for multi-turn trajectories, or set "
                "strict_single_turn=False to flatten (not recommended)."
            )

        # extract prompt (same for both modes)
        prompt_text = _extract_prompt(traj.messages)

        # extract completion based on mode
        if multi_turn and assistant_count > 1:
            # multi-turn: preserve full conversation structure
            completion_text, turn_boundaries = _extract_multi_turn_completion(
                traj.messages
            )
            all_turn_boundaries.append(turn_boundaries)
        else:
            # single-turn: just assistant messages
            completion_text = _extract_single_turn_completion(traj.messages)
            all_turn_boundaries.append([])

        prompts.append(prompt_text)
        completions.append(completion_text)

        # compute log reward, clamping to prevent -inf
        # Note: Reward.total from LikelihoodJudge is already exp(log_reward),
        # so we're converting back to log space for SubTB
        reward_val = max(traj.reward.total, reward_floor)
        log_rewards.append(math.log(reward_val))

        # extract loss_mask (single-turn only - multi-turn uses turn boundaries)
        if not multi_turn:
            if traj.loss_mask and len(traj.loss_mask) > 0 and traj.loss_mask[0]:
                loss_masks.append(traj.loss_mask[0])
                has_any_mask = True
            else:
                loss_masks.append([])

    # determine if this is actually a multi-turn batch
    is_multi_turn_batch = multi_turn and any(len(tb) > 0 for tb in all_turn_boundaries)

    result: dict[str, Any] = {
        "prompts": prompts,
        "completions": completions,
        "log_reward": torch.tensor(log_rewards, dtype=torch.float32, device=device),
        "is_multi_turn": is_multi_turn_batch,
    }

    # add turn boundaries for multi-turn batches
    if is_multi_turn_batch:
        result["turn_boundaries"] = all_turn_boundaries

    # build mask tensor for single-turn (multi-turn builds mask in train_step)
    if not is_multi_turn_batch and has_any_mask and loss_masks:
        max_mask_len = max(len(m) for m in loss_masks) if loss_masks else 0
        if max_mask_len > 0:
            mask_tensor = torch.ones(len(loss_masks), max_mask_len, dtype=torch.bool, device=device)
            for i, mask in enumerate(loss_masks):
                if mask:
                    mask_len = len(mask)
                    mask_tensor[i, :mask_len] = torch.tensor(mask, dtype=torch.bool)
            result["mask"] = mask_tensor

    return result


@dataclass
class TinkerPolicy:
    """Policy that uses Tinker API for sampling.

    Matches the HFPolicy interface so it can be used as a drop-in replacement
    in collectors and training loops.

    Uses Tinker's SamplingClient for generation and logprob computation.

    Args:
        config: TinkerConfig with model and API settings
        require_grad_logp: If True, returns tensors (not supported, raises error)

    Example:
        >>> policy = TinkerPolicy(TinkerConfig(model="Qwen/Qwen3-4B"))
        >>> action, logp, entropy = policy("What is 2+2?")
    """

    config: TinkerConfig
    require_grad_logp: bool = False
    _service_client: Any = field(default=None, init=False, repr=False)
    _sampling_client: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _tokenizer_checked: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.require_grad_logp:
            raise NotImplementedError(
                "TinkerPolicy does not support require_grad_logp=True. "
                "Use TinkerTrainer.forward_backward_custom() for gradient computation."
            )

    @property
    def service_client(self) -> Any:
        """Lazy-initialize Tinker service client."""
        if self._service_client is None:
            import os

            tinker = require_tinker()
            # set API key in environment (Tinker reads from env)
            os.environ["TINKER_API_KEY"] = self.config.get_api_key()
            self._service_client = tinker.ServiceClient()
        return self._service_client

    @property
    def sampling_client(self) -> Any:
        """Lazy-initialize Tinker sampling client."""
        if self._sampling_client is None:
            # create sampling client via service client
            self._sampling_client = self.service_client.create_sampling_client(
                base_model=self.config.model,
            )
            # load tokenizer from HuggingFace (Tinker's get_tokenizer returns None)
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.config.model)
            except Exception as e:
                logger.warning(f"Could not load tokenizer for {self.config.model}: {e}")
                self._tokenizer = None

        # check tokenizer compatibility on first real use
        if not self._tokenizer_checked and self._tokenizer is not None:
            self._check_tokenizer_compatibility()
            self._tokenizer_checked = True

        return self._sampling_client

    def __call__(
        self, obs: str, temperature: float | None = None
    ) -> PolicyOutput:
        """Generate action from observation.

        Args:
            obs: Observation text
            temperature: Sampling temperature (uses config default if None)

        Returns:
            Tuple of (action_dict, log_prob, entropy)
        """
        temp = temperature if temperature is not None else self.config.temperature
        prompt = self._build_prompt(obs)

        # access sampling_client to ensure tokenizer is loaded
        _ = self.sampling_client

        # tokenize prompt
        if self._tokenizer:
            prompt_tokens = self._tokenizer.encode(prompt)
        else:
            # fallback: use character-level approximation
            prompt_tokens = list(range(len(prompt)))

        # build model input - use Tinker types if available
        if is_tinker_available():
            tinker = require_tinker()
            model_input = tinker.ModelInput.from_ints(prompt_tokens)
            sampling_params = tinker.SamplingParams(
                max_tokens=self.config.max_tokens,
                temperature=temp,
            )
        else:
            # mock mode - pass tokens directly
            model_input = prompt_tokens
            sampling_params = {"max_tokens": self.config.max_tokens, "temperature": temp}

        # call Tinker sample API (returns Future, call .result())
        future = self.sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        )
        # get result from future (real API) or use directly (mock)
        result = future.result() if hasattr(future, 'result') else future

        # decode response
        gen_text = ""
        logprobs = None
        if hasattr(result, 'sequences') and result.sequences:
            seq = result.sequences[0]
            logprobs = getattr(seq, 'logprobs', None)
            gen_tokens = getattr(seq, 'tokens', [])
            if self._tokenizer and gen_tokens:
                gen_text = self._tokenizer.decode(gen_tokens, skip_special_tokens=True)
        # handle mock result format
        elif hasattr(result, 'text'):
            gen_text = result.text
            logprobs = getattr(result, 'logprobs', None)

        # parse response
        action = self._parse_action(gen_text)

        # extract logprob and entropy from result
        if logprobs:
            logp = sum(lp for lp in logprobs if lp is not None)
            entropy = self._estimate_entropy([lp for lp in logprobs if lp is not None])
        else:
            logp = -1.0
            entropy = 0.1

        return action, logp, entropy

    def score_action(
        self, obs: str, action: dict[str, Any]
    ) -> tuple[Any, Any]:
        """Score an action under the current policy.

        Uses Tinker's compute_logprobs API to get log probability of action.

        Args:
            obs: Observation text
            action: Action dict to score

        Returns:
            Tuple of (log_prob, entropy) as tensors
        """
        import torch

        prompt = self._build_prompt(obs)
        action_text = self._render_action(action)
        full_text = prompt + action_text

        # access sampling_client to ensure tokenizer is loaded
        _ = self.sampling_client

        # tokenize
        if self._tokenizer:
            full_tokens = self._tokenizer.encode(full_text)
            prompt_tokens = self._tokenizer.encode(prompt)
        else:
            full_tokens = list(range(len(full_text)))
            prompt_tokens = list(range(len(prompt)))

        # build model input - use Tinker types if available
        if is_tinker_available():
            tinker = require_tinker()
            model_input = tinker.ModelInput.from_ints(full_tokens)
        else:
            # mock mode - pass tokens directly
            model_input = full_tokens

        # use Tinker's compute_logprobs API (returns Future of list[float])
        future = self.sampling_client.compute_logprobs(model_input)
        # get result from future (real API) or use directly (mock)
        logprobs = future.result() if hasattr(future, 'result') else future
        # handle mock result object format
        if hasattr(logprobs, 'logprobs'):
            logprobs = logprobs.logprobs

        # extract logprobs for action tokens only
        action_start = len(prompt_tokens)
        if logprobs:
            action_logprobs = [lp for lp in logprobs[action_start:] if lp is not None]
            logp = sum(action_logprobs) if action_logprobs else -1.0
            entropy = self._estimate_entropy(action_logprobs)
        else:
            logp = -1.0
            entropy = 0.1

        # return as tensors for compatibility with training
        return (
            torch.tensor(logp, requires_grad=False),
            torch.tensor(entropy, requires_grad=False),
        )

    def _build_prompt(self, obs: str) -> str:
        """Build prompt from observation."""
        return (
            "You are an agent that responds to observations.\n"
            f"Observation: {obs}\n"
            "Respond with a JSON action: "
        )

    def _parse_action(self, text: str) -> dict[str, Any]:
        """Parse action from generated text."""
        import json

        text = text.strip()
        try:
            if "{" in text:
                start = text.index("{")
                end = text.rindex("}") + 1
                return json.loads(text[start:end])
        except (json.JSONDecodeError, ValueError):
            pass
        return {"type": "answer", "payload": text}

    def _render_action(self, action: dict[str, Any]) -> str:
        """Render action to text."""
        import json

        return json.dumps(action)

    @staticmethod
    def _estimate_entropy(logprobs: list[float]) -> float:
        """Estimate entropy from logprobs (lower bound)."""
        if not logprobs:
            return 0.0
        # average negative logprob is an entropy estimate
        return -sum(logprobs) / len(logprobs)

    def _check_tokenizer_compatibility(self, test_text: str = "Hello world") -> bool:
        """Warn if local tokenizer disagrees with Tinker's."""
        if self._tokenizer is None:
            return True

        try:
            local_tokens = self._tokenizer.encode(test_text, add_special_tokens=False)
        except TypeError:
            local_tokens = self._tokenizer.encode(test_text)
        local_count = len(local_tokens)

        try:
            if is_tinker_available() and hasattr(self.sampling_client, "tokenize"):
                tinker_result = self.sampling_client.tokenize(test_text)
                tinker_count = len(tinker_result) if tinker_result else local_count

                if abs(local_count - tinker_count) > 1:
                    logger.warning(
                        f"Tokenizer mismatch: HF={local_count} tokens, Tinker={tinker_count} "
                        f"for '{test_text}'. May cause logprob misalignment."
                    )
                    return False
        except Exception:
            pass

        return True


@dataclass
class TinkerTrainer:
    """Trainer that uses Tinker API for distributed training.

    Uses Tinker's forward_backward_custom() to compute TB (Trajectory Balance)
    loss with gradient accumulation across distributed workers.

    Args:
        config: TinkerConfig with model and training settings
        logZ_init: Initial value for learned log partition function

    Example:
        >>> trainer = TinkerTrainer(TinkerConfig(model="Qwen/Qwen3-4B"))
        >>> result = trainer.train_step(batch)
        >>> print(f"Loss: {result['loss']}, logZ: {result['logZ']}")
    """

    config: TinkerConfig
    logZ_init: float = 0.0
    _service_client: Any = field(default=None, init=False, repr=False)
    _training_client: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _logZ: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        import torch
        import torch.nn as nn

        # learned log partition function
        self._logZ = nn.Parameter(torch.tensor(self.logZ_init))

    @property
    def logZ(self) -> Any:
        """Access logZ parameter."""
        return self._logZ

    @property
    def service_client(self) -> Any:
        """Lazy-initialize Tinker service client."""
        if self._service_client is None:
            import os

            tinker = require_tinker()
            os.environ["TINKER_API_KEY"] = self.config.get_api_key()
            self._service_client = tinker.ServiceClient()
        return self._service_client

    @property
    def training_client(self) -> Any:
        """Lazy-initialize Tinker training client with LoRA."""
        if self._training_client is None:
            # create LoRA training client
            self._training_client = self.service_client.create_lora_training_client(
                base_model=self.config.model,
                rank=self.config.lora_rank if self.config.lora_rank > 0 else 32,
            )
            # get tokenizer from training client
            self._tokenizer = self._training_client.get_tokenizer()
        return self._training_client

    @property
    def tokenizer(self) -> Any:
        """Access tokenizer (initializes training client if needed)."""
        _ = self.training_client  # ensure initialized
        return self._tokenizer

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Execute one training step with TB (Trajectory Balance) loss.

        Supports both single-turn and multi-turn batches. Multi-turn batches
        use turn boundaries to build masks that only include assistant turns.

        Args:
            batch: Training batch with keys:
                - log_reward: [B] log rewards
                - prompts: list[str] prompts (for Tinker API)
                - completions: list[str] completions (for Tinker API)
                - mask: [B, T] optional think-block mask (single-turn only)
                - is_multi_turn: bool, whether this is a multi-turn batch
                - turn_boundaries: list[list[TurnBoundary]] (multi-turn only)

        Returns:
            Dict with 'loss', 'logZ', 'is_multi_turn', and other metrics

        Raises:
            ValueError: If batch is malformed (missing keys, misaligned sizes)
        """
        import torch

        from synthstats.training.losses.tb_loss import subtb_loss

        # validate batch before processing
        self._validate_batch(batch)

        # extract multi-turn metadata
        is_multi_turn = batch.get("is_multi_turn", False)
        turn_boundaries = batch.get("turn_boundaries", [])
        completions = batch["completions"]
        prompts = batch["prompts"]

        # prepare data for Tinker API - create proper Datum objects
        # following tinker_cookbook pattern: shifted tokens for next-token prediction
        tinker = require_tinker()
        tokenizer = self.training_client.get_tokenizer()

        data = []
        log_rewards_list = batch["log_reward"].tolist()
        for prompt, completion, log_r in zip(
            batch["prompts"], batch["completions"], log_rewards_list, strict=True
        ):
            full_text = prompt + completion
            # use add_special_tokens=False to match offset calculation in custom_subtb_loss
            tokens = tokenizer.encode(full_text, add_special_tokens=False)

            # shift tokens: input = tokens[:-1], target = tokens[1:]
            input_tokens = tokens[:-1]
            target_tokens = tokens[1:]
            seq_len = len(target_tokens)

            model_input = tinker.ModelInput.from_ints(input_tokens)
            # Tinker API requires only 'weights' and 'target_tokens' in loss_fn_inputs
            # following datum_from_tokens_weights pattern from tinker_cookbook
            loss_inputs = {
                "weights": tinker.TensorData.from_torch(
                    torch.ones(seq_len, dtype=torch.float32)
                ),
                "target_tokens": tinker.TensorData.from_torch(
                    torch.tensor(target_tokens, dtype=torch.int64)
                ),
            }
            datum = tinker.Datum(model_input=model_input, loss_fn_inputs=loss_inputs)
            data.append(datum)

        # extract from batch - mask may not match Tinker's token count
        log_rewards = batch["log_reward"]
        original_mask = batch.get("mask")  # may be None or wrong size
        logZ = self._logZ

        def custom_tb_loss(
            data: list[dict[str, str]], logprobs_list: list[Any]
        ) -> tuple[Any, dict[str, float]]:
            """Custom loss function for Tinker's forward_backward_custom.

            Tinker provides fresh logprobs from current policy, we use them
            to compute TB (Trajectory Balance) loss. Supports both single-turn
            and multi-turn trajectories.
            """
            # stack logprobs into [B, T] tensor, padding if needed
            max_len = max(lp.shape[0] for lp in logprobs_list)
            batch_size = len(logprobs_list)
            device = logprobs_list[0].device

            log_probs = torch.zeros(batch_size, max_len, device=device)
            mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)

            # pre-compute prompt token counts for offset calculation
            prompt_offsets = []
            for prompt in prompts:
                prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
                # subtract 1 for the shift: target tokens start at position 1
                # so completion starts at position len(prompt_tokens) - 1 in targets
                prompt_offsets.append(max(0, len(prompt_tokens) - 1))

            for i, lp in enumerate(logprobs_list):
                seq_len = lp.shape[0]
                log_probs[i, :seq_len] = lp

                if is_multi_turn and i < len(turn_boundaries) and turn_boundaries[i]:
                    # multi-turn: build mask from turn boundaries
                    # only assistant turns contribute to loss
                    # offset by prompt tokens since logprobs include full text
                    turn_mask = _build_turn_mask(
                        completions[i],
                        turn_boundaries[i],
                        tokenizer,
                        seq_len,
                        device,
                        prompt_offset=prompt_offsets[i],
                    )
                    mask[i, :seq_len] = turn_mask
                else:
                    # single-turn: include all non-padding tokens
                    mask[i, :seq_len] = True

                    # apply original mask if available (e.g., think-block exclusion)
                    if original_mask is not None and i < original_mask.shape[0]:
                        orig_seq_len = original_mask.shape[1]
                        overlap = min(orig_seq_len, seq_len)
                        mask[i, :overlap] &= original_mask[i, :overlap].to(device)

            # compute TB loss
            loss = subtb_loss(
                log_probs=log_probs,
                loss_mask=mask,
                log_rewards=log_rewards,
                logZ=logZ,
            )

            metrics = {
                "loss": loss.item(),
                "logZ": logZ.item(),
                "mean_log_reward": log_rewards.mean().item(),
                "is_multi_turn": float(is_multi_turn),
            }

            return loss, metrics

        # call Tinker's forward_backward_custom
        future = self.training_client.forward_backward_custom(data, custom_tb_loss)
        result = future.result()

        # step the optimizer with Adam params
        adam_params = tinker.AdamParams(
            learning_rate=self.config.learning_rate,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
        )
        self.training_client.optim_step(adam_params)

        # extract loss from result.metrics (Tinker returns loss in metrics dict)
        loss_value = result.metrics.get("loss:sum", 0.0)

        return {
            "loss": loss_value,
            "logZ": self._logZ.item(),
            **result.metrics,
        }

    def parameters(self) -> list[Any]:
        """Return trainable parameters (just logZ for local updates)."""
        return [self._logZ]

    def save_checkpoint(self, path: str) -> None:
        """Save trainer state to checkpoint."""
        import torch

        checkpoint = {
            "logZ": self._logZ.data.clone(),
            "config": {
                "model": self.config.model,
                "lora_rank": self.config.lora_rank,
                "learning_rate": self.config.learning_rate,
            },
        }

        if self._training_client is not None:
            try:
                if hasattr(self._training_client, "save_state"):
                    # save_state requires a name argument
                    import os
                    state_name = os.path.splitext(os.path.basename(path))[0]
                    checkpoint["tinker_state"] = self._training_client.save_state(state_name)
            except Exception as e:
                logger.warning(f"Could not save Tinker state: {e}")

        torch.save(checkpoint, path)
        logger.info(f"Saved TinkerTrainer checkpoint to {path}")

    def load_checkpoint(self, path: str, strict: bool = True) -> dict[str, Any]:
        """Load trainer state from checkpoint."""
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        saved_config = checkpoint.get("config", {})
        if strict and saved_config.get("model") != self.config.model:
            raise ValueError(
                f"Model mismatch: checkpoint={saved_config.get('model')}, "
                f"config={self.config.model}"
            )

        if "logZ" not in checkpoint:
            raise ValueError(f"Checkpoint missing 'logZ': {path}")
        saved_logZ = checkpoint["logZ"]
        if not hasattr(saved_logZ, "data"):
            raise ValueError(f"logZ must be Tensor, got {type(saved_logZ).__name__}")
        self._logZ.data.copy_(saved_logZ)
        logger.info(f"Restored logZ={self._logZ.item():.4f} from {path}")

        if "tinker_state" in checkpoint and self._training_client is not None:
            try:
                if hasattr(self._training_client, "load_state"):
                    self._training_client.load_state(checkpoint["tinker_state"])
                    logger.info("Restored Tinker training state from checkpoint")
            except Exception as e:
                logger.warning(f"Could not restore Tinker training state: {e}")

        return checkpoint

    def _validate_batch(self, batch: dict[str, Any]) -> None:
        """Check batch has required keys and consistent shapes."""
        required_keys = ["prompts", "completions", "log_reward"]
        for key in required_keys:
            if key not in batch:
                raise ValueError(f"Batch missing '{key}'")

        prompts = batch["prompts"]
        completions = batch["completions"]
        log_reward = batch["log_reward"]

        if len(prompts) == 0:
            raise ValueError("Batch cannot be empty")

        if len(prompts) != len(completions):
            raise ValueError(f"Size mismatch: {len(prompts)} prompts vs {len(completions)} completions")

        if hasattr(log_reward, "__len__") and len(log_reward) != len(prompts):
            raise ValueError(f"Size mismatch: {len(prompts)} prompts vs {len(log_reward)} rewards")

        if not all(isinstance(p, str) for p in prompts):
            raise ValueError("Prompts must be strings")
        if not all(isinstance(c, str) for c in completions):
            raise ValueError("Completions must be strings")


@runtime_checkable
class TinkerEnvProtocol(Protocol):
    """Protocol matching Tinker Cookbook's Env interface.

    SynthStats Task can be adapted to this interface for use with
    Tinker's RL training recipes.
    """

    def initial_observation(self) -> str:
        """Return initial observation/prompt."""
        ...

    async def step(self, action: Any) -> Any:
        """Process action and return result."""
        ...


@dataclass
class MockTinkerClient:
    """Mock Tinker client for testing without API access.

    Provides the same interface as TinkerPolicy/TinkerTrainer but returns
    fixed/random values. Useful for unit tests.
    """

    model: str = "mock-model"

    def sample(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> Any:
        """Return mock sample result."""

        @dataclass
        class MockSampleResult:
            text: str = '{"type": "answer", "payload": "42"}'
            logprobs: list[float] = field(
                default_factory=lambda: [-0.1, -0.2, -0.15, -0.1, -0.2]
            )

        return MockSampleResult()

    def logprobs(self, prompt: str, completion: str) -> Any:
        """Return mock logprobs result."""

        @dataclass
        class MockLogprobsResult:
            logprobs: list[float] = field(
                default_factory=lambda: [-0.1] * len(completion.split())
            )

        return MockLogprobsResult()


@dataclass
class MockTokenizer:
    """Mock tokenizer for testing."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Simple word-level tokenization for testing."""
        return list(range(len(text.split())))

    def encode_plus(self, text: str, **kwargs) -> dict:
        """Return offset_mapping treating each character as a token."""
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}

    def decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
        """Mock decode - just return placeholder."""
        return " ".join(["word"] * len(tokens))


@dataclass
class MockTinkerTrainingClient:
    """Mock Tinker training client for testing."""

    model: str = "mock-model"
    learning_rate: float = 1e-5
    lora_rank: int | None = 32
    _step_count: int = field(default=0, init=False)
    _tokenizer: MockTokenizer = field(default_factory=MockTokenizer, init=False)

    def get_tokenizer(self) -> MockTokenizer:
        """Return mock tokenizer."""
        return self._tokenizer

    def forward_backward_custom(
        self, data: list[Any], loss_fn: Any
    ) -> Any:
        """Return mock forward_backward result."""
        import torch

        # create mock logprobs with variable length based on model_input
        logprobs_list = []
        for d in data:
            # handle Datum objects or dicts
            if hasattr(d, "model_input"):
                # Datum object - get length from model_input
                n_tokens = d.model_input.length if hasattr(d.model_input, "length") else 10
            else:
                # fallback for dict format
                completion = d.get("completion", "")
                n_tokens = max(1, len(completion.split()))
            logprobs_list.append(
                torch.tensor([-0.1] * n_tokens)
            )

        # call the actual loss function
        loss, metrics = loss_fn(data, logprobs_list)

        # capture values for closure
        loss_value = loss.item() if hasattr(loss, "item") else float(loss)
        metrics_copy = dict(metrics)

        @dataclass
        class MockFuture:
            _loss: float = loss_value
            _metrics: dict[str, float] = field(default_factory=lambda: metrics_copy)

            def result(self) -> Any:
                @dataclass
                class MockResult:
                    loss: float
                    metrics: dict[str, float]

                return MockResult(loss=self._loss, metrics=self._metrics)

        return MockFuture()

    def optim_step(self, adam_params: Any = None) -> None:
        """Mock optimizer step."""
        self._step_count += 1
