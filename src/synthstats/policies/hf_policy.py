"""HuggingFace LM policy with LoRA support and correct logprob accounting.

This module provides:
- HFPolicy: Full implementation with HuggingFace model loading
- MockHFPolicy: Lightweight mock for testing without model loading
- MockPolicy: Alias for MockHFPolicy (for modeling compatibility)

HFPolicy features:
- Loads a causal LM + tokenizer from HuggingFace
- Computes per-token logprobs and entropy
- Supports LoRA fine-tuning via PEFT
- Supports 4-bit quantization via BitsAndBytes
- Gradient checkpointing for memory efficiency

Interfaces:
- SkyRL-style: `__call__(obs, temperature)` -> (action_dict, logp, entropy)
- Protocol-style: `generate(messages, gen=GenConfig)` -> Generation
                  `logprobs(messages, tokens)` -> TokenLogProbs

Usage:
    # For testing (no model loading)
    policy = MockHFPolicy()
    action, logp, ent = policy("Observation text")

    # For real training
    policy = HFPolicy(model_name="Qwen/Qwen3-0.6B", device="cuda")
    action, logp, ent = policy("Observation text")

    # Protocol-style (for Trainer integration)
    from synthstats.core.policy import GenConfig
    gen = policy.generate(messages, gen=GenConfig())
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from synthstats.core.policy import GenConfig, Generation, TokenLogProbs
from synthstats.core.types import Message

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# type alias for policy output
PolicyOutput = tuple[dict[str, Any], float | torch.Tensor, float | torch.Tensor]


class MockHFPolicy:
    """Mock policy for testing without loading actual models.

    This provides the same interface as HFPolicy but returns random/fixed
    outputs. Useful for unit tests and integration tests.

    Supports both interfaces:
    - SkyRL-style: `__call__(obs, temperature)` -> (action_dict, logp, entropy)
    - Protocol-style: `generate(messages, gen=GenConfig)` -> Generation

    Args:
        require_grad_logp: If True, returns tensors with requires_grad=True
        device: Device for output tensors
        fixed_text: Fixed text to return from generate()
        fixed_token_ids: Fixed token IDs to return from generate()
        fixed_token_logprobs: Fixed token logprobs to return from generate()
        fixed_eos_logprobs: Fixed EOS logprobs for SubTB flow matching
        fixed_finish_reason: Fixed finish reason to return from generate()
    """

    def __init__(
        self,
        require_grad_logp: bool = False,
        device: str = "cpu",
        fixed_text: str = "This is a mock response.",
        fixed_token_ids: list[int] | None = None,
        fixed_token_logprobs: list[float] | None = None,
        fixed_eos_logprobs: list[float] | None = None,
        fixed_finish_reason: str = "stop",
    ) -> None:
        self.require_grad_logp = require_grad_logp
        self.device = device

        # fixed outputs for Protocol-style interface
        self._fixed_text = fixed_text
        self._fixed_token_ids = fixed_token_ids or [101, 102, 103, 104, 105]
        self._fixed_token_logprobs = fixed_token_logprobs or [-0.1, -0.2, -0.15, -0.25, -0.1]
        self._fixed_finish_reason = fixed_finish_reason

        # EOS logprobs for SubTB flow matching
        # auto-generate to match token_ids length if not explicitly provided
        if fixed_eos_logprobs is not None:
            self._fixed_eos_logprobs = fixed_eos_logprobs
        else:
            # generate realistic EOS logprobs: starts low (unlikely to stop), increases toward end
            n = len(self._fixed_token_ids)
            self._fixed_eos_logprobs = [-5.0 + 4.9 * i / max(n - 1, 1) for i in range(n)]

        # ensure alignment invariant
        if len(self._fixed_token_ids) != len(self._fixed_token_logprobs):
            raise ValueError(
                f"token_ids length ({len(self._fixed_token_ids)}) must match "
                f"token_logprobs length ({len(self._fixed_token_logprobs)})"
            )
        if len(self._fixed_token_ids) != len(self._fixed_eos_logprobs):
            raise ValueError(
                f"token_ids length ({len(self._fixed_token_ids)}) must match "
                f"eos_logprobs length ({len(self._fixed_eos_logprobs)})"
            )

        # dummy parameter for optimizer (so policy.parameters() works)
        self._dummy = nn.Parameter(torch.zeros(1, device=device))

        # SubTB: track last EOS logprob for SimpleCollector
        self._last_eos_logprob_final: float | None = None

    def __call__(
        self, obs: str, temperature: float | None = None
    ) -> PolicyOutput:
        """Generate an action from observation.

        Args:
            obs: Observation text
            temperature: Sampling temperature (ignored in mock)

        Returns:
            Tuple of (action_dict, log_prob, entropy)
        """
        # return fixed action with random logp/ent
        action = {"type": "answer", "payload": "42"}

        logp: float | torch.Tensor
        ent: float | torch.Tensor
        if self.require_grad_logp:
            logp = torch.tensor(-0.5, device=self.device, requires_grad=True)
            ent = torch.tensor(0.1, device=self.device, requires_grad=True)
        else:
            logp = -0.5
            ent = 0.1

        # store final EOS logprob for SubTB flow matching
        # (last element of eos_logprobs from the most recent generation)
        self._last_eos_logprob_final = self._fixed_eos_logprobs[-1]

        return action, logp, ent

    def score_action(
        self, obs: str, action: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score an action under the current policy.

        Args:
            obs: Observation text
            action: Action dict to score

        Returns:
            Tuple of (log_prob, entropy) as tensors
        """
        logp = torch.tensor(-0.5, device=self.device, requires_grad=True)
        ent = torch.tensor(0.1, device=self.device, requires_grad=True)
        return logp, ent

    def parameters(self) -> Iterator[nn.Parameter]:
        """Return model parameters for optimizer."""
        yield self._dummy

    def named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        """Return named parameters for gradient verification."""
        yield ("dummy", self._dummy)

    # -------------------- Protocol-style interface --------------------

    def generate(self, messages: list[Message], *, gen: GenConfig) -> Generation:
        """Return fixed generation output (Protocol-style interface).

        Args:
            messages: Ignored in mock.
            gen: Ignored in mock.

        Returns:
            Fixed Generation with aligned token_ids, token_logprobs, and eos_logprobs.
        """
        return Generation(
            text=self._fixed_text,
            token_ids=self._fixed_token_ids.copy(),
            token_logprobs=self._fixed_token_logprobs.copy(),
            finish_reason=self._fixed_finish_reason,
            eos_logprobs=self._fixed_eos_logprobs.copy(),
        )

    def logprobs(self, messages: list[Message], tokens: list[int]) -> TokenLogProbs:
        """Return fixed log probabilities for given tokens.

        Args:
            messages: Ignored in mock.
            tokens: Token IDs to compute logprobs for.

        Returns:
            TokenLogProbs with deterministic negative values.
        """
        # return slightly negative logprobs proportional to token index
        logprobs = [-0.1 * (i + 1) for i in range(len(tokens))]
        return TokenLogProbs(token_ids=tokens.copy(), logprobs=logprobs)

    def score_tokens(self, messages: list[Message], tokens: list[int]) -> torch.Tensor:
        """Return differentiable log probabilities (mock version).

        Returns a tensor with requires_grad=True for gradient flow testing.
        """
        if not tokens:
            return torch.tensor([], device=self.device, requires_grad=True)

        # create tensor with gradient tracking
        logprobs = torch.tensor(
            [-0.1 * (i + 1) for i in range(len(tokens))],
            device=self.device,
            requires_grad=True,
        )
        return logprobs


# backward compatibility alias
MockPolicy = MockHFPolicy


class HFPolicy:
    """HuggingFace LM policy with LoRA support.

    This policy:
    - Loads a causal LM + tokenizer
    - Generates text and parses into action dicts
    - Computes per-token logprobs and entropy from generation
    - Optionally applies LoRA for parameter-efficient fine-tuning

    Args:
        model_name: HuggingFace model name
        device: Device to load model on
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        require_grad_logp: If True, returns tensors with gradient flow
        lora_config: Optional LoRA configuration dict
        use_4bit: Use 4-bit quantization
        gradient_checkpointing: Enable gradient checkpointing

    Example:
        >>> policy = HFPolicy(model_name="Qwen/Qwen3-0.6B", device="cuda")
        >>> action, logp, ent = policy("What is 2+2?")
        >>> print(action)  # {"type": "answer", "payload": "4"}
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: str = "cpu",
        max_new_tokens: int = 96,
        temperature: float = 0.7,
        top_p: float = 0.9,
        require_grad_logp: bool = False,
        lora_config: dict[str, Any] | None = None,
        use_4bit: bool = False,
        gradient_checkpointing: bool = False,
        dtype: str = "float16",
    ) -> None:
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.require_grad_logp = require_grad_logp
        self.lora_config = lora_config
        self.use_4bit = use_4bit
        self.gradient_checkpointing = gradient_checkpointing

        # lazy imports for optional dependencies
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers is required for HFPolicy. "
                "Install with: pip install transformers"
            ) from e

        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)

        # determine dtype
        if use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                logger.info("Loading model with 4-bit quantization")
            except ImportError as e:
                raise ImportError(
                    "bitsandbytes is required for 4-bit quantization. "
                    "Install with: pip install bitsandbytes"
                ) from e
            torch_dtype = torch.bfloat16
            load_kwargs = {
                "quantization_config": quantization_config,
                "device_map": "auto",
                "torch_dtype": torch_dtype,
            }
        else:
            dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
            torch_dtype = dtype_map.get(dtype, torch.bfloat16) if device != "cpu" else torch.float32
            load_kwargs = {"torch_dtype": torch_dtype}

        self.model: Any = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        if "device_map" not in load_kwargs:
            self.model = self.model.to(device)  # type: ignore[arg-type]

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # apply LoRA if configured
        if lora_config is not None:
            self._apply_lora(lora_config)

        # gradient checkpointing
        if gradient_checkpointing and not (use_4bit and lora_config is not None):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        # set training mode if needed
        if require_grad_logp or lora_config is not None:
            self.model.train()
        else:
            self.model.eval()

        # SubTB: track last EOS logprob for SimpleCollector
        self._last_eos_logprob_final: float | None = None

    def _apply_lora(self, lora_config: dict[str, Any]) -> None:
        """Apply LoRA adapter using PEFT."""
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError as e:
            raise ImportError(
                "peft is required for LoRA training. Install with: pip install peft"
            ) from e

        if self.use_4bit:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.gradient_checkpointing,
            )
            logger.info("Model prepared for k-bit training")

        peft_config = LoraConfig(
            r=lora_config.get("r", 32),
            lora_alpha=lora_config.get("alpha", 64),
            target_modules=lora_config.get(
                "target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]
            ),
            lora_dropout=lora_config.get("dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        logger.info("LoRA adapter applied")

    def __call__(
        self, obs: str, temperature: float | None = None
    ) -> PolicyOutput:
        """Generate an action from observation.

        Args:
            obs: Observation text
            temperature: Optional sampling temperature (uses self.temperature if None)

        Returns:
            Tuple of (action_dict, log_prob, entropy)
        """
        import torch.nn.functional as F

        prompt = self._build_prompt(obs)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = int(inputs.input_ids.shape[1])

        temp = self.temperature if temperature is None else float(temperature)

        generation = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=temp,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

        seq = generation.sequences
        gen_ids = seq[:, prompt_len:]
        gen_text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

        if self.require_grad_logp:
            logp, ent = self._score_generated(seq, prompt_len)
        else:
            logp, ent = self._compute_logp_entropy(generation.scores, gen_ids)

        # extract final EOS logprob for SubTB flow matching
        self._last_eos_logprob_final = None
        if generation.scores:
            last_score = generation.scores[-1]
            log_probs = F.log_softmax(last_score[0], dim=-1)
            eos_id = self.tokenizer.eos_token_id
            if eos_id is not None:
                self._last_eos_logprob_final = float(log_probs[eos_id].item())

        # parse action from text
        action = self._parse_action(gen_text)

        if self.require_grad_logp:
            return action, logp, ent
        return action, float(logp.item()), float(ent.item())

    def score_action(
        self, obs: str, action: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score an action under the current policy.

        Used for off-policy training with replay buffers.

        Args:
            obs: Observation text
            action: Action dict to score

        Returns:
            Tuple of (log_prob, entropy) as tensors with gradient flow
        """
        prompt = self._build_prompt(obs)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True).input_ids
        action_text = self._render_action(action)
        action_ids = self.tokenizer(action_text, add_special_tokens=False).input_ids
        full_ids = torch.tensor([prompt_ids + action_ids], device=self.device)
        return self._score_generated(full_ids, prompt_len=len(prompt_ids))

    def _build_prompt(self, obs: str) -> str:
        """Build prompt from observation.

        If obs is a JSON-serialized list of message dicts (from SimpleCollector),
        uses the tokenizer chat template for proper formatting. Otherwise falls
        back to a simple wrapper.
        """
        import json

        # detect structured messages from SimpleCollector
        if obs.startswith("["):
            try:
                messages = json.loads(obs)
                if isinstance(messages, list) and messages and "role" in messages[0]:
                    if hasattr(self.tokenizer, "apply_chat_template"):
                        return self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        # fallback for plain text observations
        return f"User: {obs}\nAssistant:"

    def _parse_action(self, text: str) -> dict[str, Any]:
        """Parse action from generated text.

        Handles:
        - XML tool calls: <tool_call>{"name": ..., "input": ...}</tool_call>
        - XML programs: <submit_program>code</submit_program>
        - Raw JSON: {"name": ..., "input": ...}
        - Fallback: treat as answer text
        """
        import json
        import re

        text = text.strip()

        # strip thinking blocks (Qwen3 generates <think>...</think> before acting)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # handle <tool_call>JSON</tool_call>
        tc_match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
        if tc_match:
            try:
                return json.loads(tc_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # handle <submit_program>code</submit_program>
        sp_match = re.search(r"<submit_program>(.*?)</submit_program>", text, re.DOTALL)
        if sp_match:
            return {"type": "submit_program", "payload": sp_match.group(1).strip()}

        # try raw JSON
        try:
            if "{" in text:
                start = text.index("{")
                end = text.rindex("}") + 1
                return json.loads(text[start:end])
        except (json.JSONDecodeError, ValueError):
            pass

        # fallback: treat as answer
        return {"type": "answer", "payload": text}

    def _render_action(self, action: dict[str, Any]) -> str:
        """Render action to text."""
        import json
        return json.dumps(action)

    @staticmethod
    def _compute_logp_entropy(
        scores: Sequence[torch.Tensor],
        gen_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute log-prob and entropy from generation scores."""
        if not scores or gen_ids.numel() == 0:
            device = gen_ids.device
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        logps: list[torch.Tensor] = []
        ents: list[torch.Tensor] = []

        for step, step_scores in enumerate(scores):
            step_log_probs = torch.log_softmax(step_scores, dim=-1)
            step_token = gen_ids[:, step]
            tok_logp = step_log_probs.gather(-1, step_token.unsqueeze(-1)).squeeze(-1)
            logps.append(tok_logp)

            step_probs = torch.softmax(step_scores, dim=-1)
            # handle 0 * -inf = nan (IEEE 754 edge case when probs=0, log_probs=-inf)
            term = step_probs * step_log_probs
            term = torch.nan_to_num(term, nan=0.0)
            step_ent = -term.sum(dim=-1)
            ents.append(step_ent)

        logp = torch.stack(logps, dim=1).sum(dim=1)
        ent = torch.stack(ents, dim=1).mean(dim=1)
        return logp.squeeze(0), ent.squeeze(0)

    def _score_generated(
        self, full_ids: torch.Tensor, prompt_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute logprob/entropy for generated tokens with gradient flow."""
        gen_len = int(full_ids.shape[1] - prompt_len)
        if gen_len <= 0:
            device = full_ids.device
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        with torch.enable_grad():
            outputs = self.model(full_ids)
            logits = outputs.logits[:, :-1, :]
            targets = full_ids[:, 1:]
            log_probs = torch.log_softmax(logits, dim=-1)
            tok_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

            start = max(prompt_len - 1, 0)
            end = start + gen_len
            gen_logp = tok_logp[:, start:end]
            logp = gen_logp.sum(dim=1)

            # handle gradient NaN: clamp log_probs before computing p*log(p)
            # this prevents d/dp[p*log(p)] = log(p) + 1 from becoming -inf + 1 = NaN at p=0
            eps = torch.finfo(log_probs.dtype).tiny
            safe_min = torch.log(torch.tensor(eps, device=log_probs.device))
            safe_log_probs = torch.clamp(log_probs, min=safe_min)
            term = safe_log_probs.exp() * safe_log_probs
            # also sanitize forward pass for any remaining numerical issues
            term = torch.nan_to_num(term, nan=0.0)
            ent = -term.sum(dim=-1)
            gen_ent = ent[:, start:end].mean(dim=1)
            return logp.squeeze(0), gen_ent.squeeze(0)

    def parameters(self) -> Iterator[nn.Parameter]:
        """Return model parameters for optimizer."""
        return self.model.parameters()

    def named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        """Return named parameters for gradient verification."""
        return self.model.named_parameters()

    # -------------------- Protocol-style interface --------------------

    def _messages_to_input_ids(self, messages: list[Message]) -> torch.Tensor:
        """Convert messages to input token IDs using chat template."""
        # convert to list of dicts for apply_chat_template
        chat_messages = [{"role": m.role, "content": m.content} for m in messages]

        # try chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # fallback to simple concatenation
            text = "\n".join(f"{m['role']}: {m['content']}" for m in chat_messages)
            text += "\nassistant:"

        # tokenize
        inputs = self.tokenizer(text, return_tensors="pt")
        return inputs["input_ids"].to(self.device)

    def generate(self, messages: list[Message], *, gen: GenConfig) -> Generation:
        """Generate a completion for the given messages (Protocol-style interface).

        Args:
            messages: Conversation history.
            gen: Generation configuration.

        Returns:
            Generation result with text, tokens, and logprobs.
        """
        input_ids = self._messages_to_input_ids(messages)
        prompt_length = input_ids.shape[1]

        # prepare generation kwargs
        gen_kwargs = {
            "max_new_tokens": gen.max_tokens,
            "do_sample": gen.temperature > 0,
            "temperature": max(gen.temperature, 1e-7),
            "top_p": gen.top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "output_scores": True,
            "return_dict_in_generate": True,
        }

        # add stop sequences as eos tokens if provided
        if gen.stop_sequences:
            stop_token_ids = []
            for seq in gen.stop_sequences:
                tokens = self.tokenizer.encode(seq, add_special_tokens=False)
                if tokens:
                    stop_token_ids.append(tokens[0])
            if stop_token_ids:
                gen_kwargs["eos_token_id"] = stop_token_ids

        with torch.no_grad():
            outputs: Any = self.model.generate(input_ids, **gen_kwargs)

        # extract generated tokens (exclude prompt)
        generated_ids = outputs.sequences[0, prompt_length:].tolist()

        # decode to text
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # compute log probabilities for each generated token
        token_logprobs, eos_logprobs = self._compute_generation_logprobs(
            outputs.scores, generated_ids
        )

        # determine finish reason
        finish_reason = "length"
        if generated_ids:
            last_token = generated_ids[-1]
            eos_ids = gen_kwargs.get("eos_token_id")
            if isinstance(eos_ids, int):
                eos_ids = [eos_ids]
            elif eos_ids is None:
                eos_ids = [self.tokenizer.eos_token_id]
            if last_token in eos_ids:
                finish_reason = "stop"

        return Generation(
            text=text,
            token_ids=generated_ids,
            token_logprobs=token_logprobs,
            finish_reason=finish_reason,
            eos_logprobs=eos_logprobs,
        )

    def _compute_generation_logprobs(
        self, scores: tuple[torch.Tensor, ...], generated_ids: list[int]
    ) -> tuple[list[float], list[float]]:
        """Extract log probabilities from generation scores.

        Returns:
            Tuple of (token_logprobs, eos_logprobs):
            - token_logprobs: log p(token_t) at each step
            - eos_logprobs: log p(EOS) at each step (for SubTB flow matching)
        """
        import torch.nn.functional as F

        logprobs = []
        eos_logprobs = []
        eos_token_id = self.tokenizer.eos_token_id

        for score, token_id in zip(scores, generated_ids, strict=False):
            # score shape: [batch_size, vocab_size]
            log_probs = F.log_softmax(score[0], dim=-1)
            token_logprob = log_probs[token_id].item()
            logprobs.append(token_logprob)

            # extract EOS probability for SubTB flow matching
            if eos_token_id is not None:
                eos_logprob = log_probs[eos_token_id].item()
            else:
                eos_logprob = float("-inf")  # no EOS token defined
            eos_logprobs.append(eos_logprob)

        return logprobs, eos_logprobs

    def logprobs(self, messages: list[Message], tokens: list[int]) -> TokenLogProbs:
        """Compute log probabilities for a token sequence (Protocol-style interface).

        Args:
            messages: Conversation context.
            tokens: Token IDs to compute logprobs for.

        Returns:
            TokenLogProbs with per-token log probabilities.
        """
        import torch.nn.functional as F

        if not tokens:
            return TokenLogProbs(token_ids=[], logprobs=[])

        input_ids = self._messages_to_input_ids(messages)
        prompt_length = input_ids.shape[1]

        # append the tokens we want to score
        token_tensor = torch.tensor([tokens], device=self.device)
        full_input = torch.cat([input_ids, token_tensor], dim=1)

        with torch.no_grad():
            outputs = self.model(full_input)
            logits = outputs.logits

        # compute log probabilities for each token position
        logprobs = []
        for i, token_id in enumerate(tokens):
            # position of the logit that predicts this token
            logit_pos = prompt_length + i - 1
            if logit_pos < 0:
                logit_pos = 0

            log_probs = F.log_softmax(logits[0, logit_pos], dim=-1)
            token_logprob = log_probs[token_id].item()
            logprobs.append(token_logprob)

        return TokenLogProbs(token_ids=tokens.copy(), logprobs=logprobs)

    def score_tokens(
        self, messages: list[Message], tokens: list[int]
    ) -> torch.Tensor:
        """Compute differentiable log probabilities for a token sequence.

        Unlike logprobs(), this returns a tensor with gradient tracking enabled,
        allowing gradients to flow back to model parameters during training.

        Args:
            messages: Conversation context.
            tokens: Token IDs to compute logprobs for.

        Returns:
            Tensor of shape [len(tokens)] with log probabilities.
            The tensor is connected to the model's computational graph.
        """
        import torch.nn.functional as F

        if not tokens:
            return torch.tensor([], device=self.device)

        input_ids = self._messages_to_input_ids(messages)
        prompt_length = input_ids.shape[1]

        # append the tokens we want to score
        token_tensor = torch.tensor([tokens], device=self.device)
        full_input = torch.cat([input_ids, token_tensor], dim=1)

        # forward pass WITH gradients enabled
        outputs = self.model(full_input)
        logits = outputs.logits

        # compute log probabilities for each token position
        logprobs_list = []
        for i, token_id in enumerate(tokens):
            logit_pos = prompt_length + i - 1
            if logit_pos < 0:
                logit_pos = 0

            log_probs = F.log_softmax(logits[0, logit_pos], dim=-1)
            token_logprob = log_probs[token_id]  # keep as tensor, don't .item()
            logprobs_list.append(token_logprob)

        return torch.stack(logprobs_list)
