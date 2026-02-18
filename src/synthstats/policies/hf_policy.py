"""HuggingFace LM policy with LoRA support and logprob accounting."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from synthstats.core.policy import GenConfig, Generation, TokenLogProbs
from synthstats.core.types import Message
from synthstats.policies.parsing import parse_action, render_action

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

PolicyOutput = tuple[dict[str, Any], float | torch.Tensor, float | torch.Tensor]


class MockHFPolicy:
    """Mock policy for testing without loading actual models."""

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

        self._fixed_text = fixed_text
        self._fixed_token_ids = fixed_token_ids or [101, 102, 103, 104, 105]
        self._fixed_token_logprobs = fixed_token_logprobs or [-0.1, -0.2, -0.15, -0.25, -0.1]
        self._fixed_finish_reason = fixed_finish_reason

        if fixed_eos_logprobs is not None:
            self._fixed_eos_logprobs = fixed_eos_logprobs
        else:
            # starts low (unlikely to stop), increases toward end
            n = len(self._fixed_token_ids)
            self._fixed_eos_logprobs = [-5.0 + 4.9 * i / max(n - 1, 1) for i in range(n)]

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

        # dummy parameter so policy.parameters() works in optimizer
        self._dummy = nn.Parameter(torch.zeros(1, device=device))

        self._last_eos_logprob_final: float | None = None

    def __call__(self, obs: str, temperature: float | None = None) -> PolicyOutput:
        action = {"type": "answer", "payload": "42"}

        logp: float | torch.Tensor
        ent: float | torch.Tensor
        if self.require_grad_logp:
            logp = torch.tensor(-0.5, device=self.device, requires_grad=True)
            ent = torch.tensor(0.1, device=self.device, requires_grad=True)
        else:
            logp = -0.5
            ent = 0.1

        self._last_eos_logprob_final = self._fixed_eos_logprobs[-1]

        return action, logp, ent

    def sample_with_eos(
        self,
        obs: str,
        temperature: float | None = None,
    ) -> tuple[
        dict[str, Any], float | torch.Tensor, float | torch.Tensor, float | torch.Tensor | None
    ]:
        """Explicit sampling API that includes final EOS logprob."""
        action, logp, ent = self(obs, temperature=temperature)
        return action, logp, ent, self._last_eos_logprob_final

    def score_action(self, obs: str, action: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        logp = torch.tensor(-0.5, device=self.device, requires_grad=True)
        ent = torch.tensor(0.1, device=self.device, requires_grad=True)
        self._last_eos_logprob_final = self._fixed_eos_logprobs[-1]
        return logp, ent

    def score_action_with_eos(
        self,
        obs: str,
        action: dict[str, Any],
        temperature: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, float | torch.Tensor | None]:
        """Explicit scoring API that includes final EOS logprob."""
        del temperature
        logp, ent = self.score_action(obs, action)
        return logp, ent, self._last_eos_logprob_final

    def parameters(self) -> Iterator[nn.Parameter]:
        yield self._dummy

    def named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        yield ("dummy", self._dummy)

    # -------------------- Protocol-style interface --------------------

    def generate(self, messages: list[Message], *, gen: GenConfig) -> Generation:
        return Generation(
            text=self._fixed_text,
            token_ids=self._fixed_token_ids.copy(),
            token_logprobs=self._fixed_token_logprobs.copy(),
            finish_reason=self._fixed_finish_reason,
            eos_logprobs=self._fixed_eos_logprobs.copy(),
        )

    def logprobs(self, messages: list[Message], tokens: list[int]) -> TokenLogProbs:
        logprobs = [-0.1 * (i + 1) for i in range(len(tokens))]
        return TokenLogProbs(token_ids=tokens.copy(), logprobs=logprobs)

    def score_tokens(self, messages: list[Message], tokens: list[int]) -> torch.Tensor:
        if not tokens:
            return torch.tensor([], device=self.device, requires_grad=True)

        logprobs = torch.tensor(
            [-0.1 * (i + 1) for i in range(len(tokens))],
            device=self.device,
            requires_grad=True,
        )
        return logprobs


# backward compatibility alias
MockPolicy = MockHFPolicy


class HFPolicy:
    """HuggingFace causal LM policy with optional LoRA and 4-bit quantization."""

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

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers is required for HFPolicy. Install with: pip install transformers"
            ) from e

        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)

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
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map.get(dtype, torch.bfloat16) if device != "cpu" else torch.float32
            load_kwargs = {"torch_dtype": torch_dtype}

        self.model: Any = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        if "device_map" not in load_kwargs:
            self.model = self.model.to(device)  # type: ignore[arg-type]

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if lora_config is not None:
            self._apply_lora(lora_config)

        if gradient_checkpointing and not (use_4bit and lora_config is not None):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        if require_grad_logp or lora_config is not None:
            self.model.train()
        else:
            self.model.eval()

        self._last_eos_logprob_final: float | None = None

    def _apply_lora(self, lora_config: dict[str, Any]) -> None:
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

    def __call__(self, obs: str, temperature: float | None = None) -> PolicyOutput:
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
            logp, ent, _ = self._score_generated(seq, prompt_len)
        else:
            logp, ent = self._compute_logp_entropy(generation.scores, gen_ids)

        self._last_eos_logprob_final = None
        if generation.scores:
            last_score = generation.scores[-1]
            log_probs = F.log_softmax(last_score[0], dim=-1)
            eos_id = self.tokenizer.eos_token_id
            if eos_id is not None:
                self._last_eos_logprob_final = float(log_probs[eos_id].item())

        action = parse_action(gen_text)

        if self.require_grad_logp:
            return action, logp, ent
        return action, float(logp.item()), float(ent.item())

    def sample_with_eos(
        self,
        obs: str,
        temperature: float | None = None,
    ) -> tuple[
        dict[str, Any], float | torch.Tensor, float | torch.Tensor, float | torch.Tensor | None
    ]:
        """Explicit sampling API that includes final EOS logprob."""
        action, logp, ent = self(obs, temperature=temperature)
        return action, logp, ent, self._last_eos_logprob_final

    def score_action(self, obs: str, action: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        prompt = self._build_prompt(obs)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True).input_ids
        action_text = render_action(action)
        action_ids = self.tokenizer(action_text, add_special_tokens=False).input_ids
        full_ids = torch.tensor([prompt_ids + action_ids], device=self.device)
        logp, ent, final_logits = self._score_generated(full_ids, prompt_len=len(prompt_ids))

        # reuse logits from same forward pass
        with torch.no_grad():
            log_probs = torch.log_softmax(final_logits[0], dim=-1)
            eos_id = self.tokenizer.eos_token_id
            if eos_id is not None:
                self._last_eos_logprob_final = float(log_probs[eos_id].item())
            else:
                self._last_eos_logprob_final = None

        return logp, ent

    def score_action_with_eos(
        self,
        obs: str,
        action: dict[str, Any],
        temperature: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, float | torch.Tensor | None]:
        """Explicit scoring API that includes final EOS logprob."""
        del temperature
        logp, ent = self.score_action(obs, action)
        return logp, ent, self._last_eos_logprob_final

    def _build_prompt(self, obs: str) -> str:
        """Build prompt, using chat template for structured messages."""
        import json

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

        return f"User: {obs}\nAssistant:"

    @staticmethod
    def _compute_logp_entropy(
        scores: Sequence[torch.Tensor],
        gen_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

            step_probs = step_log_probs.exp()
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Logprob/entropy with gradient flow. Also returns final_logits for EOS."""
        gen_len = int(full_ids.shape[1] - prompt_len)
        if gen_len <= 0:
            device = full_ids.device
            z = torch.tensor(0.0, device=device)
            empty_logits = torch.zeros(1, 1, device=device)
            return z, z, empty_logits

        with torch.enable_grad():
            outputs = self.model(full_ids)
            final_logits = outputs.logits[:, -1, :]
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
            term = torch.nan_to_num(term, nan=0.0)
            ent = -term.sum(dim=-1)
            gen_ent = ent[:, start:end].mean(dim=1)
            return logp.squeeze(0), gen_ent.squeeze(0), final_logits

    def parameters(self) -> Iterator[nn.Parameter]:
        return self.model.parameters()

    def named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        return self.model.named_parameters()

    # -------------------- Protocol-style interface --------------------

    def _messages_to_input_ids(self, messages: list[Message]) -> torch.Tensor:
        chat_messages = [{"role": m.role, "content": m.content} for m in messages]

        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = "\n".join(f"{m['role']}: {m['content']}" for m in chat_messages)
            text += "\nassistant:"

        inputs = self.tokenizer(text, return_tensors="pt")
        return inputs["input_ids"].to(self.device)

    def generate(self, messages: list[Message], *, gen: GenConfig) -> Generation:
        input_ids = self._messages_to_input_ids(messages)
        prompt_length = input_ids.shape[1]

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

        generated_ids = outputs.sequences[0, prompt_length:].tolist()
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        token_logprobs, eos_logprobs = self._compute_generation_logprobs(
            outputs.scores, generated_ids
        )

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
        """Extract per-token and per-step EOS log probabilities."""
        import torch.nn.functional as F

        if not scores or not generated_ids:
            return [], []

        scores_tensor = torch.stack([s[0] for s in scores[: len(generated_ids)]], dim=0)
        log_probs = F.log_softmax(scores_tensor, dim=-1)

        token_ids_tensor = torch.tensor(generated_ids, device=scores_tensor.device)
        token_logprobs = log_probs.gather(-1, token_ids_tensor.unsqueeze(-1)).squeeze(-1)

        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is not None:
            eos_logprobs = log_probs[:, eos_token_id]
            return token_logprobs.tolist(), eos_logprobs.tolist()
        return token_logprobs.tolist(), [float("-inf")] * len(generated_ids)

    def logprobs(self, messages: list[Message], tokens: list[int]) -> TokenLogProbs:
        import torch.nn.functional as F

        if not tokens:
            return TokenLogProbs(token_ids=[], logprobs=[])

        input_ids = self._messages_to_input_ids(messages)
        prompt_length = input_ids.shape[1]

        token_tensor = torch.tensor([tokens], device=self.device)
        full_input = torch.cat([input_ids, token_tensor], dim=1)

        with torch.no_grad():
            outputs = self.model(full_input)
            logits = outputs.logits

        n = len(tokens)
        positions = (torch.arange(n, device=self.device) + prompt_length - 1).clamp(min=0)
        log_probs = F.log_softmax(logits[0, positions], dim=-1)
        token_ids_t = torch.tensor(tokens, device=self.device)
        logprobs_t = log_probs.gather(-1, token_ids_t.unsqueeze(-1)).squeeze(-1)

        return TokenLogProbs(token_ids=tokens.copy(), logprobs=logprobs_t.tolist())

    def score_tokens(self, messages: list[Message], tokens: list[int]) -> torch.Tensor:
        """Like logprobs() but returns gradient-connected tensor."""
        import torch.nn.functional as F

        if not tokens:
            return torch.tensor([], device=self.device)

        input_ids = self._messages_to_input_ids(messages)
        prompt_length = input_ids.shape[1]

        token_tensor = torch.tensor([tokens], device=self.device)
        full_input = torch.cat([input_ids, token_tensor], dim=1)

        outputs = self.model(full_input)
        logits = outputs.logits

        n = len(tokens)
        positions = (torch.arange(n, device=self.device) + prompt_length - 1).clamp(min=0)
        log_probs = F.log_softmax(logits[0, positions], dim=-1)
        token_ids_t = torch.tensor(tokens, device=self.device)
        return log_probs.gather(-1, token_ids_t.unsqueeze(-1)).squeeze(-1)
