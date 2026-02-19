"""RunPod vLLM serverless policy.

Uses RunPod's OpenAI-compatible vLLM endpoints for inference,
freeing the local GPU for loss computation and optimizer steps.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import torch

from synthstats.policies.parsing import estimate_entropy, parse_action, render_action

logger = logging.getLogger(__name__)

PolicyOutput = tuple[dict[str, Any], float | torch.Tensor, float | torch.Tensor]


@dataclass
class RunPodConfig:
    """Configuration for RunPod vLLM serverless endpoint."""

    endpoint_id: str = ""
    api_key: str | None = None
    model: str = "default"
    max_tokens: int = 300
    temperature: float = 0.7
    top_logprobs: int = 20
    timeout: float = 120.0
    max_retries: int = 3

    def get_api_key(self) -> str:
        key = self.api_key or os.environ.get("RUNPOD_API_KEY")
        if not key:
            raise ValueError(
                "RunPod API key required. Set RUNPOD_API_KEY env var or pass api_key."
            )
        return key

    def get_endpoint_id(self) -> str:
        eid = self.endpoint_id or os.environ.get("RUNPOD_ENDPOINT_ID", "")
        if not eid:
            raise ValueError(
                "RunPod endpoint ID required. Set RUNPOD_ENDPOINT_ID env var or pass endpoint_id."
            )
        return eid


@dataclass
class RunPodPolicy:
    """Policy using RunPod's vLLM serverless endpoint via OpenAI-compatible API.

    Plugs into LocalRunner's TrajectoryCollector with zero runner changes.
    The collector probes for sample_with_eos / score_action_with_eos via
    getattr and discovers them automatically.
    """

    config: RunPodConfig
    _client: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _tokenizer_loaded: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        logger.debug(
            "RunPodPolicy: score_action_with_eos returns eos=None; "
            "replay mixed with local HF policy triggers eos_downgraded fallback."
        )

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError(
                    "openai package required for RunPodPolicy. "
                    "Install with: uv pip install 'synthstats[runpod]'"
                ) from e

            endpoint_id = self.config.get_endpoint_id()
            api_key = self.config.get_api_key()
            base_url = f"https://api.runpod.ai/v2/{endpoint_id}/openai/v1"

            self._client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
        return self._client

    @property
    def tokenizer(self) -> Any:
        if not self._tokenizer_loaded:
            self._tokenizer_loaded = True
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.config.model)
            except (OSError, ValueError, ImportError) as e:
                logger.warning("Could not load tokenizer for %s: %s", self.config.model, e)
        return self._tokenizer

    def __call__(self, obs: str, temperature: float | None = None) -> PolicyOutput:
        action, logp, entropy, _ = self.sample_with_eos(obs, temperature=temperature)
        return action, logp, entropy

    def sample_with_eos(
        self,
        obs: str,
        temperature: float | None = None,
    ) -> tuple[dict[str, Any], float, float, float | None]:
        temp = temperature if temperature is not None else self.config.temperature
        messages = self._build_chat_messages(obs)

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=temp,
            logprobs=True,
            top_logprobs=self.config.top_logprobs,
        )

        choice = response.choices[0]
        gen_text = choice.message.content or ""
        action = parse_action(gen_text)

        token_logprobs: list[float] = []
        eos_logprob: float | None = None

        if choice.logprobs and choice.logprobs.content:
            for token_info in choice.logprobs.content:
                if token_info.logprob is not None:
                    token_logprobs.append(token_info.logprob)

            eos_logprob = self._extract_eos_logprob(choice.logprobs.content)

        logp = sum(token_logprobs) if token_logprobs else -1.0
        entropy = estimate_entropy(token_logprobs)

        return action, logp, entropy, eos_logprob

    def score_action(
        self, obs: str, action: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Re-score an action using chat completions with echoed prompt tokens.

        This keeps replay scoring aligned with `sample_with_eos`, which also uses
        chat messages and the chat completions endpoint.
        """
        messages = self._build_chat_messages(obs)
        action_text = render_action(action)
        score_messages = messages + [{"role": "assistant", "content": action_text}]

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=score_messages,
                max_tokens=0,
                temperature=0.0,
                logprobs=True,
                top_logprobs=self.config.top_logprobs,
                extra_body={"echo": True},
            )
        except Exception as exc:
            raise RuntimeError(
                "RunPod endpoint does not support chat.completions with "
                "logprobs + echo=True for replay scoring."
            ) from exc

        choice = response.choices[0]
        all_logprobs: list[float] = []
        if choice.logprobs and choice.logprobs.content:
            for token_info in choice.logprobs.content:
                lp = getattr(token_info, "logprob", None)
                if lp is not None:
                    all_logprobs.append(lp)

        action_token_count = self._estimate_action_token_count(action_text, len(all_logprobs))
        action_lps = all_logprobs[-action_token_count:] if action_token_count > 0 else []

        logp = sum(action_lps) if action_lps else -1.0
        entropy = estimate_entropy(action_lps)

        return (
            torch.tensor(logp, requires_grad=False),
            torch.tensor(entropy, requires_grad=False),
        )

    def score_action_with_eos(
        self,
        obs: str,
        action: dict[str, Any],
        temperature: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """EOS always None here; replay scoring API doesn't expose final EOS."""
        del temperature
        logp, entropy = self.score_action(obs, action)
        return logp, entropy, None

    def _extract_eos_logprob(self, content: list[Any]) -> float | None:
        if not content:
            return None

        tokenizer = self.tokenizer
        if tokenizer is None:
            return None

        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            return None

        eos_token_str = tokenizer.decode([eos_token_id])

        last_info = content[-1]
        if not hasattr(last_info, "top_logprobs") or not last_info.top_logprobs:
            return None

        for alt in last_info.top_logprobs:
            if alt.token == eos_token_str:
                return alt.logprob

        return None

    def _estimate_action_token_count(self, action_text: str, total_tokens: int) -> int:
        if total_tokens <= 0:
            return 0

        tokenizer = self.tokenizer
        if tokenizer is not None:
            try:
                action_tokens = tokenizer.encode(action_text, add_special_tokens=False)
                if action_tokens:
                    return min(len(action_tokens), total_tokens)
            except Exception as exc:
                logger.warning(
                    "Tokenizer encode failed; using token-count midpoint heuristic for action boundary: %s",
                    exc,
                )
                return min(max(1, total_tokens // 2), total_tokens)

        # ~4 chars/token fallback for tokenizer-less environments
        logger.warning(
            "No tokenizer available; using char-count heuristic for action boundary. "
            "Log-probs may be inaccurate."
        )
        return min(max(1, len(action_text) // 4), total_tokens)

    def _build_chat_messages(self, obs: str) -> list[dict[str, str]]:
        try:
            parsed = json.loads(obs)
            if isinstance(parsed, list) and parsed and "role" in parsed[0]:
                return parsed
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        return [
            {"role": "system", "content": "You are an agent that responds to observations."},
            {"role": "user", "content": obs},
        ]

