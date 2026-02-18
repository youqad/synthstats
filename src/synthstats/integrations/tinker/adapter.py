"""Tinker API integration for GFlowNet training.

Import-safe: works without Tinker installed. Use `is_tinker_available()` to check.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    pass

from synthstats.core.constants import (
    LOGZ_LR_DEFAULT,
    REWARD_FLOOR_DEFAULT,
    SUBTB_LAMBDA_DEFAULT,
    TB_MAX_RESIDUAL_DEFAULT,
)
from synthstats.policies.parsing import build_prompt, estimate_entropy, parse_action, render_action

logger = logging.getLogger(__name__)


class TinkerOptionalDependencyError(RuntimeError):
    pass


def is_tinker_available() -> bool:
    try:
        import tinker  # noqa: F401

        return True
    except ImportError:
        return False


def require_tinker() -> Any:
    try:
        import tinker

        return tinker
    except ImportError as e:
        raise TinkerOptionalDependencyError(
            "tinker not installed. pip install tinker"
        ) from e


def _make_service_client(config: TinkerConfig) -> Any:
    """Lazy-init a Tinker ServiceClient from config."""
    tinker = require_tinker()
    os.environ["TINKER_API_KEY"] = config.get_api_key()
    base_url = config.get_base_url()
    kwargs: dict[str, Any] = {}
    if base_url:
        kwargs["base_url"] = base_url
    return tinker.ServiceClient(**kwargs)


@dataclass
class TinkerConfig:
    model: str = "Qwen/Qwen3-4B"
    api_key: str | None = None
    base_url: str | None = None
    max_tokens: int = 256
    temperature: float = 0.7
    lora_rank: int = 32
    learning_rate: float = 1e-5
    logZ_lr: float = LOGZ_LR_DEFAULT
    subtb_alpha: float = 0.1
    subtb_lambda: float = SUBTB_LAMBDA_DEFAULT  # length discount (fixed per paper)
    tb_max_residual: float = TB_MAX_RESIDUAL_DEFAULT  # clamp for stability
    log_sparse_reward: float | None = None  # per-prefix reward for incomplete programs
    reward_schedule: dict[str, Any] | Any | None = None  # {start, end, horizon, mode}

    def get_api_key(self) -> str:
        """Resolve API key from config or env var."""
        key = self.api_key or os.environ.get("TINKER_API_KEY")
        if not key:
            raise ValueError("Tinker API key required. Set TINKER_API_KEY env var or pass api_key.")
        return key

    def get_base_url(self) -> str | None:
        return self.base_url or os.environ.get("TINKER_BASE_URL")


PolicyOutput = tuple[dict[str, Any], float, float]


@dataclass
class TurnBoundary:
    """Character-level turn boundary (not token-level; Tinker retokenizes)."""

    start_char: int
    end_char: int
    role: str
    generation_idx: int  # Trajectory.token_ids index for assistant turns, -1 otherwise
    has_reward: bool = False


def _extract_prompt(messages: list[Any]) -> str:
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
    parts = [m.content for m in messages if m.role == "assistant"]
    return "\n".join(parts)


def _extract_multi_turn_completion(
    messages: list[Any],
) -> tuple[str, list[TurnBoundary]]:
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

        if parts:
            parts.append("\n")
            char_pos += 1

        parts.append(content)

        boundaries.append(
            TurnBoundary(
                start_char=char_pos,
                end_char=char_pos + len(content),
                role=msg.role,
                generation_idx=assistant_idx if msg.role == "assistant" else -1,
                has_reward=False,
            )
        )

        char_pos += len(content)

        if msg.role == "assistant":
            last_assistant_idx = len(boundaries) - 1
            assistant_idx += 1

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
    """Build per-token loss mask from character-level turn boundaries.

    BPE context-dependence may cause off-by-one at turn edges. Prompts
    end with newline to force a token boundary in most tokenizers.
    """
    import torch

    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)

    try:
        encoding = tokenizer.encode_plus(
            completion,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        offsets = encoding.get("offset_mapping", [])
    except (TypeError, AttributeError) as e:
        raise ValueError(
            f"Tokenizer {type(tokenizer).__name__} does not support offset_mapping."
        ) from e

    for boundary in boundaries:
        if boundary.role != "assistant":
            continue

        tok_start, tok_end = None, None
        for tok_idx, (char_start, char_end) in enumerate(offsets):
            adjusted_idx = tok_idx + prompt_offset
            if adjusted_idx >= seq_len:
                break
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
    reward_floor: float = REWARD_FLOOR_DEFAULT,
    strict_single_turn: bool = True,
    multi_turn: bool = False,
) -> dict[str, Any]:
    """Convert Trajectories to TinkerTrainer batch format."""
    import torch

    prompts: list[str] = []
    completions: list[str] = []
    log_rewards: list[float] = []
    loss_masks: list[list[bool]] = []
    all_turn_boundaries: list[list[TurnBoundary]] = []
    has_any_mask = False

    for i, traj in enumerate(trajectories):
        assistant_count = sum(1 for m in traj.messages if m.role == "assistant")

        if not multi_turn and strict_single_turn and assistant_count > 1:
            raise ValueError(
                f"Trajectory {i} has {assistant_count} assistant messages. "
                "Use multi_turn=True for multi-turn trajectories, or set "
                "strict_single_turn=False to flatten (not recommended)."
            )

        prompt_text = _extract_prompt(traj.messages)

        if multi_turn and assistant_count > 1:
            completion_text, turn_boundaries = _extract_multi_turn_completion(traj.messages)
            all_turn_boundaries.append(turn_boundaries)
        else:
            completion_text = _extract_single_turn_completion(traj.messages)
            all_turn_boundaries.append([])

        prompts.append(prompt_text)
        completions.append(completion_text)

        reward_val = max(traj.reward.total, reward_floor)
        log_rewards.append(math.log(reward_val))

        if not multi_turn:
            if traj.loss_mask and len(traj.loss_mask) > 0 and traj.loss_mask[0]:
                loss_masks.append(traj.loss_mask[0])
                has_any_mask = True
            else:
                loss_masks.append([])

    is_multi_turn_batch = multi_turn and any(len(tb) > 0 for tb in all_turn_boundaries)

    result: dict[str, Any] = {
        "prompts": prompts,
        "completions": completions,
        "log_reward": torch.tensor(log_rewards, dtype=torch.float32, device=device),
        "is_multi_turn": is_multi_turn_batch,
    }

    if is_multi_turn_batch:
        result["turn_boundaries"] = all_turn_boundaries

    if not is_multi_turn_batch and has_any_mask and loss_masks:
        max_mask_len = max(len(m) for m in loss_masks) if loss_masks else 0
        if max_mask_len > 0:
            mask_tensor = torch.ones(len(loss_masks), max_mask_len, dtype=torch.bool, device=device)
            for i, mask in enumerate(loss_masks):
                if mask:
                    mask_len = len(mask)
                    mask_tensor[i, :mask_len] = torch.tensor(mask, dtype=torch.bool)
            result["loss_mask"] = mask_tensor

    return result


@dataclass
class TinkerPolicy:
    """Policy using Tinker API for sampling."""

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
        if self._service_client is None:
            self._service_client = _make_service_client(self.config)
        return self._service_client

    @property
    def sampling_client(self) -> Any:
        if self._sampling_client is None:
            self._sampling_client = self.service_client.create_sampling_client(
                base_model=self.config.model,
            )
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.config.model)
            except Exception as e:
                logger.warning(f"Could not load tokenizer for {self.config.model}: {e}")
                self._tokenizer = None

        if not self._tokenizer_checked and self._tokenizer is not None:
            self._check_tokenizer_compatibility()
            self._tokenizer_checked = True

        return self._sampling_client

    def __call__(self, obs: str, temperature: float | None = None) -> PolicyOutput:
        temp = temperature if temperature is not None else self.config.temperature
        prompt = build_prompt(obs)

        _ = self.sampling_client

        if self._tokenizer:
            prompt_tokens = self._tokenizer.encode(prompt)
        else:
            prompt_tokens = list(range(len(prompt)))

        if is_tinker_available():
            tinker = require_tinker()
            model_input = tinker.ModelInput.from_ints(prompt_tokens)
            sampling_params = tinker.SamplingParams(
                max_tokens=self.config.max_tokens,
                temperature=temp,
            )
        else:
            model_input = prompt_tokens
            sampling_params = {"max_tokens": self.config.max_tokens, "temperature": temp}

        future = self.sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        )
        result = future.result() if hasattr(future, "result") else future

        gen_text = ""
        logprobs = None
        if hasattr(result, "sequences") and result.sequences:
            seq = result.sequences[0]
            logprobs = getattr(seq, "logprobs", None)
            gen_tokens = getattr(seq, "tokens", [])
            if self._tokenizer and gen_tokens:
                gen_text = self._tokenizer.decode(gen_tokens, skip_special_tokens=True)
        elif hasattr(result, "text"):
            gen_text = result.text
            logprobs = getattr(result, "logprobs", None)

        action = parse_action(gen_text)

        if logprobs:
            logp = sum(lp for lp in logprobs if lp is not None)
            entropy = estimate_entropy([lp for lp in logprobs if lp is not None])
        else:
            logp = -1.0
            entropy = 0.0

        return action, logp, entropy

    def sample_with_eos(
        self,
        obs: str,
        temperature: float | None = None,
    ) -> tuple[dict[str, Any], float | Any, float | Any, float | Any | None]:
        """Sampling with EOS logprob slot (always None; Tinker doesn't expose it)."""
        action, logp, entropy = self(obs, temperature=temperature)
        return action, logp, entropy, None

    def score_action(self, obs: str, action: dict[str, Any]) -> tuple[Any, Any]:
        import torch

        prompt = build_prompt(obs)
        action_text = render_action(action)
        full_text = prompt + action_text

        _ = self.sampling_client

        if self._tokenizer:
            full_tokens = self._tokenizer.encode(full_text)
            prompt_tokens = self._tokenizer.encode(prompt)
        else:
            full_tokens = list(range(len(full_text)))
            prompt_tokens = list(range(len(prompt)))

        if is_tinker_available():
            tinker = require_tinker()
            model_input = tinker.ModelInput.from_ints(full_tokens)
        else:
            model_input = full_tokens

        future = self.sampling_client.compute_logprobs(model_input)
        logprobs = future.result() if hasattr(future, "result") else future
        if hasattr(logprobs, "logprobs"):
            logprobs = logprobs.logprobs

        action_start = len(prompt_tokens)
        if logprobs:
            action_logprobs = [lp for lp in logprobs[action_start:] if lp is not None]
            logp = sum(action_logprobs) if action_logprobs else -1.0
            entropy = estimate_entropy(action_logprobs)
        else:
            logp = -1.0
            entropy = 0.0

        return (
            torch.tensor(logp, requires_grad=False),
            torch.tensor(entropy, requires_grad=False),
        )

    def score_action_with_eos(
        self,
        obs: str,
        action: dict[str, Any],
        temperature: float | None = None,
    ) -> tuple[Any, Any, None]:
        del temperature
        logp, entropy = self.score_action(obs, action)
        return logp, entropy, None

    def _check_tokenizer_compatibility(self, test_text: str = "Hello world") -> bool:
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
        except Exception as e:
            logger.debug("tokenizer compatibility check failed: %s", e)

        return True


@dataclass
class TinkerTrainer:
    """Tinker API trainer with TB loss."""

    config: TinkerConfig
    logZ_init: float = 0.0
    _service_client: Any = field(default=None, init=False, repr=False)
    _training_client: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _logZ: Any = field(default=None, init=False, repr=False)
    _step: int = field(default=0, init=False, repr=False)
    _reward_schedule: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        import torch
        import torch.nn as nn

        self._logZ = nn.Parameter(torch.tensor(self.logZ_init))
        self._init_reward_schedule()

    def _init_reward_schedule(self) -> None:
        reward_cfg = getattr(self.config, "reward_schedule", None)
        if reward_cfg is None and isinstance(self.config, dict):
            reward_cfg = self.config.get("reward_schedule")

        if reward_cfg is None:
            return

        from synthstats.train.utils.schedulers import RewardTemperatureSchedule

        def get_val(key: str, default: Any) -> Any:
            if isinstance(reward_cfg, dict):
                return reward_cfg.get(key, default)
            return getattr(reward_cfg, key, default)

        start = get_val("start", 1.0)
        end = get_val("end", 0.1)

        if start <= 0 or end <= 0:
            raise ValueError(
                f"reward_schedule temperatures must be positive: start={start}, end={end}"
            )

        self._reward_schedule = RewardTemperatureSchedule(
            start=start,
            end=end,
            horizon=get_val("horizon", 1000),
            mode=get_val("mode", "linear"),
        )

    @property
    def step(self) -> int:
        return self._step

    @property
    def reward_temperature(self) -> float:
        if self._reward_schedule is None:
            return 1.0
        return self._reward_schedule.get(self._step)

    @property
    def logZ(self) -> Any:
        return self._logZ

    @property
    def service_client(self) -> Any:
        if self._service_client is None:
            self._service_client = _make_service_client(self.config)
        return self._service_client

    @property
    def training_client(self) -> Any:
        if self._training_client is None:
            self._training_client = self.service_client.create_lora_training_client(
                base_model=self.config.model,
                rank=self.config.lora_rank if self.config.lora_rank > 0 else 32,
            )
            self._tokenizer = self._training_client.get_tokenizer()
        return self._training_client

    @property
    def tokenizer(self) -> Any:
        _ = self.training_client
        return self._tokenizer

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Run one TB loss step."""
        import torch

        self._validate_batch(batch)

        is_multi_turn = batch.get("is_multi_turn", False)
        turn_boundaries = batch.get("turn_boundaries", [])
        completions = batch["completions"]
        prompts = batch["prompts"]

        tinker = require_tinker()
        tokenizer = self.training_client.get_tokenizer()

        data = []
        log_rewards_list = batch["log_reward"].tolist()
        for prompt, completion, _log_r in zip(
            batch["prompts"], batch["completions"], log_rewards_list, strict=True
        ):
            full_text = prompt + completion
            tokens = tokenizer.encode(full_text, add_special_tokens=False)

            input_tokens = tokens[:-1]
            target_tokens = tokens[1:]
            seq_len = len(target_tokens)

            model_input = tinker.ModelInput.from_ints(input_tokens)
            loss_inputs = {
                "weights": tinker.TensorData.from_torch(torch.ones(seq_len, dtype=torch.float32)),
                "target_tokens": tinker.TensorData.from_torch(
                    torch.tensor(target_tokens, dtype=torch.int64)
                ),
            }
            datum = tinker.Datum(model_input=model_input, loss_fn_inputs=loss_inputs)
            data.append(datum)

        log_rewards_raw = batch["log_reward"]
        original_mask = batch.get("loss_mask")
        logZ = self._logZ

        step_used = self._step
        reward_temperature = self.reward_temperature
        if reward_temperature <= 0:
            raise ValueError(f"reward_temperature must be positive, got {reward_temperature}")
        if reward_temperature != 1.0:
            log_rewards = log_rewards_raw / reward_temperature
        else:
            log_rewards = log_rewards_raw

        from synthstats.train.objectives.subtb_endpoint import LOG_SPARSE_REWARD_DEFAULT

        scaled_sparse: float | None = None
        if reward_temperature != 1.0:
            scaled_sparse = LOG_SPARSE_REWARD_DEFAULT / reward_temperature
        cfg_sparse = getattr(self.config, "log_sparse_reward", None)
        if cfg_sparse is not None:
            scaled_sparse = float(cfg_sparse)

        def custom_tb_loss(
            data: list[dict[str, str]], logprobs_list: list[Any]
        ) -> tuple[Any, dict[str, float]]:
            max_len = max(lp.shape[0] for lp in logprobs_list)
            batch_size = len(logprobs_list)
            device = logprobs_list[0].device

            log_probs = torch.zeros(batch_size, max_len, device=device)
            mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)

            prompt_offsets = []
            for prompt in prompts:
                prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
                prompt_offsets.append(max(0, len(prompt_tokens) - 1))

            for i, lp in enumerate(logprobs_list):
                seq_len = lp.shape[0]
                log_probs[i, :seq_len] = lp

                if is_multi_turn and i < len(turn_boundaries) and turn_boundaries[i]:
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
                    mask[i, :seq_len] = True
                    prompt_off = prompt_offsets[i]
                    mask[i, :prompt_off] = False

                    if original_mask is not None and i < original_mask.shape[0]:
                        orig_len = original_mask.shape[1]
                        if orig_len == seq_len:
                            mask[i, :seq_len] &= original_mask[i, :seq_len].to(device)
                        else:
                            overlap = min(orig_len, seq_len - prompt_off)
                            if overlap > 0:
                                mask[i, prompt_off : prompt_off + overlap] &= original_mask[
                                    i, :overlap
                                ].to(device)

            from synthstats.integrations.tinker.losses import compute_combined_tb_subtb_loss

            eos_lp = batch.get("eos_logprob")
            eos_avail = batch.get("eos_available")

            loss, loss_metrics = compute_combined_tb_subtb_loss(
                log_pf=log_probs,
                log_reward=log_rewards,
                logZ=logZ,
                loss_mask=mask,
                eos_logprob=eos_lp,
                eos_available=eos_avail,
                log_sparse_reward=scaled_sparse,
                subtb_alpha=getattr(self.config, "subtb_alpha", 0.1),
                subtb_lambda=getattr(self.config, "subtb_lambda", SUBTB_LAMBDA_DEFAULT),
                max_residual=getattr(self.config, "tb_max_residual", TB_MAX_RESIDUAL_DEFAULT),
            )

            metrics = {
                "loss": loss.item(),
                "logZ": logZ.item(),
                "mean_log_reward": log_rewards.mean().item(),
                "is_multi_turn": float(is_multi_turn),
                **loss_metrics,
            }

            return loss, metrics

        future = self.training_client.forward_backward_custom(data, custom_tb_loss)
        result = future.result()

        adam_params = tinker.AdamParams(
            learning_rate=self.config.learning_rate,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
        )
        self.training_client.optim_step(adam_params)

        # update logZ with analytic gradient: dL/d(logZ) = 2 * residual
        tb_residual = result.metrics.get("tb_residual", 0.0)
        with torch.no_grad():
            self._logZ.add_(-2.0 * self.config.logZ_lr * tb_residual)

        loss_value = result.metrics.get("loss:sum") or result.metrics.get("loss", 0.0)

        self._step += 1

        return {
            **result.metrics,
            "loss": loss_value,
            "logZ": self._logZ.item(),
            "step": step_used,
            "reward_temperature": reward_temperature,
        }

    def parameters(self) -> list[Any]:
        return [self._logZ]

    def save_checkpoint(self, path: str) -> None:
        import torch

        checkpoint = {
            "logZ": self._logZ.data.clone(),
            "step": self._step,
            "config": {
                "model": self.config.model,
                "lora_rank": self.config.lora_rank,
                "learning_rate": self.config.learning_rate,
            },
        }

        if self._training_client is not None:
            try:
                if hasattr(self._training_client, "save_state"):
                    state_name = os.path.splitext(os.path.basename(path))[0]
                    checkpoint["tinker_state"] = self._training_client.save_state(state_name)
            except Exception as e:
                logger.warning(f"Could not save Tinker state: {e}")

        torch.save(checkpoint, path)
        logger.info(f"Saved TinkerTrainer checkpoint to {path}")

    def load_checkpoint(self, path: str, strict: bool = True) -> dict[str, Any]:
        import torch

        # weights_only=False: tinker_state may contain arbitrary types
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        saved_config = checkpoint.get("config", {})
        if strict and saved_config.get("model") != self.config.model:
            raise ValueError(
                f"Model mismatch: checkpoint={saved_config.get('model')}, "
                f"config={self.config.model}"
            )

        if "logZ" not in checkpoint:
            raise ValueError(f"Checkpoint missing 'logZ': {path}")
        self._logZ.data.copy_(checkpoint["logZ"])

        self._step = checkpoint.get("step", self._step)
        logger.info(f"Restored step={self._step}, logZ={self._logZ.item():.4f}")

        if "tinker_state" in checkpoint and self._training_client is not None:
            try:
                self._training_client.load_state(checkpoint["tinker_state"])
                logger.info("Restored Tinker training state from checkpoint")
            except Exception as e:
                logger.warning(f"Could not restore Tinker training state: {e}")

        return checkpoint

    def _validate_batch(self, batch: dict[str, Any]) -> None:
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
            raise ValueError(
                f"Size mismatch: {len(prompts)} prompts vs {len(completions)} completions"
            )

        if hasattr(log_reward, "__len__") and len(log_reward) != len(prompts):
            raise ValueError(f"Size mismatch: {len(prompts)} prompts vs {len(log_reward)} rewards")


@runtime_checkable
class TinkerEnvProtocol(Protocol):

    def initial_observation(self) -> str: ...

    async def step(self, action: Any) -> Any: ...


@dataclass
class MockTinkerClient:

    model: str = "mock-model"

    def sample(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> Any:
        @dataclass
        class MockSampleResult:
            text: str = '{"type": "answer", "payload": "42"}'
            logprobs: list[float] = field(default_factory=lambda: [-0.1, -0.2, -0.15, -0.1, -0.2])

        return MockSampleResult()

    def logprobs(self, prompt: str, completion: str) -> Any:
        @dataclass
        class MockLogprobsResult:
            logprobs: list[float] = field(default_factory=lambda: [-0.1] * len(completion.split()))

        return MockLogprobsResult()


@dataclass
class MockTokenizer:

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return list(range(len(text)))

    def encode_plus(self, text: str, **kwargs: object) -> dict[str, list[tuple[int, int]]]:
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}

    def decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
        return "x" * len(tokens)


@dataclass
class MockTinkerTrainingClient:

    model: str = "mock-model"
    learning_rate: float = 1e-5
    lora_rank: int | None = 32
    _step_count: int = field(default=0, init=False)
    _tokenizer: MockTokenizer = field(default_factory=MockTokenizer, init=False)

    def get_tokenizer(self) -> MockTokenizer:
        return self._tokenizer

    def forward_backward_custom(self, data: list[Any], loss_fn: Any) -> Any:
        import torch

        logprobs_list = []
        for d in data:
            if hasattr(d, "model_input"):
                n_tokens = d.model_input.length if hasattr(d.model_input, "length") else 10
            else:
                completion = d.get("completion", "")
                n_tokens = max(1, len(completion.split()))
            logprobs_list.append(torch.tensor([-0.1] * n_tokens))

        loss, metrics = loss_fn(data, logprobs_list)

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
        self._step_count += 1
