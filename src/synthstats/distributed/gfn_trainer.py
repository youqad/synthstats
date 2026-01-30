"""GFlowNet trainer for distributed training with SkyRL.

RayPPOTrainer subclass implementing SubTB loss. No critic (GFlowNets don't
use value functions), no advantages. Uses replay buffer with on-sample
re-scoring to eliminate off-policy bias. logZ is a learned parameter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch import Tensor

from synthstats.distributed.driver_replay_buffer import (
    BufferEntry,
    DriverGFNReplayBuffer,
)
from synthstats.distributed.scoring import (
    build_response_mask,
    compute_log_probs_with_eos,
    get_stop_token_ids,
)
from synthstats.training.losses.trajectory_balance import (
    compute_modified_subtb_loss,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class ConfigProxy:
    """Wraps dict to support both .get() and attribute access."""

    def __init__(self, base: dict[str, Any]) -> None:
        self._base = base

    def get(self, key: str, default: Any = None) -> Any:
        return self._base.get(key, default)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            base = object.__getattribute__(self, "_base")
            if name in base:
                return base[name]
            raise AttributeError(f"ConfigProxy has no attribute '{name}'")
        base = object.__getattribute__(self, "_base")
        if name in base:
            return base[name]
        raise AttributeError(f"ConfigProxy has no attribute '{name}'")


# try to import SkyRL components (optional)
try:
    from skyrl_train.trainer import RayPPOTrainer

    SKYRL_AVAILABLE = True
except ImportError:
    RayPPOTrainer = object  # type: ignore[assignment,misc]
    SKYRL_AVAILABLE = False


@dataclass
class GFNConfig:
    """GFlowNet-specific configuration."""

    # loss type
    loss_type: str = "modified_subtb"  # "tb" or "modified_subtb"
    subtb_lambda: float = 0.9
    tb_max_residual: float = 100.0

    # logZ learning
    logZ_init: float = 0.0
    lr_logZ: float = 0.001  # typically 10x base LR

    # replay buffer
    replay_buffer_size: int = 10000
    replay_ratio: float = 0.5  # fraction of batch from replay
    prioritized_replay: bool = True
    replay_alpha: float = 1.0
    min_buffer_before_replay: int = 100

    # FlowRL/TBA enhancements (from FlowRL arXiv:2509.15207, TBA arXiv:2503.18929)
    # length normalization: divide log probs by response length (prevents short-response bias)
    length_normalize: bool = True
    # recency ratio (TBA's m parameter): probability of recency-prioritized sampling
    # 0.0 = pure reward-prioritized, 1.0 = pure recency (FIFO), 0.5-0.6 optimal
    recency_ratio: float = 0.5
    # buffer deduplication: hash token sequences to prevent overfitting on duplicates
    deduplicate_buffer: bool = True

    # entropy bonus
    entropy_coef: float = 0.01

    # reward floor for log-transform stability
    reward_floor: float = 1e-4

    # scoring
    score_chunk_size: int | None = None  # for large models (30B+)

    # LOCAL SCORING FIX (tb-local worktree)
    # When True, re-score batch in train_critic_and_policy using local model
    # with gradients enabled. This fixes the gradient flow issue where
    # distributed scoring returns detached tensors.
    # Set to True for debugging/validation, False for worker-side training.
    use_local_scoring_for_training: bool = True

    # temperature
    temperature: float = 1.0

    # VarGrad logZ mode (TBA arXiv:2503.18929)
    # "learned": use nn.Parameter logZ with gradient-based optimization (default)
    # "vargrad": estimate logZ from batch (no learned param needed, requires K>1 responses)
    logz_mode: str = "learned"

    # reward temperature β (FlowRL arXiv:2509.15207)
    # scales log_rewards in loss: higher β means sharper reward signal
    # FlowRL uses β=15; default 1.0 for backward compatibility
    reward_temp: float = 1.0

    # reference model KL regularization (FlowRL, TBA)
    # when True, subtracts (1/|y|)·log π_ref from TB residual
    use_reference_kl: bool = False

    # importance weights for off-policy correction (FlowRL)
    use_importance_weights: bool = False


@dataclass
class GFNBatch:
    """Batch for GFlowNet training. No advantages or values, just SubTB fields."""

    input_ids: Tensor  # [B, L]
    attention_mask: Tensor  # [B, L]
    response_mask: Tensor  # [B, L-1]
    prompt_lengths: Tensor  # [B]
    log_rewards: Tensor  # [B]
    terminated: Tensor  # [B] bool - True if EOS, False if truncated
    temperature: Tensor  # [B]

    # computed during scoring
    log_probs: Tensor | None = None  # [B, L-1]
    eos_logprobs: Tensor | None = None  # [B, L-1]

    is_replay: Tensor | None = None  # [B] bool

    def to(self, device: str | torch.device) -> GFNBatch:
        """Move batch to device."""
        return GFNBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            response_mask=self.response_mask.to(device),
            prompt_lengths=self.prompt_lengths.to(device),
            log_rewards=self.log_rewards.to(device),
            terminated=self.terminated.to(device),
            temperature=self.temperature.to(device),
            log_probs=self.log_probs.to(device) if self.log_probs is not None else None,
            eos_logprobs=self.eos_logprobs.to(device) if self.eos_logprobs is not None else None,
            is_replay=self.is_replay.to(device) if self.is_replay is not None else None,
        )


class GFlowNetTrainer(RayPPOTrainer):  # type: ignore[misc]
    """GFlowNet trainer for distributed SubTB training.

    Subclasses SkyRL's RayPPOTrainer to replace PPO with SubTB loss.
    Key overrides:
    - compute_advantages_and_returns(): Returns GFN-specific data (no advantages)
    - train_critic_and_policy(): Uses SubTB loss instead of PPO

    Args:
        cfg: Hydra config with trainer settings
        gfn_config: GFN-specific configuration (or pulled from cfg.gfn)
        **kwargs: Passed to RayPPOTrainer

    Example:
        >>> trainer = GFlowNetTrainer(cfg, gfn_config=GFNConfig())
        >>> trainer.train()  # runs distributed training
    """

    def __init__(
        self,
        cfg: DictConfig,
        gfn_config: GFNConfig | None = None,
        **kwargs: Any,
    ) -> None:
        # extract GFN config from cfg if not provided
        if gfn_config is None:
            gfn_cfg = getattr(cfg, "gfn", {})
            gfn_config = GFNConfig(
                loss_type=gfn_cfg.get("loss_type", "modified_subtb"),
                subtb_lambda=gfn_cfg.get("subtb_lambda", 0.9),
                tb_max_residual=gfn_cfg.get("tb_max_residual", 100.0),
                logZ_init=gfn_cfg.get("logZ_init", 0.0),
                lr_logZ=gfn_cfg.get("lr_logZ", 0.001),
                replay_buffer_size=gfn_cfg.get("replay_buffer_size", 10000),
                replay_ratio=gfn_cfg.get("replay_ratio", 0.5),
                prioritized_replay=gfn_cfg.get("prioritized_replay", True),
                replay_alpha=gfn_cfg.get("replay_alpha", 1.0),
                min_buffer_before_replay=gfn_cfg.get("min_buffer_before_replay", 100),
                length_normalize=gfn_cfg.get("length_normalize", True),
                recency_ratio=gfn_cfg.get("recency_ratio", 0.5),
                deduplicate_buffer=gfn_cfg.get("deduplicate_buffer", True),
                entropy_coef=gfn_cfg.get("entropy_coef", 0.01),
                reward_floor=gfn_cfg.get("reward_floor", 1e-4),
                score_chunk_size=gfn_cfg.get("score_chunk_size", None),
                use_local_scoring_for_training=gfn_cfg.get("use_local_scoring_for_training", True),
                temperature=gfn_cfg.get("temperature", 1.0),
                logz_mode=gfn_cfg.get("logz_mode", "learned"),
                reward_temp=gfn_cfg.get("reward_temp", 1.0),
                use_reference_kl=gfn_cfg.get("use_reference_kl", False),
                use_importance_weights=gfn_cfg.get("use_importance_weights", False),
            )

        self.gfn_config = gfn_config

        # set device before parent init (parent may not set _device)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # initialize parent (if SkyRL available)
        if SKYRL_AVAILABLE:
            super().__init__(cfg, **kwargs)
            # parent may override _device; keep ours as fallback
        else:
            # standalone mode - minimal init
            self.cfg = cfg

        # learned log partition function
        self.logZ = nn.Parameter(torch.tensor(gfn_config.logZ_init, device=self._device))

        # replay buffer (driver-side) with FlowRL/TBA enhancements
        self.replay_buffer = DriverGFNReplayBuffer(
            capacity=gfn_config.replay_buffer_size,
            prioritized=gfn_config.prioritized_replay,
            alpha=gfn_config.replay_alpha,
            min_entries_before_sample=gfn_config.min_buffer_before_replay,
            deduplicate=gfn_config.deduplicate_buffer,
            recency_ratio=gfn_config.recency_ratio,
        )

        # stop token IDs (populated when tokenizer available)
        self._stop_token_ids: list[int] | None = None

        # tracking
        self._train_step_count = 0

    @property
    def device(self) -> str:
        if hasattr(self, "_device"):
            return self._device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _get_stop_token_ids(self) -> list[int]:
        if self._stop_token_ids is None:
            if hasattr(self, "tokenizer") and self.tokenizer is not None:
                model_name = getattr(self.cfg.trainer.policy.model, "path", None)
                self._stop_token_ids = get_stop_token_ids(self.tokenizer, model_name)
            else:
                # fallback
                self._stop_token_ids = [2]  # common EOS
        return self._stop_token_ids

    def compute_advantages_and_returns(
        self,
        batch: dict[str, Any],
    ) -> dict[str, Any]:
        """GFlowNets don't use advantages - build combined fresh+replay batch instead."""
        # extract fresh batch info
        fresh_input_ids = batch["input_ids"]  # [B_fresh, L]
        fresh_attention_mask = batch["attention_mask"]
        fresh_rewards = batch.get("rewards", batch.get("token_level_rewards"))
        fresh_response_mask = batch.get("response_mask")

        B_fresh = fresh_input_ids.shape[0]
        device = fresh_input_ids.device

        # compute log rewards for fresh trajectories
        fresh_log_rewards = batch.get("log_rewards", batch.get("log_reward"))
        if fresh_log_rewards is None:
            if fresh_rewards is None:
                raise KeyError("Batch missing rewards/log_rewards for GFlowNet training")
            # sum token-level rewards to get trajectory reward, then log-transform
            if fresh_rewards.dim() == 2:
                if fresh_response_mask is not None:
                    trajectory_rewards = (fresh_rewards * fresh_response_mask).sum(dim=-1)
                else:
                    trajectory_rewards = fresh_rewards.sum(dim=-1)
            else:
                trajectory_rewards = fresh_rewards

            reward_floor = max(float(self.gfn_config.reward_floor), 1e-12)
            trajectory_rewards = trajectory_rewards.clamp(min=reward_floor)
            fresh_log_rewards = torch.log(trajectory_rewards).detach()
        elif fresh_log_rewards.dim() == 2:
            # allow token-level log rewards if provided
            if fresh_response_mask is not None:
                fresh_log_rewards = (fresh_log_rewards * fresh_response_mask).sum(dim=-1)
            else:
                fresh_log_rewards = fresh_log_rewards.sum(dim=-1)
        # ensure log rewards are treated as constants
        fresh_log_rewards = fresh_log_rewards.detach()

        # get prompt lengths
        if "prompt_lengths" in batch:
            fresh_prompt_lengths = batch["prompt_lengths"]
        elif fresh_response_mask is not None:
            # estimate from response_mask: first position where mask is 1
            # response_mask[i] is 1 where position i+1 is trainable (response)
            # so prompt_length = index of first 1 + 1
            first_response_pos = fresh_response_mask.argmax(dim=-1)  # [B]
            # handle all-zeros (no response) by using full length
            has_response = fresh_response_mask.any(dim=-1)
            fresh_prompt_lengths = torch.where(
                has_response,
                first_response_pos + 1,  # argmax gives 0-indexed position
                torch.full_like(first_response_pos, fresh_input_ids.shape[1]),
            )
        else:
            # no prompt_lengths or response_mask - log warning and use zeros
            logger.warning(
                "No prompt_lengths or response_mask in batch. Loss may be computed incorrectly."
            )
            fresh_prompt_lengths = torch.zeros(B_fresh, device=device, dtype=torch.long)

        # determine replay batch size
        total_batch = B_fresh
        if (
            self.gfn_config.replay_ratio > 0
            and len(self.replay_buffer) >= self.gfn_config.min_buffer_before_replay
        ):
            replay_count = int(total_batch * self.gfn_config.replay_ratio)
            fresh_count = total_batch - replay_count
        else:
            replay_count = 0
            fresh_count = B_fresh

        # sample from replay buffer
        replay_entries: list[BufferEntry] = []
        if replay_count > 0:
            try:
                replay_entries = self.replay_buffer.sample(replay_count)
            except ValueError:
                replay_count = 0
                fresh_count = B_fresh

        # add fresh trajectories to replay buffer
        self._add_fresh_to_buffer(
            fresh_input_ids,
            fresh_prompt_lengths,
            fresh_log_rewards,
            batch.get("terminated"),
            batch.get("temperature"),
        )

        # build combined batch
        combined = self._build_combined_batch(
            fresh_input_ids=fresh_input_ids[:fresh_count],
            fresh_attention_mask=fresh_attention_mask[:fresh_count],
            fresh_prompt_lengths=fresh_prompt_lengths[:fresh_count],
            fresh_log_rewards=fresh_log_rewards[:fresh_count],
            fresh_terminated=batch.get(
                "terminated", torch.ones(fresh_count, device=device, dtype=torch.bool)
            )[:fresh_count],
            fresh_temperature=batch.get(
                "temperature",
                torch.full((fresh_count,), self.gfn_config.temperature, device=device),
            )[:fresh_count],
            replay_entries=replay_entries,
            device=device,
        )

        # score combined batch (compute log_probs and eos_logprobs)
        combined = self._score_batch(combined)

        # attach logZ for loss computation
        combined["logZ"] = self.logZ

        # log diagnostics
        if replay_count > 0:
            staleness = self.replay_buffer.get_staleness_stats()
            logger.debug(
                f"GFN batch: {fresh_count} fresh + {replay_count} replay, "
                f"mean_staleness={staleness['mean_staleness']:.1f}"
            )

        return combined

    def _add_fresh_to_buffer(
        self,
        input_ids: Tensor,
        prompt_lengths: Tensor,
        log_rewards: Tensor,
        terminated: Tensor | None,
        temperature: Tensor | None,
    ) -> None:
        # detach to avoid keeping grad graph in CPU memory
        B = input_ids.shape[0]

        input_ids_list = input_ids.detach().cpu().tolist()
        prompt_lengths_list = prompt_lengths.detach().cpu().tolist()
        log_rewards_list = log_rewards.detach().cpu().tolist()

        terminated_list = (
            terminated.detach().cpu().tolist() if terminated is not None else [True] * B
        )
        temperature_list = (
            temperature.detach().cpu().tolist()
            if temperature is not None
            else [self.gfn_config.temperature] * B
        )

        self.replay_buffer.add_from_batch(
            input_ids=input_ids_list,
            prompt_lengths=prompt_lengths_list,
            log_rewards=log_rewards_list,
            terminated=terminated_list,
            temperatures=temperature_list,
        )

    def _build_combined_batch(
        self,
        fresh_input_ids: Tensor,
        fresh_attention_mask: Tensor,
        fresh_prompt_lengths: Tensor,
        fresh_log_rewards: Tensor,
        fresh_terminated: Tensor,
        fresh_temperature: Tensor,
        replay_entries: list[BufferEntry],
        device: torch.device | str,
    ) -> dict[str, Any]:
        B_fresh = fresh_input_ids.shape[0]
        B_replay = len(replay_entries)

        if B_replay == 0:
            fresh_input_ids.shape[1]
            response_mask = build_response_mask(
                fresh_input_ids,
                fresh_prompt_lengths,
                attention_mask=fresh_attention_mask,
            )
            return {
                "input_ids": fresh_input_ids,
                "attention_mask": fresh_attention_mask,
                "response_mask": response_mask,
                "prompt_lengths": fresh_prompt_lengths,
                "log_rewards": fresh_log_rewards,
                "terminated": fresh_terminated,
                "temperature": fresh_temperature,
                "is_replay": torch.zeros(B_fresh, device=device, dtype=torch.bool),
            }

        # pad replay entries to same length
        max_len = max(
            fresh_input_ids.shape[1],
            max(len(e.prompt_token_ids) + len(e.action_token_ids) for e in replay_entries),
        )

        # get pad token (prefer pad_token_id, fall back to eos_token_id, then 0)
        # NOTE: -100 is invalid for embeddings; use 0 as safe fallback
        pad_token_id = 0
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            if self.tokenizer.pad_token_id is not None:
                pad_token_id = self.tokenizer.pad_token_id
            elif self.tokenizer.eos_token_id is not None:
                pad_token_id = self.tokenizer.eos_token_id

        # build replay tensors
        replay_input_ids = []
        replay_attention_mask = []
        replay_prompt_lengths = []
        replay_log_rewards = []
        replay_terminated = []
        replay_temperature = []

        for entry in replay_entries:
            seq = entry.prompt_token_ids + entry.action_token_ids
            pad_len = max_len - len(seq)
            padded_seq = seq + [pad_token_id] * pad_len
            attention = [1] * len(seq) + [0] * pad_len

            replay_input_ids.append(padded_seq)
            replay_attention_mask.append(attention)
            replay_prompt_lengths.append(len(entry.prompt_token_ids))
            replay_log_rewards.append(entry.log_reward)
            replay_terminated.append(entry.terminated)
            replay_temperature.append(entry.temperature)

        replay_input_ids_t = torch.tensor(replay_input_ids, device=device, dtype=torch.long)
        replay_attention_mask_t = torch.tensor(
            replay_attention_mask, device=device, dtype=torch.long
        )
        replay_prompt_lengths_t = torch.tensor(
            replay_prompt_lengths, device=device, dtype=torch.long
        )
        replay_log_rewards_t = torch.tensor(replay_log_rewards, device=device, dtype=torch.float)
        replay_terminated_t = torch.tensor(replay_terminated, device=device, dtype=torch.bool)
        replay_temperature_t = torch.tensor(replay_temperature, device=device, dtype=torch.float)

        # pad fresh to same max_len if needed
        if fresh_input_ids.shape[1] < max_len:
            pad_size = max_len - fresh_input_ids.shape[1]
            fresh_input_ids = torch.nn.functional.pad(
                fresh_input_ids, (0, pad_size), value=pad_token_id
            )
            fresh_attention_mask = torch.nn.functional.pad(
                fresh_attention_mask, (0, pad_size), value=0
            )

        combined_input_ids = torch.cat([fresh_input_ids, replay_input_ids_t], dim=0)
        combined_attention_mask = torch.cat([fresh_attention_mask, replay_attention_mask_t], dim=0)
        combined_prompt_lengths = torch.cat([fresh_prompt_lengths, replay_prompt_lengths_t], dim=0)
        combined_log_rewards = torch.cat([fresh_log_rewards, replay_log_rewards_t], dim=0)
        combined_terminated = torch.cat([fresh_terminated, replay_terminated_t], dim=0)
        combined_temperature = torch.cat([fresh_temperature, replay_temperature_t], dim=0)

        # build response mask using attention_mask (avoids pad==eos masking bug)
        response_mask = build_response_mask(
            combined_input_ids,
            combined_prompt_lengths,
            attention_mask=combined_attention_mask,
        )

        is_replay = torch.cat(
            [
                torch.zeros(B_fresh, device=device, dtype=torch.bool),
                torch.ones(B_replay, device=device, dtype=torch.bool),
            ]
        )

        return {
            "input_ids": combined_input_ids,
            "attention_mask": combined_attention_mask,
            "response_mask": response_mask,
            "prompt_lengths": combined_prompt_lengths,
            "log_rewards": combined_log_rewards,
            "terminated": combined_terminated,
            "temperature": combined_temperature,
            "is_replay": is_replay,
        }

    def _score_batch_distributed(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        response_mask: Tensor,
        temperature: Tensor,
        stop_token_ids: list[int],
    ) -> tuple[Tensor, Tensor]:
        """Score batch via distributed actor workers. Returns (log_probs, eos_logprobs)."""
        B, L = input_ids.shape
        device = input_ids.device

        # build DataProto for dispatch to workers
        # import here to avoid circular deps and handle missing SkyRL
        try:
            from tensordict import TensorDict
            from verl import DataProto
        except ImportError:
            # fall back to dict-based dispatch if TensorDict unavailable
            logger.debug("TensorDict/DataProto unavailable, using dict dispatch")
            return self._score_batch_via_actor_group_dict(
                input_ids, attention_mask, response_mask, temperature, stop_token_ids
            )

        # construct TensorDict batch
        batch_td = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "temperature": temperature,
            },
            batch_size=[B],
            device=device,
        )

        # add stop token ids to meta_info
        data = DataProto(
            batch=batch_td,
            meta_info={
                "stop_token_ids": stop_token_ids,
                "compute_eos_logprobs": True,  # signal to compute EOS logprobs
            },
        )

        # dispatch to actor group using DP_COMPUTE_PROTO mode
        # the actor group will split the batch across workers
        result = self.policy_actor_group.compute_log_probs(data)

        # extract results from DataProto
        log_probs = result.batch["log_probs"]
        eos_logprobs = result.batch["eos_logprobs"]

        # warn if detached (Ray RPC loses grad_fn) and local re-scoring disabled
        if not log_probs.requires_grad and not self.gfn_config.use_local_scoring_for_training:
            logger.warning(
                "Distributed scoring returned detached tensors. "
                "Set use_local_scoring_for_training=True to fix gradient flow."
            )

        return log_probs, eos_logprobs

    def _score_batch_via_actor_group_dict(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        response_mask: Tensor,
        temperature: Tensor,
        stop_token_ids: list[int],
    ) -> tuple[Tensor, Tensor]:
        """Dict-based fallback when DataProto unavailable."""
        batch_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "temperature": temperature,
            "stop_token_ids": stop_token_ids,
        }

        # try different method names that actor groups might support
        if hasattr(self.policy_actor_group, "compute_log_probs"):
            result = self.policy_actor_group.compute_log_probs(batch_dict)
        elif hasattr(self.policy_actor_group, "forward"):
            result = self.policy_actor_group.forward(batch_dict)
        else:
            raise NotImplementedError("Actor group has no compute_log_probs or forward method")

        # extract results - handle both dict and DataProto returns
        if hasattr(result, "batch"):
            log_probs = result.batch["log_probs"]
            eos_logprobs = result.batch["eos_logprobs"]
        else:
            log_probs = result["log_probs"]
            eos_logprobs = result["eos_logprobs"]

        return log_probs, eos_logprobs

    def _rescore_batch_with_gradients(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Re-score with gradients so loss.backward() updates policy parameters."""
        if not hasattr(self, "policy_model") or self.policy_model is None:
            logger.warning(
                "No local policy_model available for gradient scoring. "
                "Using pre-computed log_probs (gradient flow may be broken)."
            )
            return batch

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        response_mask = batch["response_mask"]
        temperature = batch["temperature"]

        stop_token_ids = self._get_stop_token_ids()

        log_probs, eos_logprobs = compute_log_probs_with_eos(
            model=self.policy_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            response_mask=response_mask,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
            chunk_size=self.gfn_config.score_chunk_size,
            no_grad=False,  # keep gradients
        )

        # replace pre-computed (detached) with gradient-enabled tensors
        batch["log_probs"] = log_probs
        batch["eos_logprobs"] = eos_logprobs

        return batch

    def _score_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Add log_probs and eos_logprobs to batch via forward pass.

        Uses distributed scoring via actor workers if available (SkyRL mode),
        otherwise falls back to local model scoring (standalone mode).

        NOTE: This scoring uses no_grad=True for replay efficiency. For training
        with gradient flow, use_local_scoring_for_training=True will call
        _rescore_batch_with_gradients in train_critic_and_policy.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        response_mask = batch["response_mask"]
        temperature = batch["temperature"]  # [B] per-sample temperatures

        stop_token_ids = self._get_stop_token_ids()

        # try distributed scoring via actor workers
        if (
            SKYRL_AVAILABLE
            and hasattr(self, "policy_actor_group")
            and self.policy_actor_group is not None
        ):
            try:
                log_probs, eos_logprobs = self._score_batch_distributed(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    response_mask=response_mask,
                    temperature=temperature,
                    stop_token_ids=stop_token_ids,
                )
                batch["log_probs"] = log_probs
                batch["eos_logprobs"] = eos_logprobs
                return batch
            except Exception as e:
                logger.warning(f"Distributed scoring failed, falling back to local: {e}")

        # local scoring (standalone mode or fallback)
        # use no_grad=True for efficiency; training re-scores with gradients
        if hasattr(self, "policy_model") and self.policy_model is not None:
            log_probs, eos_logprobs = compute_log_probs_with_eos(
                model=self.policy_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                response_mask=response_mask,
                stop_token_ids=stop_token_ids,
                temperature=temperature,  # pass per-sample temperatures
                chunk_size=self.gfn_config.score_chunk_size,
                no_grad=True,  # efficiency: training will re-score with gradients
            )
            batch["log_probs"] = log_probs
            batch["eos_logprobs"] = eos_logprobs
        else:
            # no model available - this should not happen in training
            logger.error("No policy model available for scoring")
            B, L = input_ids.shape
            batch["log_probs"] = torch.zeros(B, L - 1, device=input_ids.device)
            batch["eos_logprobs"] = torch.zeros(B, L - 1, device=input_ids.device)

        return batch

    def train_critic_and_policy(
        self,
        batch: dict[str, Any],
    ) -> dict[str, float]:
        """Use SubTB loss: (logZ + sum(log_probs) + log_eos - log_reward)^2

        CRITICAL FIX (tb-local worktree):
        When use_local_scoring_for_training=True, we RE-SCORE the batch using
        the local model with gradients enabled. This fixes the bug where
        distributed scoring returns detached tensors, causing the policy
        model to never receive gradients.

        The scoring done in compute_advantages_and_returns uses no_grad=True
        (for efficiency with replay samples). Here we re-score WITH gradients
        so that loss.backward() updates both logZ AND the policy model.
        """
        self._train_step_count += 1

        # GRADIENT FIX: Re-score batch using local model with gradients
        if self.gfn_config.use_local_scoring_for_training:
            batch = self._rescore_batch_with_gradients(batch)

        # extract tensors
        log_probs = batch["log_probs"]  # [B, T]
        eos_logprobs = batch.get("eos_logprobs")  # [B, T] or None
        log_rewards = batch["log_rewards"]  # [B]
        response_mask = batch["response_mask"]  # [B, T]
        logZ = batch["logZ"]  # scalar parameter

        B, T = log_probs.shape
        device = log_probs.device

        # get reference model log probs if KL regularization enabled
        ref_log_probs = batch.get("ref_log_probs")  # [B, T] or None

        # dispatch: augmented TB path handles vargrad, KL, importance weights,
        # reward tempering. SubTB always uses the basic path (no augmented features).
        use_augmented_tb_loss = self.gfn_config.loss_type != "modified_subtb" and (
            self.gfn_config.logz_mode == "vargrad"
            or self.gfn_config.use_reference_kl
            or self.gfn_config.use_importance_weights
            or self.gfn_config.reward_temp != 1.0
        )

        if self.gfn_config.loss_type == "modified_subtb" and (
            self.gfn_config.use_reference_kl
            or self.gfn_config.use_importance_weights
            or self.gfn_config.reward_temp != 1.0
        ):
            logger.warning(
                "SubTB does not support reference KL, importance weights, or "
                "reward_temp != 1.0; these settings will be ignored"
            )

        if use_augmented_tb_loss:
            loss, loss_metrics = self._compute_augmented_tb_loss(
                log_probs=log_probs,
                log_rewards=log_rewards,
                response_mask=response_mask,
                logZ=logZ,
                ref_log_probs=ref_log_probs,
                old_log_probs=batch.get("old_log_probs"),
            )
        else:
            # legacy path: use original TB/SubTB losses with correct
            # length normalization applied at the residual level
            loss, loss_metrics = self._compute_legacy_loss(
                log_probs=log_probs,
                eos_logprobs=eos_logprobs,
                log_rewards=log_rewards,
                response_mask=response_mask,
                logZ=logZ,
            )

        # entropy bonus (use raw log_probs — entropy should not be length-normalized)
        entropy_loss = torch.tensor(0.0, device=device)
        if self.gfn_config.entropy_coef > 0:
            response_counts = response_mask.float().sum(dim=-1).clamp(min=1)
            masked_neg_log_probs = -log_probs * response_mask.float()
            per_seq_entropy = masked_neg_log_probs.sum(dim=-1) / response_counts
            entropy_loss = -self.gfn_config.entropy_coef * per_seq_entropy.mean()

        total_loss = loss + entropy_loss

        # backward and step
        if hasattr(self, "optimizer") and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()

            # gradient clipping
            max_grad_norm = getattr(self.gfn_config, "max_grad_norm", None)
            if max_grad_norm is None and hasattr(self, "cfg"):
                try:
                    max_grad_norm = self.cfg.trainer.max_grad_norm
                except (AttributeError, TypeError):
                    max_grad_norm = 1.0
            if isinstance(max_grad_norm, (int, float)) and max_grad_norm > 0:
                all_params = [p for g in self.get_optimizer_param_groups() for p in g["params"]]
                torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)

            self.optimizer.step()

            # sync updated weights to distributed actors for next rollout
            self._sync_weights_to_actors()

        # update replay buffer policy version
        self.replay_buffer.increment_policy_version()

        # collect metrics
        metrics = {
            "loss": total_loss.item(),
            "tb_loss": loss.item(),
            "entropy_loss": entropy_loss.item(),
            "logZ": logZ.item(),
            "mean_log_reward": log_rewards.mean().item(),
            "mean_log_prob": (log_probs * response_mask.float()).sum(dim=-1).mean().item(),
        }

        # add augmented TB loss metrics
        if use_augmented_tb_loss and loss_metrics:
            metrics.update({f"augmented_tb/{k}": v for k, v in loss_metrics.items()})

        # add replay buffer stats
        buffer_stats = self.replay_buffer.get_stats()
        metrics["buffer_size"] = buffer_stats["size"]
        metrics["buffer_mean_staleness"] = buffer_stats["mean_staleness"]

        # track replay ratio actually used
        is_replay = batch.get("is_replay")
        if is_replay is not None:
            metrics["actual_replay_ratio"] = is_replay.float().mean().item()

        return metrics

    def _sync_weights_to_actors(self) -> None:
        """Push updated driver model weights to distributed actor workers.

        After optimizer.step() updates self.policy_model, actor copies become
        stale. This syncs the new weights so the next rollout uses the current policy.
        """
        if not (hasattr(self, "policy_actor_group") and self.policy_actor_group is not None):
            return

        try:
            state_dict = {k: v.detach().cpu() for k, v in self.policy_model.state_dict().items()}
            self.policy_actor_group.set_weights(state_dict)
        except AttributeError:
            # actor group may not support set_weights (e.g. in tests)
            logger.debug("policy_actor_group has no set_weights method, skipping sync")
        except Exception as e:
            logger.warning(f"Weight sync to actors failed: {e}")

    def _compute_augmented_tb_loss(
        self,
        log_probs: Tensor,
        log_rewards: Tensor,
        response_mask: Tensor,
        logZ: Tensor,
        ref_log_probs: Tensor | None = None,
        old_log_probs: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute loss using FlowRL/TBA v2 functions.

        Dispatches to the appropriate v2 loss based on config:
        - vargrad mode: uses compute_vargrad_tb_loss (no learned logZ)
        - flowrl mode (importance weights + reference KL): uses compute_flowrl_loss
        - kl mode: uses compute_tb_loss_with_kl
        """
        from synthstats.training.losses.trajectory_balance import (
            compute_flowrl_loss,
            compute_tb_loss_with_kl,
            compute_vargrad_tb_loss,
        )

        if self.gfn_config.logz_mode == "vargrad":
            return compute_vargrad_tb_loss(
                log_probs=log_probs,
                log_rewards=log_rewards,
                response_mask=response_mask,
                ref_log_probs=ref_log_probs if self.gfn_config.use_reference_kl else None,
                length_normalize=self.gfn_config.length_normalize,
                reward_temp=self.gfn_config.reward_temp,
                max_residual=self.gfn_config.tb_max_residual,
                prompt_ids=None,  # single-prompt batches for now
            )

        if self.gfn_config.use_importance_weights:
            return compute_flowrl_loss(
                log_probs=log_probs,
                log_rewards=log_rewards,
                response_mask=response_mask,
                logZ=logZ,
                ref_log_probs=ref_log_probs if self.gfn_config.use_reference_kl else None,
                old_log_probs=old_log_probs,
                length_normalize=self.gfn_config.length_normalize,
                reward_temp=self.gfn_config.reward_temp,
                max_residual=self.gfn_config.tb_max_residual,
                use_importance_weights=True,
            )

        # default v2: TB with KL and/or reward temp
        loss, metrics = compute_tb_loss_with_kl(
            log_probs=log_probs,
            log_rewards=log_rewards,
            response_mask=response_mask,
            logZ=logZ,
            ref_log_probs=ref_log_probs if self.gfn_config.use_reference_kl else None,
            length_normalize=self.gfn_config.length_normalize,
            reward_temp=self.gfn_config.reward_temp,
            max_residual=self.gfn_config.tb_max_residual,
        )
        return loss, metrics

    def _compute_legacy_loss(
        self,
        log_probs: Tensor,
        eos_logprobs: Tensor | None,
        log_rewards: Tensor,
        response_mask: Tensor,
        logZ: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute loss using legacy TB/SubTB with correct length normalization.

        FlowRL normalizes only the log-probability sum by response length,
        keeping logZ and log_R at original scale. For SubTB, normalization
        is skipped entirely because lambda-weighting handles sequence length.

        For TB: loss = mean( (logZ + (1/L)·Σlog_π - log_R)² )
        For SubTB: loss computed from raw (unnormalized) per-token values
        """
        B, T = log_probs.shape
        mask_f = response_mask.float()
        max_residual = self.gfn_config.tb_max_residual

        if self.gfn_config.loss_type == "modified_subtb" and eos_logprobs is not None:
            # SubTB: pass raw log_probs and eos_logprobs (no length normalization)
            # SubTB's lambda-weighting handles sequence length sensitivity
            log_rewards_broadcast = log_rewards.unsqueeze(1).expand(B, T)
            loss_config = {
                "logZ": logZ.item(),
                "_logZ_tensor": logZ,
                "_eos_logprobs": eos_logprobs,
                "subtb_lambda": self.gfn_config.subtb_lambda,
                "tb_max_residual": max_residual,
            }
            config_proxy = ConfigProxy(loss_config)
            loss, _ = compute_modified_subtb_loss(
                log_probs=log_probs,
                old_log_probs=log_probs.detach(),
                advantages=log_rewards_broadcast,
                config=config_proxy,
                loss_mask=response_mask,
            )
        else:
            # Vanilla TB with FlowRL length normalization:
            # L = (logZ + (1/|y|)·Σlog_π - log_R)²
            # Only the log-probability sum is normalized by L, not logZ or log_R
            masked_logprobs = (log_probs * mask_f).sum(dim=-1)  # [B]
            response_lengths = mask_f.sum(dim=-1).clamp(min=1.0)  # [B]

            if self.gfn_config.length_normalize:
                masked_logprobs = masked_logprobs / response_lengths

            # handle NaN/inf in log_rewards
            safe_log_rewards = torch.where(
                torch.isfinite(log_rewards),
                log_rewards,
                torch.full_like(log_rewards, -max_residual),
            )

            residual = logZ + masked_logprobs - safe_log_rewards  # [B]
            residual = residual.clamp(-max_residual, max_residual)
            loss = (residual**2).mean()

        return loss, {}

    def get_optimizer_param_groups(self) -> list[dict[str, Any]]:
        """Get parameter groups with separate LR for logZ.

        Returns list of param groups suitable for optimizer construction:
        - Model parameters with base LR
        - logZ with higher LR (typically 10x) -- skipped in VarGrad mode
        """
        base_lr = getattr(self.cfg.trainer, "lr", 1e-4)

        groups = []

        # model parameters
        if hasattr(self, "policy_model") and self.policy_model is not None:
            groups.append(
                {
                    "params": self.policy_model.parameters(),
                    "lr": base_lr,
                }
            )

        # logZ with higher LR (skip in VarGrad mode where logZ is estimated)
        if self.gfn_config.logz_mode != "vargrad":
            groups.append(
                {
                    "params": [self.logZ],
                    "lr": self.gfn_config.lr_logZ,
                }
            )

        return groups

    def save_checkpoint(self, checkpoint_dir: str | Path) -> None:
        """Save checkpoint including logZ and buffer state.

        Extends parent checkpoint to save GFN-specific state that isn't part
        of the model weights (logZ, replay buffer stats, policy version).

        Args:
            checkpoint_dir: Directory to save checkpoint files
        """
        from pathlib import Path

        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # save logZ separately (not part of model state_dict)
        logz_path = checkpoint_dir / "logZ.pt"
        torch.save(
            {
                "logZ": self.logZ.data.cpu(),
                "policy_version": self.replay_buffer.policy_version,
                "train_step_count": self._train_step_count,
            },
            logz_path,
        )

        logger.info(f"Saved logZ checkpoint to {logz_path}")

        # call parent save if available (saves model weights)
        if SKYRL_AVAILABLE and hasattr(super(), "save_checkpoint"):
            super().save_checkpoint(checkpoint_dir)

    def load_checkpoint(self, checkpoint_dir: str | Path) -> None:
        """Load checkpoint including logZ and buffer state.

        Args:
            checkpoint_dir: Directory containing checkpoint files
        """
        from pathlib import Path

        checkpoint_dir = Path(checkpoint_dir)

        # load logZ
        logz_path = checkpoint_dir / "logZ.pt"
        if logz_path.exists():
            state = torch.load(logz_path, map_location=self.device)
            self.logZ.data.copy_(state["logZ"].to(self.device))

            # restore policy version for staleness tracking
            if "policy_version" in state:
                self.replay_buffer._policy_version = state["policy_version"]
            if "train_step_count" in state:
                self._train_step_count = state["train_step_count"]

            logger.info(
                f"Loaded logZ checkpoint: logZ={self.logZ.item():.4f}, "
                f"policy_version={self.replay_buffer.policy_version}"
            )
        else:
            logger.warning(f"No logZ checkpoint found at {logz_path}")

        # call parent load if available (loads model weights)
        if SKYRL_AVAILABLE and hasattr(super(), "load_checkpoint"):
            super().load_checkpoint(checkpoint_dir)
