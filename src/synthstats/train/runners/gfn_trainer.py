"""GFlowNet training orchestrator with SubTB loss, replay buffer, and learned logZ."""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from synthstats.core.constants import LOGZ_LR_DEFAULT, REWARD_FLOOR_DEFAULT
from synthstats.core.types import Trajectory
from synthstats.runtime.codecs import ActionCodec
from synthstats.runtime.rollout import RolloutConfig, rollout_episode
from synthstats.train.data.replay import ReplayBuffer
from synthstats.train.objectives.losses import subtb_loss


@dataclass
class TrainerConfig:

    batch_size: int = 4
    learning_rate: float = 1e-4
    logZ_lr: float = LOGZ_LR_DEFAULT
    max_episodes: int = 1000
    max_steps_per_episode: int = 10
    replay_buffer_capacity: int = 1000
    replay_sample_ratio: float = 0.5  # fraction of batch from replay
    log_interval: int = 10
    eval_interval: int = 100
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    normalize_by_length: bool = False
    seed: int = 42


@dataclass
class TrainMetrics:

    loss: float
    logZ: float
    avg_reward: float
    num_episodes: int
    replay_fraction: float = 0.0


class Trainer:
    """GFlowNet training orchestrator.

    Caller provides policy, task, codec, and judge. Caller handles model loading,
    WandB logging (via callback), and checkpointing.
    """

    def __init__(
        self,
        config: TrainerConfig,
        policy: Any,  # implements Policy protocol
        task: Any,  # implements Task protocol
        codec: ActionCodec,
        judge: Any,  # implements Judge protocol
        executor_registry: dict[str, Any] | None = None,
        device: str = "cpu",
        log_callback: Callable[[TrainMetrics, int], None] | None = None,
    ):
        self.config = config
        self.policy = policy
        self.task = task
        self.codec = codec
        self.executors = executor_registry or {}
        self.judge = judge
        self.device = torch.device(device)
        self.log_callback = log_callback

        random.seed(config.seed)
        torch.manual_seed(config.seed)

        self._logZ = nn.Parameter(torch.tensor(0.0, device=self.device))

        self.replay_buffer = ReplayBuffer(capacity=config.replay_buffer_capacity)

        param_groups = [
            {"params": [self._logZ], "lr": config.logZ_lr},
        ]

        if hasattr(self.policy, "parameters") and callable(self.policy.parameters):
            try:
                policy_params = list(self.policy.parameters())
                if policy_params:
                    param_groups.append({"params": policy_params, "lr": config.learning_rate})
            except Exception:
                import warnings

                warnings.warn(
                    "policy.parameters() failed; only logZ will be optimized",
                    stacklevel=2,
                )

        self.optimizer = torch.optim.Adam(param_groups)

        self._step = 0
        self._accumulated_loss = 0.0
        self._accumulation_count = 0

    @property
    def logZ(self) -> float:
        return self._logZ.item()

    def _collect_episodes(self, num_episodes: int) -> list[Trajectory]:
        trajectories = []
        rollout_cfg = RolloutConfig(max_steps=self.config.max_steps_per_episode)

        for _ in range(num_episodes):
            traj = rollout_episode(
                task=self.task,
                policy=self.policy,
                codec=self.codec,
                executors=self.executors,
                judge=self.judge,
                cfg=rollout_cfg,
            )
            trajectories.append(traj)
            self.replay_buffer.add(traj)

        return trajectories

    def _recompute_logprobs_differentiable(self, traj: Trajectory) -> list[torch.Tensor]:
        result = []

        assistant_indices = [i for i, m in enumerate(traj.messages) if m.role == "assistant"]

        for gen_idx, token_ids in enumerate(traj.token_ids):
            if not token_ids:
                result.append(torch.tensor([], device=self.device))
                continue

            if gen_idx < len(assistant_indices):
                context_end = assistant_indices[gen_idx]
            else:
                context_end = len(traj.messages)

            context_messages = [m for m in traj.messages[:context_end] if m.role != "assistant"]

            if not context_messages:
                context_messages = [m for m in traj.messages if m.role in ("system", "user")][:2]

            logprobs_tensor = self.policy.score_tokens(context_messages, token_ids)
            result.append(logprobs_tensor)

        return result

    def _trajectories_to_tensors(
        self, trajectories: list[Trajectory]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(trajectories)
        use_differentiable = hasattr(self.policy, "score_tokens") and callable(
            getattr(self.policy, "score_tokens", None)
        )

        all_logprobs = []
        all_masks = []
        log_rewards = []

        for traj in trajectories:
            if use_differentiable:
                gen_logprobs_tensors = self._recompute_logprobs_differentiable(traj)
                if gen_logprobs_tensors:
                    flat_logprobs = torch.cat(gen_logprobs_tensors)
                else:
                    flat_logprobs = torch.tensor([], device=self.device)
            else:
                flat_logprobs_list = []
                for gen_logprobs in traj.token_logprobs:
                    flat_logprobs_list.extend(gen_logprobs)
                flat_logprobs = torch.tensor(flat_logprobs_list, device=self.device)

            flat_mask = []
            for gen_mask in traj.loss_mask:
                flat_mask.extend(gen_mask)
            all_logprobs.append(flat_logprobs)
            all_masks.append(flat_mask)

            reward = max(traj.reward.total, REWARD_FLOOR_DEFAULT)
            log_rewards.append(math.log(reward))

        max_len = max(len(lp) for lp in all_logprobs) if all_logprobs else 1

        log_probs_tensor = torch.zeros(batch_size, max_len, device=self.device)
        mask_tensor = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self.device)

        for i, (lps, masks) in enumerate(zip(all_logprobs, all_masks, strict=False)):
            seq_len = len(lps)
            if seq_len > 0:
                log_probs_tensor[i, :seq_len] = lps
                mask_tensor[i, :seq_len] = torch.tensor(masks, device=self.device)

        log_rewards_tensor = torch.tensor(log_rewards, device=self.device)

        return log_probs_tensor, mask_tensor, log_rewards_tensor

    def train_step(self) -> TrainMetrics:
        num_fresh = self.config.batch_size
        num_replay = 0
        replay_fraction = 0.0

        if len(self.replay_buffer) >= self.config.batch_size:
            num_replay = int(self.config.batch_size * self.config.replay_sample_ratio)
            num_fresh = self.config.batch_size - num_replay
            replay_fraction = num_replay / self.config.batch_size

        fresh_trajectories = self._collect_episodes(num_fresh)

        replay_trajectories = []
        if num_replay > 0:
            replay_trajectories = self.replay_buffer.sample(num_replay)

        all_trajectories = fresh_trajectories + replay_trajectories

        log_probs, loss_mask, log_rewards = self._trajectories_to_tensors(all_trajectories)

        loss = subtb_loss(
            log_probs,
            loss_mask,
            log_rewards,
            self._logZ,
            normalize_by_length=self.config.normalize_by_length,
        )

        self._accumulated_loss += loss.item()
        self._accumulation_count += 1

        scaled_loss = loss / self.config.gradient_accumulation_steps
        scaled_loss.backward()

        if self._accumulation_count >= self.config.gradient_accumulation_steps:
            all_params = [self._logZ]
            if hasattr(self.policy, "parameters") and callable(self.policy.parameters):
                try:
                    all_params.extend(list(self.policy.parameters()))
                except Exception:
                    import warnings

                    warnings.warn("policy.parameters() failed during grad clipping", stacklevel=2)
            torch.nn.utils.clip_grad_norm_(all_params, self.config.max_grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()

            self._accumulation_count = 0

        avg_reward = sum(t.reward.total for t in all_trajectories) / len(all_trajectories)
        avg_loss = self._accumulated_loss / max(1, self._accumulation_count + 1)
        if self._accumulation_count == 0:
            avg_loss = loss.item()
            self._accumulated_loss = 0.0

        self._step += 1

        metrics = TrainMetrics(
            loss=avg_loss,
            logZ=self.logZ,
            avg_reward=avg_reward,
            num_episodes=len(all_trajectories),
            replay_fraction=replay_fraction,
        )

        if self.log_callback is not None:
            self.log_callback(metrics, self._step)

        return metrics

    def train(self, num_steps: int | None = None) -> list[TrainMetrics]:
        steps = num_steps if num_steps is not None else self.config.max_episodes
        all_metrics = []

        for _ in range(steps):
            metrics = self.train_step()
            all_metrics.append(metrics)

        return all_metrics

    def evaluate(self, num_episodes: int = 10) -> dict[str, float]:
        rollout_cfg = RolloutConfig(max_steps=self.config.max_steps_per_episode)
        rewards = []
        successes = 0

        with torch.no_grad():
            for _ in range(num_episodes):
                traj = rollout_episode(
                    task=self.task,
                    policy=self.policy,
                    codec=self.codec,
                    executors=self.executors,
                    judge=self.judge,
                    cfg=rollout_cfg,
                )
                rewards.append(traj.reward.total)
                if traj.reward.total > 0:
                    successes += 1

        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        success_rate = successes / num_episodes if num_episodes > 0 else 0.0

        return {
            "avg_reward": avg_reward,
            "success_rate": success_rate,
            "num_episodes": num_episodes,
            "logZ": self.logZ,
        }
