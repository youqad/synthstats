"""Main training orchestrator for GFlowNet-style RL.

Coordinates episode rollout, SubTB loss computation, replay buffer
management, and gradient updates with learned logZ.
"""

import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from synthstats.core.types import Message, Reward, StepResult, Trajectory
from synthstats.runtime.codecs import ActionCodec
from synthstats.runtime.rollout import RolloutConfig, rollout_episode
from synthstats.training.buffers.replay import ReplayBuffer
from synthstats.training.losses.tb_loss import subtb_loss


@dataclass
class TrainerConfig:
    """Configuration for the trainer."""

    batch_size: int = 4
    learning_rate: float = 1e-4
    logZ_learning_rate: float = 1e-2  # often higher for logZ
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
    """Metrics from a training step."""

    loss: float
    logZ: float
    avg_reward: float
    num_episodes: int
    replay_fraction: float = 0.0


@dataclass
class ToyState:
    """State for ToyTask."""

    step: int = 0
    done: bool = False


class ToyTask:
    """Minimal task for testing trainer without real environment dependencies.

    Always terminates after first step, assigns fixed reward.
    """

    name = "toy"

    def reset(self, seed: int | None = None) -> ToyState:
        if seed is not None:
            random.seed(seed)
        return ToyState(step=0, done=False)

    def observe(self, state: ToyState) -> list[Message]:
        return [
            Message(role="system", content="You are a test agent."),
            Message(role="user", content="Please provide a JSON answer."),
        ]

    def step(self, state: ToyState, action: Any) -> StepResult:
        return StepResult(
            next_state=ToyState(step=state.step + 1, done=True),
            done=True,
            artifacts={"completed": True},
        )


class ToyJudge:
    """Minimal judge for testing trainer.

    Returns fixed positive reward.
    """

    def score(
        self, *, task_name: str, trajectory: Trajectory, artifacts: dict
    ) -> Reward:
        return Reward(
            total=1.0,
            components={"base": 1.0},
            info={"task": task_name},
        )


class Trainer:
    """Training orchestrator for GFlowNet-style RL.

    Coordinates:
    - Rollout collection with the policy
    - SubTB loss computation
    - Replay buffer management
    - Gradient updates

    Does NOT manage:
    - Model loading (caller provides policy)
    - WandB logging (caller provides callback)
    - Checkpointing (caller handles externally)
    """

    def __init__(
        self,
        config: TrainerConfig,
        policy: Any,  # implements Policy protocol
        task: Any,  # implements Task protocol
        codec: ActionCodec,
        executor_registry: dict[str, Any] | None = None,
        judge: Any | None = None,  # implements Judge protocol
        device: str = "cpu",
        log_callback: Callable[[TrainMetrics, int], None] | None = None,
    ):
        self.config = config
        self.policy = policy
        self.task = task
        self.codec = codec
        self.executors = executor_registry or {}
        self.judge = judge if judge is not None else ToyJudge()
        self.device = torch.device(device)
        self.log_callback = log_callback

        # set seed
        random.seed(config.seed)
        torch.manual_seed(config.seed)

        # learned log partition function
        self._logZ = nn.Parameter(torch.tensor(0.0, device=self.device))

        # replay buffer
        self.replay_buffer = ReplayBuffer(capacity=config.replay_buffer_capacity)

        # optimizer: separate param groups for policy and logZ
        # logZ always trained; policy params added if available
        param_groups = [
            {"params": [self._logZ], "lr": config.logZ_learning_rate},
        ]

        # add policy parameters if policy is trainable (has parameters method)
        if hasattr(self.policy, "parameters") and callable(self.policy.parameters):
            try:
                policy_params = list(self.policy.parameters())
                if policy_params:
                    param_groups.append(
                        {"params": policy_params, "lr": config.learning_rate}
                    )
            except Exception:
                # policy.parameters() may fail for some policy types
                pass

        self.optimizer = torch.optim.Adam(param_groups)

        # step counter
        self._step = 0
        self._accumulated_loss = 0.0
        self._accumulation_count = 0

    @property
    def logZ(self) -> float:
        """Current value of learned log partition function."""
        return self._logZ.item()

    def _collect_episodes(self, num_episodes: int) -> list[Trajectory]:
        """Collect new episodes using rollout."""
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
            # add to replay buffer
            self.replay_buffer.add(traj)

        return trajectories

    def _recompute_logprobs_differentiable(
        self, traj: Trajectory
    ) -> list[torch.Tensor]:
        """Recompute logprobs with gradient tracking for training.

        For single-generation trajectories, uses all non-assistant messages as context.
        For multi-generation, approximates by using messages up to each generation point.

        Returns:
            List of tensors, one per generation, each with gradient tracking.
        """
        result = []

        # find assistant message indices to reconstruct context
        assistant_indices = [
            i for i, m in enumerate(traj.messages) if m.role == "assistant"
        ]

        for gen_idx, token_ids in enumerate(traj.token_ids):
            if not token_ids:
                result.append(torch.tensor([], device=self.device))
                continue

            # reconstruct context: messages before this generation
            if gen_idx < len(assistant_indices):
                context_end = assistant_indices[gen_idx]
            else:
                # fallback: use all non-assistant messages
                context_end = len(traj.messages)

            context_messages = [
                m for m in traj.messages[:context_end] if m.role != "assistant"
            ]

            # if no context, use first system/user messages
            if not context_messages:
                context_messages = [
                    m for m in traj.messages if m.role in ("system", "user")
                ][:2]

            # call policy's differentiable scoring
            logprobs_tensor = self.policy.score_tokens(context_messages, token_ids)
            result.append(logprobs_tensor)

        return result

    def _trajectories_to_tensors(
        self, trajectories: list[Trajectory]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert trajectories to padded tensors for loss computation.

        If policy supports score_tokens(), recomputes logprobs with gradient tracking.
        Otherwise falls back to stored floats (no gradient flow to policy).

        Returns:
            log_probs: [batch, max_seq_len] padded with 0
            loss_mask: [batch, max_seq_len] boolean mask
            log_rewards: [batch] log of rewards
        """
        batch_size = len(trajectories)
        use_differentiable = hasattr(self.policy, "score_tokens") and callable(
            getattr(self.policy, "score_tokens", None)
        )

        all_logprobs = []  # list of tensors or lists
        all_masks = []
        log_rewards = []

        for traj in trajectories:
            if use_differentiable:
                # recompute with gradient tracking
                gen_logprobs_tensors = self._recompute_logprobs_differentiable(traj)
                # concatenate all generations into one tensor
                if gen_logprobs_tensors:
                    flat_logprobs = torch.cat(gen_logprobs_tensors)
                else:
                    flat_logprobs = torch.tensor([], device=self.device)
            else:
                # fallback: use stored floats (no gradient to policy)
                flat_logprobs_list = []
                for gen_logprobs in traj.token_logprobs:
                    flat_logprobs_list.extend(gen_logprobs)
                flat_logprobs = torch.tensor(flat_logprobs_list, device=self.device)

            # flatten masks
            flat_mask = []
            for gen_mask in traj.loss_mask:
                flat_mask.extend(gen_mask)
            all_logprobs.append(flat_logprobs)
            all_masks.append(flat_mask)

            # compute log reward (handle zero/negative rewards)
            reward = max(traj.reward.total, 1e-10)  # clamp to avoid log(0)
            log_rewards.append(math.log(reward))

        # find max length
        max_len = max(len(lp) for lp in all_logprobs) if all_logprobs else 1

        # pad and stack - preserve gradient if tensors have it
        log_probs_tensor = torch.zeros(batch_size, max_len, device=self.device)
        mask_tensor = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self.device)

        for i, (lps, masks) in enumerate(zip(all_logprobs, all_masks, strict=False)):
            seq_len = len(lps)
            if seq_len > 0:
                # use slice assignment to preserve gradients
                log_probs_tensor[i, :seq_len] = lps
                mask_tensor[i, :seq_len] = torch.tensor(masks, device=self.device)

        log_rewards_tensor = torch.tensor(log_rewards, device=self.device)

        return log_probs_tensor, mask_tensor, log_rewards_tensor

    def train_step(self) -> TrainMetrics:
        """Run a single training step.

        1. Collect `batch_size` new episodes using rollout
        2. Optionally sample from replay buffer
        3. Compute SubTB loss
        4. Update policy and logZ

        Returns:
            Metrics from this training step
        """
        # determine how many fresh vs replay episodes
        num_fresh = self.config.batch_size
        num_replay = 0
        replay_fraction = 0.0

        if len(self.replay_buffer) >= self.config.batch_size:
            num_replay = int(self.config.batch_size * self.config.replay_sample_ratio)
            num_fresh = self.config.batch_size - num_replay
            replay_fraction = num_replay / self.config.batch_size

        # collect fresh episodes
        fresh_trajectories = self._collect_episodes(num_fresh)

        # sample from replay
        replay_trajectories = []
        if num_replay > 0:
            replay_trajectories = self.replay_buffer.sample(num_replay)

        # combine trajectories
        all_trajectories = fresh_trajectories + replay_trajectories

        # convert to tensors
        log_probs, loss_mask, log_rewards = self._trajectories_to_tensors(all_trajectories)

        # compute loss
        loss = subtb_loss(
            log_probs,
            loss_mask,
            log_rewards,
            self._logZ,
            normalize_by_length=self.config.normalize_by_length,
        )

        # accumulate gradients
        self._accumulated_loss += loss.item()
        self._accumulation_count += 1

        # backprop
        scaled_loss = loss / self.config.gradient_accumulation_steps
        scaled_loss.backward()

        # update if accumulated enough
        if self._accumulation_count >= self.config.gradient_accumulation_steps:
            # gradient clipping for all trainable parameters
            all_params = [self._logZ]
            if hasattr(self.policy, "parameters") and callable(self.policy.parameters):
                try:
                    all_params.extend(list(self.policy.parameters()))
                except Exception:
                    pass
            torch.nn.utils.clip_grad_norm_(all_params, self.config.max_grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()

            self._accumulation_count = 0

        # compute metrics
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

        # call log callback
        if self.log_callback is not None:
            self.log_callback(metrics, self._step)

        return metrics

    def train(self, num_steps: int | None = None) -> list[TrainMetrics]:
        """Run full training loop.

        Args:
            num_steps: Override config.max_episodes if provided

        Returns:
            List of metrics from each training step
        """
        steps = num_steps if num_steps is not None else self.config.max_episodes
        all_metrics = []

        for _ in range(steps):
            metrics = self.train_step()
            all_metrics.append(metrics)

        return all_metrics

    def evaluate(self, num_episodes: int = 10) -> dict[str, float]:
        """Evaluate current policy without training.

        Returns:
            Dict with avg_reward, success_rate, etc.
        """
        rollout_cfg = RolloutConfig(max_steps=self.config.max_steps_per_episode)
        rewards = []
        successes = 0

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
            # consider success if reward > 0
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
