"""SkyRL-integrated training loop.

Uses the SkyRL integration components:
- SimpleCollector for trajectory collection
- SkyRLSubTBTrainer for loss computation
- HFPolicy / MockHFPolicy for action generation

This is the canonical training loop for SkyRL-based training.
"""

from __future__ import annotations

import math
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from synthstats.collectors.simple_collector import (
    SimpleCollector,
    build_subtb_batch,
    build_tinker_batch,
)
from synthstats.envs.skyrl_text_env import SynthStatsTextEnv
from synthstats.trainers.skyrl_subtb import SkyRLSubTBTrainer
from synthstats.training.buffers.gfn_replay import GFNReplayBuffer
from synthstats.training.buffers.replay import ReplayBuffer

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for the SkyRL training loop."""

    # episodes per batch
    num_episodes: int = 100
    batch_size: int = 4

    # optimizer
    learning_rate: float = 1e-4
    logZ_learning_rate: float = 1e-2

    # generation
    temperature: float = 0.7

    # evaluation
    eval_interval: int = 10
    eval_episodes: int = 5

    # SubTB
    reward_floor: float = 1e-4
    logZ_init: float = 0.0

    # device
    device: str = "cpu"

    # logging
    log_interval: int = 10

    # replay buffer (off-policy training)
    replay_buffer_size: int = 0  # 0 = disabled
    replay_ratio: float = 0.5  # fraction of batch from replay
    replay_prioritized: bool = False
    replay_alpha: float = 1.0

    # GFlowNet-specific replay with on-sample re-scoring
    # when True: stores action sequences, re-scores with current policy (no stale log_probs)
    # when False: stores full trajectories with original log_probs (simpler but off-policy)
    use_gfn_replay: bool = True


class TrainingLoop:
    """SkyRL-integrated training loop.

    Orchestrates:
    - Trajectory collection via SimpleCollector
    - Batch building with padding/masking
    - SubTB loss computation
    - Gradient updates

    Usage:
        config = TrainingConfig(batch_size=4)
        loop = TrainingLoop(config)
        loop.setup(policy=my_policy, trainer=my_trainer, env=my_env)
        metrics = loop.run(steps=100)
    """

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()
        self.policy = None
        self.trainer = None
        self.env = None
        self.collector = None
        self.step_count = 0
        self._all_metrics: list[dict[str, float]] = []
        self._log_callback: Callable[[dict, int], None] | None = None

        # replay buffer for off-policy training
        self._replay_buffer: ReplayBuffer | None = None
        self._gfn_replay_buffer: GFNReplayBuffer | None = None

        if self.config.replay_buffer_size > 0:
            if self.config.use_gfn_replay:
                self._gfn_replay_buffer = GFNReplayBuffer(
                    capacity=self.config.replay_buffer_size,
                    prioritized=self.config.replay_prioritized,
                    alpha=self.config.replay_alpha,
                )
            else:
                self._replay_buffer = ReplayBuffer(
                    capacity=self.config.replay_buffer_size,
                    prioritized=self.config.replay_prioritized,
                    alpha=self.config.replay_alpha,
                )

    @staticmethod
    def _ensure_ref_policy_compat(
        policy: Any, ref_policy: Any, *, allow_mismatch: bool
    ) -> None:
        """Ensure reference policy tokenization matches behavior policy."""
        if allow_mismatch:
            return

        tok = getattr(policy, "tokenizer", None)
        ref_tok = getattr(ref_policy, "tokenizer", None)
        if tok is None or ref_tok is None:
            return

        mismatches: list[str] = []
        if tok.__class__ is not ref_tok.__class__:
            mismatches.append(
                f"class {tok.__class__.__name__} != {ref_tok.__class__.__name__}"
            )

        name = getattr(tok, "name_or_path", None)
        ref_name = getattr(ref_tok, "name_or_path", None)
        if name and ref_name and name != ref_name:
            mismatches.append(f"name_or_path {name} != {ref_name}")

        vocab_size = getattr(tok, "vocab_size", None)
        ref_vocab_size = getattr(ref_tok, "vocab_size", None)
        if vocab_size is not None and ref_vocab_size is not None:
            if int(vocab_size) != int(ref_vocab_size):
                mismatches.append(
                    f"vocab_size {vocab_size} != {ref_vocab_size}"
                )

        if mismatches:
            details = "; ".join(mismatches)
            raise ValueError(
                "ref_policy tokenizer appears incompatible with policy: "
                f"{details}. Set allow_mismatched_tokenizer=True to override."
            )

    def setup(
        self,
        policy: Any,
        trainer: SkyRLSubTBTrainer,
        env: SynthStatsTextEnv,
        ref_policy: Any | None = None,
        log_callback: Callable[[dict, int], None] | None = None,
    ) -> None:
        """Initialize training components.

        Args:
            policy: Policy for action generation (HFPolicy or MockHFPolicy)
            trainer: SkyRLSubTBTrainer for loss computation
            env: SynthStatsTextEnv environment
            log_callback: Optional callback for logging (e.g., WandB)
        """
        self.policy = policy
        self.trainer = trainer
        self.env = env
        self._log_callback = log_callback
        self.ref_policy = ref_policy

        # GFN replay is incompatible with ref-policy correction because replayed
        # trajectories are re-scored without ref_log_probs. Fall back to the
        # simple replay buffer when ref-policy correction is enabled.
        use_ref_policy = getattr(trainer.config, "use_ref_policy", False)
        if use_ref_policy and self._gfn_replay_buffer is not None:
            logger.warning(
                "GFN replay is incompatible with ref-policy correction; "
                "falling back to the simple replay buffer."
            )
            self._gfn_replay_buffer = None
            if self.config.replay_buffer_size > 0:
                self._replay_buffer = ReplayBuffer(
                    capacity=self.config.replay_buffer_size,
                    prioritized=self.config.replay_prioritized,
                    alpha=self.config.replay_alpha,
                )

        score_fn = None
        if ref_policy is not None:
            if not hasattr(ref_policy, "score_action"):
                raise ValueError("ref_policy must implement score_action")
            score_fn = ref_policy.score_action

        if use_ref_policy and score_fn is None:
            raise ValueError(
                "use_ref_policy=True requires a ref_policy with score_action"
            )
        if use_ref_policy and ref_policy is not None:
            allow_mismatch = getattr(trainer.config, "allow_mismatched_tokenizer", False)
            self._ensure_ref_policy_compat(
                policy,
                ref_policy,
                allow_mismatch=allow_mismatch,
            )

        # create collector
        self.collector = SimpleCollector(
            env=env,
            policy_fn=policy,
            score_fn=score_fn,
        )

    def train_step(self) -> dict[str, float]:
        """Run a single training step.

        1. Collect batch_size trajectories (mix of fresh + replay if enabled)
        2. Build padded batch
        3. Compute SubTB loss
        4. Update parameters

        Returns:
            Metrics dict with loss, logZ, avg_reward, etc.
        """
        if self.collector is None or self.trainer is None:
            raise RuntimeError("Call setup() before train_step()")

        use_ref_policy = getattr(self.trainer.config, "use_ref_policy", False)

        # determine which replay buffer is active
        has_gfn_buffer = (
            self._gfn_replay_buffer is not None
            and len(self._gfn_replay_buffer) >= self.config.batch_size
        )
        has_simple_buffer = (
            self._replay_buffer is not None
            and len(self._replay_buffer) >= self.config.batch_size
        )

        # determine fresh vs replay split
        if has_gfn_buffer or has_simple_buffer:
            num_replay = int(self.config.batch_size * self.config.replay_ratio)
            num_fresh = self.config.batch_size - num_replay
        else:
            num_fresh = self.config.batch_size
            num_replay = 0

        # collect fresh trajectories
        fresh_trajectories = self.collector.collect(
            episodes=num_fresh,
            temperature=self.config.temperature,
            compute_ref_log_probs=use_ref_policy,
        )

        # add fresh trajectories to appropriate buffer
        if self._gfn_replay_buffer is not None:
            for traj in fresh_trajectories:
                log_reward = math.log(max(traj.reward, self.config.reward_floor))
                self._gfn_replay_buffer.add_from_trajectory(traj, log_reward=log_reward)
        elif self._replay_buffer is not None:
            for traj in fresh_trajectories:
                self._replay_buffer.add(traj.detach())

        # sample from replay buffer
        replay_trajectories = []
        if num_replay > 0:
            if has_gfn_buffer:
                replay_trajectories = self._gfn_replay_buffer.sample(
                    batch_size=num_replay,
                    collector=self.collector,
                    temperature=self.config.temperature,
                )
            elif has_simple_buffer:
                replay_trajectories = self._replay_buffer.sample(num_replay)

        # combine trajectories
        trajectories = fresh_trajectories + replay_trajectories

        # build batch - use appropriate builder for trainer type
        is_tinker = hasattr(self.trainer, "__class__") and "Tinker" in self.trainer.__class__.__name__
        if is_tinker:
            batch = build_tinker_batch(
                trajectories,
                reward_floor=self.config.reward_floor,
                device=self.config.device,
            )
        else:
            batch = build_subtb_batch(
                trajectories,
                reward_floor=self.config.reward_floor,
                device=self.config.device,
            )

        # train step
        metrics = self.trainer.train_step(batch)

        # add trajectory-level metrics
        avg_reward = sum(t.reward for t in trajectories) / len(trajectories)
        metrics["avg_reward"] = avg_reward
        metrics["num_episodes"] = len(trajectories)
        metrics["replay_ratio"] = num_replay / len(trajectories) if trajectories else 0.0

        # increment policy version and add staleness stats for GFN buffer
        if self._gfn_replay_buffer is not None:
            self._gfn_replay_buffer.increment_policy_version()
            staleness = self._gfn_replay_buffer.get_staleness_stats()
            metrics["buffer_mean_staleness"] = staleness["mean_staleness"]
            metrics["buffer_max_staleness"] = staleness["max_staleness"]
            metrics["buffer_size"] = len(self._gfn_replay_buffer)

        # update step count
        self.step_count += 1

        # log callback
        if self._log_callback is not None:
            self._log_callback(metrics, self.step_count)

        return metrics

    def run(self, steps: int) -> list[dict[str, float]]:
        """Run training for n steps.

        Args:
            steps: Number of training steps to run

        Returns:
            List of metrics from each step
        """
        all_metrics = []

        for _ in range(steps):
            metrics = self.train_step()
            all_metrics.append(metrics)
            self._all_metrics.append(metrics)

        return all_metrics

    def evaluate(self, episodes: int | None = None) -> dict[str, float]:
        """Evaluate current policy without training.

        Args:
            episodes: Number of episodes to evaluate (defaults to config)

        Returns:
            Evaluation metrics
        """
        if self.collector is None:
            raise RuntimeError("Call setup() before evaluate()")

        n = episodes or self.config.eval_episodes

        # collect with temperature=0 for deterministic eval
        trajectories = self.collector.collect(
            episodes=n,
            temperature=0.0,
        )

        rewards = [t.reward for t in trajectories]
        avg_reward = sum(rewards) / len(rewards)
        max_reward = max(rewards)
        min_reward = min(rewards)

        # success rate: reward > 0
        successes = sum(1 for r in rewards if r > 0)
        success_rate = successes / len(rewards)

        return {
            "eval_avg_reward": avg_reward,
            "eval_max_reward": max_reward,
            "eval_min_reward": min_reward,
            "eval_success_rate": success_rate,
            "eval_episodes": n,
            "logZ": float(self.trainer.logZ.item()) if self.trainer else 0.0,
        }

    @property
    def metrics_history(self) -> list[dict[str, float]]:
        """All metrics collected during training."""
        return self._all_metrics
