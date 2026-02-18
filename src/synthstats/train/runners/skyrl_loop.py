"""SkyRL-compatible but standalone GFlowNet SubTB training loop.

Uses custom TrainingLoop rather than SkyRL's BasePPOExp to support
GFN-specific features (on-sample replay re-scoring, EOS logprob tracking)
that don't easily port to Ray.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from synthstats.core.constants import LOGZ_LR_DEFAULT, REWARD_FLOOR_DEFAULT
from synthstats.envs.skyrl_text_env import SynthStatsTextEnv
from synthstats.train.checkpointing.base import (
    CheckpointState,
    cleanup_old_checkpoints,
)
from synthstats.train.checkpointing.base import (
    load_checkpoint as _load_checkpoint,
)
from synthstats.train.checkpointing.base import (
    save_checkpoint as _save_checkpoint,
)
from synthstats.train.data.collate import build_subtb_batch, build_tinker_batch
from synthstats.train.data.collectors import CollectedTrajectory, TrajectoryCollector
from synthstats.train.data.metrics import summarize_eval_metrics
from synthstats.train.data.replay import GFNReplayBuffer, ReplayBuffer
from synthstats.train.runners.skyrl_subtb import SkyRLSubTBTrainer
from synthstats.train.utils.seeding import get_rng_states, set_rng_states

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:

    # episodes per batch
    num_episodes: int = 100
    batch_size: int = 4

    # optimizer
    learning_rate: float = 1e-4
    logZ_lr: float = LOGZ_LR_DEFAULT

    # generation
    temperature: float = 0.7

    # evaluation
    eval_interval: int = 10
    eval_episodes: int = 5

    # SubTB
    reward_floor: float = REWARD_FLOOR_DEFAULT
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

    Call setup() with policy/trainer/env, then run(steps=N).
    """

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()
        self.policy = None
        self.trainer = None
        self.env = None
        self.collector = None
        self.step_count = 0
        self._all_metrics: deque[dict[str, float]] = deque(maxlen=10_000)
        self._log_callback: Callable[[dict, int], None] | None = None

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
    def _ensure_ref_policy_compat(policy: Any, ref_policy: Any, *, allow_mismatch: bool) -> None:
        if allow_mismatch:
            return

        tok = getattr(policy, "tokenizer", None)
        ref_tok = getattr(ref_policy, "tokenizer", None)
        if tok is None or ref_tok is None:
            return

        mismatches = []
        if tok.__class__ is not ref_tok.__class__:
            mismatches.append(f"class {tok.__class__.__name__} != {ref_tok.__class__.__name__}")

        name = getattr(tok, "name_or_path", None)
        ref_name = getattr(ref_tok, "name_or_path", None)
        if name and ref_name and name != ref_name:
            mismatches.append(f"name_or_path {name} != {ref_name}")

        vocab_size = getattr(tok, "vocab_size", None)
        ref_vocab_size = getattr(ref_tok, "vocab_size", None)
        if vocab_size is not None and ref_vocab_size is not None:
            if int(vocab_size) != int(ref_vocab_size):
                mismatches.append(f"vocab_size {vocab_size} != {ref_vocab_size}")

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
            raise ValueError("use_ref_policy=True requires a ref_policy with score_action")
        if use_ref_policy and ref_policy is not None:
            allow_mismatch = getattr(trainer.config, "allow_mismatched_tokenizer", False)
            self._ensure_ref_policy_compat(
                policy,
                ref_policy,
                allow_mismatch=allow_mismatch,
            )

        self.collector = TrajectoryCollector(
            env=env,
            policy_fn=policy,
            score_fn=score_fn,
        )

    def train_step(self) -> dict[str, Any]:
        if self.collector is None or self.trainer is None:
            raise RuntimeError("Call setup() before train_step()")

        use_ref_policy = getattr(self.trainer.config, "use_ref_policy", False)

        has_gfn_buffer = (
            self._gfn_replay_buffer is not None
            and len(self._gfn_replay_buffer) >= self.config.batch_size
        )
        has_simple_buffer = (
            self._replay_buffer is not None and len(self._replay_buffer) >= self.config.batch_size
        )

        if has_gfn_buffer or has_simple_buffer:
            num_replay = int(self.config.batch_size * self.config.replay_ratio)
            num_fresh = self.config.batch_size - num_replay
        else:
            num_fresh = self.config.batch_size
            num_replay = 0

        fresh_trajectories = self.collector.collect(
            episodes=num_fresh,
            temperature=self.config.temperature,
            compute_ref_log_probs=use_ref_policy,
        )

        if self._gfn_replay_buffer is not None:
            for traj in fresh_trajectories:
                log_reward = math.log(max(traj.reward, self.config.reward_floor))
                self._gfn_replay_buffer.add_from_trajectory(traj, log_reward=log_reward)
        elif self._replay_buffer is not None:
            for traj in fresh_trajectories:
                self._replay_buffer.add(traj.detach())

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

        trajectories = fresh_trajectories + replay_trajectories

        # strip eos_logprobs when mixing with replay (replay can't provide them)
        # this falls back to vanilla TB instead of modified_subtb
        if replay_trajectories and any(
            getattr(t, "eos_logprobs", None) is not None for t in fresh_trajectories
        ):
            logger.warning(
                "Mixing replay trajectories with fresh - stripping eos_logprobs, "
                "falling back to vanilla TB loss instead of modified_subtb"
            )
            trajectories = [
                CollectedTrajectory(
                    observations=t.observations,
                    actions=t.actions,
                    log_probs=t.log_probs,
                    ref_log_probs=t.ref_log_probs,
                    entropy=t.entropy,
                    reward=t.reward,
                    temperature=t.temperature,
                    eos_logprobs=None,  # strip for consistency
                    prompts=getattr(t, "prompts", None),
                    completions=getattr(t, "completions", None),
                )
                for t in trajectories
            ]

        is_tinker = "Tinker" in self.trainer.__class__.__name__
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

        metrics = self.trainer.train_step(batch)

        def _reward_value(reward: Any) -> float:
            return float(getattr(reward, "total", reward))

        avg_reward = sum(_reward_value(t.reward) for t in trajectories) / len(trajectories)
        metrics["avg_reward"] = avg_reward
        metrics["num_episodes"] = len(trajectories)
        metrics["replay_ratio"] = num_replay / len(trajectories) if trajectories else 0.0

        if self._gfn_replay_buffer is not None:
            self._gfn_replay_buffer.increment_policy_version()
            staleness = self._gfn_replay_buffer.get_staleness_stats()
            metrics["buffer_mean_staleness"] = staleness["mean_staleness"]
            metrics["buffer_max_staleness"] = staleness["max_staleness"]
            metrics["buffer_size"] = len(self._gfn_replay_buffer)

        self.step_count += 1

        if self._log_callback is not None:
            self._log_callback(metrics, self.step_count)

        return metrics

    def run(self, steps: int) -> list[dict[str, float]]:
        all_metrics = []

        for _ in range(steps):
            metrics = self.train_step()
            all_metrics.append(metrics)
            self._all_metrics.append(metrics)

        return all_metrics

    def evaluate(self, episodes: int | None = None) -> dict[str, float]:
        if self.collector is None:
            raise RuntimeError("Call setup() before evaluate()")

        import torch

        n = episodes or self.config.eval_episodes

        # avoid temperature=0.0 to prevent sampler errors in HFPolicy
        eval_temp = self.config.temperature
        if eval_temp <= 0:
            eval_temp = 1e-3

        with torch.no_grad():
            trajectories = self.collector.collect(
                episodes=n,
                temperature=eval_temp,
            )

        rewards = [t.reward for t in trajectories]
        return summarize_eval_metrics(
            rewards,
            episodes=n,
            logZ=float(self.trainer.logZ.item()) if self.trainer else 0.0,
        )

    @property
    def metrics_history(self) -> list[dict[str, float]]:
        return list(self._all_metrics)

    def save_checkpoint(
        self,
        path: str | Path,
        keep_last_n: int | None = None,
    ) -> Path:
        if self.trainer is None:
            raise RuntimeError("Call setup() before save_checkpoint()")

        path = Path(path)

        model_state_dict = None
        if self.policy is not None and hasattr(self.policy, "model"):
            model = self.policy.model
            if hasattr(model, "state_dict"):
                model_state_dict = model.state_dict()

        optimizer_state_dict = None
        if self.trainer.optimizer is not None:
            optimizer_state_dict = self.trainer.optimizer.state_dict()

        replay_buffer_state = None
        if self._gfn_replay_buffer is not None:
            replay_buffer_state = self._gfn_replay_buffer.state_dict()
        elif self._replay_buffer is not None:
            replay_buffer_state = self._replay_buffer.state_dict()

        state = CheckpointState(
            step_count=self.step_count,
            logZ=self.trainer.logZ.item(),
            model_state_dict=model_state_dict,
            optimizer_state_dict=optimizer_state_dict,
            rng_states=get_rng_states(),
            replay_buffer=replay_buffer_state,
            config=asdict(self.config),
            metrics_history=list(self._all_metrics),
        )

        _save_checkpoint(path, state)

        if keep_last_n is not None and keep_last_n > 0:
            cleanup_old_checkpoints(path.parent, keep_last_n)

        return path

    def load_checkpoint(self, path: str | Path) -> None:
        if self.trainer is None:
            raise RuntimeError("Call setup() before load_checkpoint()")

        state = _load_checkpoint(Path(path))
        self._restore_from_state(state)

    def _restore_from_state(self, state: CheckpointState) -> None:
        self.step_count = state.step_count
        self._all_metrics = deque(state.metrics_history, maxlen=10_000)

        if self.trainer is not None:
            self.trainer.load_state_dict({"logZ": state.logZ, "config": state.config})

            if state.optimizer_state_dict is not None and self.trainer.optimizer is not None:
                self.trainer.optimizer.load_state_dict(state.optimizer_state_dict)

        set_rng_states(state.rng_states)

        if state.replay_buffer is not None:
            if self._gfn_replay_buffer is not None:
                self._gfn_replay_buffer.load_state_dict(state.replay_buffer)
            elif self._replay_buffer is not None:
                self._replay_buffer.load_state_dict(state.replay_buffer)

        if state.model_state_dict is not None and self.policy is not None:
            if hasattr(self.policy, "model") and hasattr(self.policy.model, "load_state_dict"):
                self.policy.model.load_state_dict(state.model_state_dict)

        logger.info(f"Restored training state from step {state.step_count}")

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        policy: Any,
        trainer: SkyRLSubTBTrainer,
        env: SynthStatsTextEnv,
        ref_policy: Any | None = None,
        log_callback: Callable[[dict, int], None] | None = None,
    ) -> TrainingLoop:
        state = _load_checkpoint(Path(path))
        cfg = dict(state.config)
        # migrate old field name from pre-WU19 checkpoints
        if "logZ_learning_rate" in cfg and "logZ_lr" not in cfg:
            cfg["logZ_lr"] = cfg.pop("logZ_learning_rate")
        config = TrainingConfig(**cfg)
        loop = cls(config=config)

        loop.setup(
            policy=policy,
            trainer=trainer,
            env=env,
            ref_policy=ref_policy,
            log_callback=log_callback,
        )

        loop._restore_from_state(state)

        return loop
