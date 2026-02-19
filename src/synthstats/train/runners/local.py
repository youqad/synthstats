"""Local PyTorch training runner."""

from __future__ import annotations

import logging
import math
import signal
from dataclasses import replace
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from synthstats.core.constants import (
    LOGZ_LR_DEFAULT,
    REWARD_FLOOR_DEFAULT,
    SUBTB_LAMBDA_DEFAULT,
    TB_MAX_RESIDUAL_DEFAULT,
)
from synthstats.envs.builders import build_env
from synthstats.train.data.collate import build_subtb_batch, extract_reward
from synthstats.train.data.collectors import TrajectoryCollector
from synthstats.train.data.replay import GFNReplayBuffer, ReplayBuffer
from synthstats.train.objectives.subtb import SubTBConfig, SubTBObjective
from synthstats.train.runners.base import RunResult
from synthstats.train.utils.device import resolve_device
from synthstats.train.utils.seeding import seed_everything

logger = logging.getLogger(__name__)

# reset at the start of each run() call; not re-entrant across threads
_shutdown_requested = False


def _handle_signal(signum: int, frame: Any) -> None:
    global _shutdown_requested
    logger.info(f"Received signal {signum}, requesting shutdown...")
    _shutdown_requested = True


class LocalRunner:
    """Local PyTorch training runner."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self._checkpoints: list[str] = []
        self._eos_downgrade_warned = False
        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)

    def run(self) -> RunResult:
        global _shutdown_requested
        _shutdown_requested = False

        try:
            seed = self.cfg.get("seed", 42)
            seed_everything(seed)
            device = resolve_device(self.cfg.get("device", "auto"))
            logger.info(f"Using device: {device}")

            env = build_env(self.cfg)
            policy = self._build_policy(device)
            objective = self._build_objective(device)

            param_groups: list[dict[str, Any]] = [
                {"params": [objective.logZ], "lr": self._optim_cfg.get("logZ_lr", LOGZ_LR_DEFAULT), "weight_decay": 0.0},
            ]
            if hasattr(policy, "parameters") and callable(policy.parameters):
                policy_params = [p for p in policy.parameters() if p.requires_grad]
                if policy_params:
                    param_groups.append({
                        "params": policy_params,
                        "lr": self._optim_cfg.get("policy_lr", 1e-5),
                        "weight_decay": self._optim_cfg.get("weight_decay", 0.0),
                    })
            if objective.boundary_critic is not None:
                param_groups.append({
                    "params": list(objective.boundary_critic.parameters()),
                    "lr": self._optim_cfg.get("policy_lr", 1e-5),
                    "weight_decay": self._optim_cfg.get("weight_decay", 0.0),
                })
            optimizer = torch.optim.AdamW(param_groups)
            max_grad_norm = self._optim_cfg.get("max_grad_norm", 1.0)

            collector = TrajectoryCollector(env=env, policy_fn=policy)
            buffer = self._build_replay_buffer()
            checkpoint_mgr = self._build_checkpoint_manager()
            logger_sink = self._build_logger()

            train_cfg = self.cfg.get("runner", {}).get("train", {})
            num_steps = train_cfg.get("steps", 1000)
            batch_size = train_cfg.get("batch_size", 4)
            temperature = train_cfg.get("temperature", 0.7)
            reward_floor = train_cfg.get("reward_floor", REWARD_FLOOR_DEFAULT)
            replay_ratio = self.cfg.get("runner", {}).get("replay", {}).get("ratio", 0.5)
            log_interval = train_cfg.get("log_interval", 10)

            step = 0
            if checkpoint_mgr is not None:
                start_step = checkpoint_mgr.restore(
                    learner=_LearnerShim(objective, optimizer),
                    policy=policy,
                    replay_buffer=buffer,
                )
                if start_step > 0:
                    step = start_step
                    logger.info(f"Resumed from step {start_step}")

            # skip warmstart when resuming — buffer state comes from checkpoint
            if step == 0:
                warmstart_cfg = self.cfg.get("sft_warmstart", {})
                if warmstart_cfg.get("enabled", False) and buffer is not None:
                    if isinstance(buffer, GFNReplayBuffer):
                        self._warmstart_from_sft(buffer, collector, warmstart_cfg)
                    else:
                        logger.warning(
                            "sft_warmstart.enabled=True requires replay.mode=gfn; "
                            "skipping warmstart for replay.mode=simple"
                        )

            logger.info(f"Starting training for {num_steps} steps")
            all_metrics: list[dict[str, float]] = []

            while step < num_steps:
                if _shutdown_requested:
                    logger.info("Shutdown requested")
                    break

                use_replay = buffer is not None and len(buffer) >= batch_size
                if use_replay:
                    num_replay = int(batch_size * replay_ratio)
                    num_fresh = batch_size - num_replay
                else:
                    num_fresh = batch_size
                    num_replay = 0

                fresh = collector.collect(episodes=num_fresh, temperature=temperature)

                if buffer is not None:
                    if isinstance(buffer, GFNReplayBuffer):
                        for traj in fresh:
                            log_r = math.log(max(extract_reward(traj), reward_floor))
                            buffer.add_from_trajectory(traj, log_reward=log_r)
                    else:
                        for traj in fresh:
                            if hasattr(traj, "detach"):
                                buffer.add(traj.detach())
                            else:
                                buffer.add(traj)

                replay = []
                if num_replay > 0 and buffer is not None:
                    if isinstance(buffer, GFNReplayBuffer):
                        replay = buffer.sample(
                            batch_size=num_replay,
                            collector=collector,
                            temperature=temperature,
                        )
                    else:
                        replay = buffer.sample(num_replay)

                trajectories = fresh + replay

                eos_downgraded = False
                if replay and fresh:
                    has_eos = [t.eos_logprobs is not None for t in trajectories]
                    if any(has_eos) and not all(has_eos):
                        if not self._eos_downgrade_warned:
                            logger.warning("stripping eos_logprobs: replay lacks EOS, falling back to vanilla TB")
                            self._eos_downgrade_warned = True
                        trajectories = [replace(t, eos_logprobs=None) for t in trajectories]
                        eos_downgraded = True

                batch = build_subtb_batch(trajectories, reward_floor=reward_floor, device=device)
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = objective(batch)
                loss.backward()

                if max_grad_norm is not None:
                    params = [p for g in optimizer.param_groups for p in g["params"] if p.grad is not None]
                    if params:
                        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                optimizer.step()

                if buffer is not None:
                    metrics["buffer_size"] = len(buffer)
                    if isinstance(buffer, GFNReplayBuffer):
                        buffer.increment_policy_version()
                        staleness = buffer.get_staleness_stats()
                        metrics["buffer_mean_staleness"] = staleness["mean_staleness"]
                        metrics["buffer_max_staleness"] = staleness["max_staleness"]

                avg_reward = sum(extract_reward(t) for t in trajectories) / len(trajectories)
                metrics["avg_reward"] = avg_reward
                metrics["num_episodes"] = len(trajectories)
                metrics["replay_ratio"] = len(replay) / len(trajectories) if trajectories else 0.0
                metrics["eos_downgraded"] = 1.0 if eos_downgraded else 0.0

                step += 1
                all_metrics.append(metrics)

                if logger_sink is not None and step % log_interval == 0:
                    logger_sink.log(step, metrics)

                if checkpoint_mgr is not None:
                    path = checkpoint_mgr.maybe_save(
                        step=step,
                        learner=_LearnerShim(objective, optimizer),
                        policy=policy,
                        replay_buffer=buffer,
                    )
                    if path is not None:
                        self._checkpoints.append(str(path))

            if _shutdown_requested and checkpoint_mgr is not None:
                path = checkpoint_mgr.save(
                    step=step,
                    learner=_LearnerShim(objective, optimizer),
                    policy=policy,
                    replay_buffer=buffer,
                    metrics_history=all_metrics,
                )
                self._checkpoints.append(str(path))

            if logger_sink is not None:
                logger_sink.close()

            final_metrics = all_metrics[-1] if all_metrics else {}
            return RunResult(metrics=final_metrics, checkpoints=self._checkpoints, interrupted=_shutdown_requested)

        except Exception as e:
            logger.exception("Training failed")
            return RunResult(error=str(e))

    # -- component builders --

    @property
    def _optim_cfg(self) -> dict[str, Any]:
        return self.cfg.get("learner", {}).get("optim", {})

    def _build_policy(self, device: str) -> Any:
        policy_cfg = self.cfg.get("policy", {})
        if "_target_" in policy_cfg:
            from hydra.utils import instantiate
            return instantiate(policy_cfg)

        model_name = policy_cfg.get("model_name", "mock")
        if model_name == "mock":
            from synthstats.policies.hf_policy import MockHFPolicy
            return MockHFPolicy(device=device)

        from synthstats.policies.hf_policy import HFPolicy
        return HFPolicy(
            model_name=model_name,
            device=device,
            max_new_tokens=policy_cfg.get("generation", {}).get("max_new_tokens", 300),
            temperature=policy_cfg.get("generation", {}).get("temperature", 0.7),
            dtype=policy_cfg.get("dtype", "bfloat16"),
        )

    def _build_objective(self, device: str) -> SubTBObjective:
        obj_cfg = self.cfg.get("objective", {})
        ref_cfg = obj_cfg.get("ref_policy", {})
        if ref_cfg.get("enabled", False):
            raise ValueError(
                "objective.ref_policy.enabled=True is not supported by LocalRunner. "
                "Use SkyRL trainer for reference policy regularization."
            )
        config = SubTBConfig(
            loss_type=obj_cfg.get("loss_type", "tb"),
            subtb_lambda=obj_cfg.get("subtb_lambda", SUBTB_LAMBDA_DEFAULT),
            tb_max_residual=obj_cfg.get("tb_max_residual", TB_MAX_RESIDUAL_DEFAULT),
            logZ_init=obj_cfg.get("logZ_init", 0.0),
            use_ref_policy=ref_cfg.get("enabled", False),
            ref_weight=ref_cfg.get("weight", 1.0),
            normalize_by_length=ref_cfg.get("normalize_by_length", False),
            entropy_coef=obj_cfg.get("entropy_coef", 0.01),
            ab_subtb_alpha=obj_cfg.get("ab_subtb_alpha", 0.1),
            use_boundary_critic=obj_cfg.get("use_boundary_critic", False),
            boundary_critic_hidden_dim=obj_cfg.get("boundary_critic_hidden_dim", 32),
            boundary_critic_loss_coef=obj_cfg.get("boundary_critic_loss_coef", 1.0),
        )
        return SubTBObjective(config=config, device=device)

    def _build_replay_buffer(self) -> GFNReplayBuffer | ReplayBuffer | None:
        replay_cfg = self.cfg.get("runner", {}).get("replay", {})
        capacity = replay_cfg.get("capacity", 0)
        if capacity <= 0:
            return None
        mode = str(replay_cfg.get("mode", "gfn")).lower()
        if mode == "gfn":
            return GFNReplayBuffer(
                capacity=capacity,
                prioritized=replay_cfg.get("prioritized", True),
                alpha=replay_cfg.get("alpha", 1.0),
            )
        if mode == "simple":
            return ReplayBuffer(
                capacity=capacity,
                prioritized=replay_cfg.get("prioritized", True),
                alpha=replay_cfg.get("alpha", 1.0),
            )
        raise ValueError(f"Unsupported runner.replay.mode={mode!r}; expected 'gfn' or 'simple'")

    def _build_checkpoint_manager(self) -> Any:
        from synthstats.train.checkpointing.torch_full import FullStateCheckpoint

        ckpt_cfg = self.cfg.get("checkpoint", {})
        if ckpt_cfg.get("every_steps", 0) <= 0:
            return None
        mgr = FullStateCheckpoint(
            save_dir=ckpt_cfg.get("save_dir", "checkpoints"),
            every_steps=ckpt_cfg.get("every_steps", 100),
            keep_last=ckpt_cfg.get("keep_last", 3),
            resume_from=ckpt_cfg.get("resume_from"),
        )
        mgr.set_config(OmegaConf.to_container(self.cfg, resolve=True))
        return mgr

    def _build_logger(self) -> Any:
        logging_cfg = self.cfg.get("logging", {})
        if "_target_" in logging_cfg:
            from hydra.utils import instantiate
            target = logging_cfg.get("_target_", "")
            if "WandbLogger" in target and "config" not in logging_cfg:
                return instantiate(logging_cfg, config=OmegaConf.to_container(self.cfg, resolve=True))
            return instantiate(logging_cfg)
        from synthstats.train.logging.stdout import StdoutLogger
        return StdoutLogger()

    def _warmstart_from_sft(
        self,
        buffer: GFNReplayBuffer,
        collector: TrajectoryCollector,
        warmstart_cfg: DictConfig | dict[str, Any],
    ) -> None:
        from pathlib import Path

        data_path = warmstart_cfg.get("data_path")
        if not data_path:
            logger.warning("sft_warmstart.enabled=True but no data_path specified")
            return

        from synthstats.data.sft_loader import (
            compute_sft_rewards,
            load_sft_jsonl,
            sft_to_buffer_entry,
        )

        path = Path(data_path).expanduser()
        strip_thinking = warmstart_cfg.get("strip_thinking", False)
        dedupe = warmstart_cfg.get("dedupe", True)
        compute_rewards = warmstart_cfg.get("compute_rewards", False)
        log_reward_default = warmstart_cfg.get("log_reward_default", -5.0)
        max_examples = warmstart_cfg.get("max_examples")

        if strip_thinking:
            logger.warning(
                "sft_warmstart.strip_thinking=True will drop <think> content. "
                "This breaks the CoT-as-latent-variable setup for TB/SubTB training. "
                "Only enable if you are intentionally changing the training objective."
            )

        logger.info(f"Loading SFT data from {path}")
        examples = load_sft_jsonl(path, max_examples=max_examples)
        if not examples:
            logger.warning("No SFT examples loaded")
            return

        if compute_rewards:
            env = getattr(collector, "env", None)
            reward_fn = getattr(env, "score_program", None)
            if not callable(reward_fn):
                raise ValueError(
                    "sft_warmstart.compute_rewards=True requires env.score_program(program: str)"
                )
            log_clamp = (-700.0, 700.0)
            log_clamp_cfg = warmstart_cfg.get("log_clamp")
            if log_clamp_cfg is not None and not isinstance(log_clamp_cfg, str):
                try:
                    seq = list(log_clamp_cfg)
                except TypeError:
                    seq = []
                if len(seq) == 2 and all(isinstance(x, int | float) for x in seq):
                    log_clamp = (float(seq[0]), float(seq[1]))

            show_progress = warmstart_cfg.get("show_progress", True)
            log_rewards = compute_sft_rewards(examples, reward_fn, log_clamp=log_clamp, show_progress=show_progress)
        else:
            log_rewards = [log_reward_default] * len(examples)

        entries = [
            sft_to_buffer_entry(ex, policy_version=0, log_reward=log_r, strip_thinking=strip_thinking)
            for ex, log_r in zip(examples, log_rewards, strict=False)
        ]
        added = buffer.pre_populate(entries, dedupe=dedupe)
        logger.info(f"Warm-started replay buffer with {added} SFT demonstrations")

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        pass


class _LearnerShim:
    """Adapter so checkpoint manager can call .logZ / .optimizer / .state_dict()."""

    def __init__(self, objective: SubTBObjective, optimizer: torch.optim.Optimizer) -> None:
        self.objective = objective
        self.optimizer = optimizer

    @property
    def logZ(self) -> float:
        return self.objective.logZ.item()

    def state_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.objective.load_state_dict(state["objective"])
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
