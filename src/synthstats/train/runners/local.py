"""Local PyTorch training runner.

Pure PyTorch training on a single node. Uses LoopRunner for the
training loop and builds all components from Hydra config.
"""

from __future__ import annotations

import logging
import signal
from typing import Any

from omegaconf import DictConfig, OmegaConf

from synthstats.core.constants import (
    LOGZ_LR_DEFAULT,
    SUBTB_LAMBDA_DEFAULT,
    TB_MAX_RESIDUAL_DEFAULT,
)
from synthstats.envs.builders import build_env
from synthstats.train.checkpointing.torch_full import FullStateCheckpoint
from synthstats.train.learners.subtb_torch import SubTBTorchConfig, SubTBTorchLearner
from synthstats.train.logging.stdout import StdoutLogger
from synthstats.train.loop.batching import build_subtb_batch
from synthstats.train.loop.collectors import TrajectoryCollector
from synthstats.train.loop.loop_runner import LoopConfig, LoopRunner
from synthstats.train.objectives.subtb import SubTBConfig, SubTBObjective
from synthstats.train.runners.base import RunResult
from synthstats.train.utils.device import resolve_device
from synthstats.train.utils.seeding import seed_everything

logger = logging.getLogger(__name__)

# global shutdown flag
_shutdown_requested = False


def _handle_signal(signum: int, frame: Any) -> None:
    """Handle shutdown signals."""
    global _shutdown_requested
    logger.info(f"Received signal {signum}, requesting shutdown...")
    _shutdown_requested = True


class LocalRunner:
    """Local PyTorch training runner.

    Builds all components from config and runs training via LoopRunner.

    Args:
        cfg: Hydra DictConfig with full training configuration
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self._checkpoints: list[str] = []

        # register signal handlers
        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)

    def run(self) -> RunResult:
        """Execute training.

        Returns:
            RunResult with final metrics and checkpoint paths
        """
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
            learner = self._build_learner(objective, policy, device)
            collector = TrajectoryCollector(env=env, policy_fn=policy)
            checkpoint_mgr = self._build_checkpoint_manager()
            logger_sink = self._build_logger()

            runner_cfg = self.cfg.get("runner", {})
            train_cfg = runner_cfg.get("train", {})
            replay_cfg = runner_cfg.get("replay", {})

            loop_config = LoopConfig(
                num_steps=train_cfg.get("steps", 1000),
                batch_size=train_cfg.get("batch_size", 4),
                temperature=train_cfg.get("temperature", 0.7),
                eval_interval=train_cfg.get("eval_every", 100),
                eval_episodes=train_cfg.get("eval_episodes", 8),
                replay_buffer_size=replay_cfg.get("capacity", 0),
                replay_ratio=replay_cfg.get("ratio", 0.5),
                replay_prioritized=replay_cfg.get("prioritized", True),
                replay_alpha=replay_cfg.get("alpha", 1.0),
                use_gfn_replay=replay_cfg.get("mode", "gfn_rescore") == "gfn_rescore",
                device=device,
            )

            loop = LoopRunner(
                collector=collector,
                learner=learner,
                logger_sink=logger_sink,
                checkpoint_manager=checkpoint_mgr,
                config=loop_config,
                batch_builder=build_subtb_batch,
            )

            # SFT warm-start: pre-populate replay buffer with expert demonstrations
            warmstart_cfg = self.cfg.get("sft_warmstart", {})
            if warmstart_cfg.get("enabled", False):
                self._warmstart_from_sft(loop, warmstart_cfg)

            if checkpoint_mgr is not None:
                start_step = checkpoint_mgr.restore(
                    learner=learner,
                    policy=policy,
                )
                if start_step > 0:
                    loop.step_count = start_step
                    logger.info(f"Resumed from step {start_step}")

            logger.info(f"Starting training for {loop_config.num_steps} steps")
            all_metrics = []

            while loop.step_count < loop_config.num_steps:
                if _shutdown_requested:
                    logger.info("Shutdown requested")
                    break

                metrics = loop.train_step()
                all_metrics.append(metrics)

                if checkpoint_mgr is not None:
                    path = checkpoint_mgr.maybe_save(
                        step=loop.step_count,
                        learner=learner,
                        policy=policy,
                    )
                    if path is not None:
                        self._checkpoints.append(str(path))

            if _shutdown_requested and checkpoint_mgr is not None:
                path = checkpoint_mgr.save(
                    step=loop.step_count,
                    learner=learner,
                    policy=policy,
                    metrics_history=all_metrics,
                )
                self._checkpoints.append(str(path))

            if logger_sink is not None:
                logger_sink.close()

            final_metrics = all_metrics[-1] if all_metrics else {}
            return RunResult(
                metrics=final_metrics,
                checkpoints=self._checkpoints,
                interrupted=_shutdown_requested,
            )

        except Exception as e:
            logger.exception("Training failed")
            return RunResult(error=str(e))

    def _build_policy(self, device: str) -> Any:
        """Build policy from config."""
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
        """Build objective from config."""
        obj_cfg = self.cfg.get("objective", {})
        ref_cfg = obj_cfg.get("ref_policy", {})
        config = SubTBConfig(
            loss_type=obj_cfg.get("loss_type", "tb"),
            subtb_lambda=obj_cfg.get("subtb_lambda", SUBTB_LAMBDA_DEFAULT),
            tb_max_residual=obj_cfg.get("tb_max_residual", TB_MAX_RESIDUAL_DEFAULT),
            logZ_init=obj_cfg.get("logZ_init", 0.0),
            use_ref_policy=ref_cfg.get("enabled", False),
            ref_weight=ref_cfg.get("weight", 1.0),
            normalize_by_length=ref_cfg.get("normalize_by_length", False),
            entropy_coef=obj_cfg.get("entropy_coef", 0.01),
        )
        return SubTBObjective(config=config, device=device)

    def _build_learner(
        self,
        objective: SubTBObjective,
        policy: Any,
        device: str,
    ) -> SubTBTorchLearner:
        """Build learner from config."""
        learner_cfg = self.cfg.get("learner", {})
        optim_cfg = learner_cfg.get("optim", {})

        config = SubTBTorchConfig(
            policy_lr=optim_cfg.get("policy_lr", 1e-5),
            logZ_lr=optim_cfg.get("logZ_lr", LOGZ_LR_DEFAULT),
            weight_decay=optim_cfg.get("weight_decay", 0.0),
            max_grad_norm=optim_cfg.get("max_grad_norm", 1.0),
            amp_enabled=learner_cfg.get("precision", {}).get("amp", False),
        )
        return SubTBTorchLearner(
            objective=objective,
            policy=policy,
            config=config,
            device=device,
        )

    def _build_checkpoint_manager(self) -> FullStateCheckpoint | None:
        """Build checkpoint manager from config."""
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
        """Build logger from config."""
        logging_cfg = self.cfg.get("logging", {})

        if "_target_" in logging_cfg:
            from hydra.utils import instantiate

            # check if WandbLogger - inject training config if not already present
            target = logging_cfg.get("_target_", "")
            if "WandbLogger" in target and "config" not in logging_cfg:
                return instantiate(  # type: ignore[misc]
                    logging_cfg,
                    config=OmegaConf.to_container(self.cfg, resolve=True),
                )
            return instantiate(logging_cfg)

        # fallback for legacy configs without _target_
        return StdoutLogger()

    def _warmstart_from_sft(
        self,
        loop: LoopRunner,
        warmstart_cfg: DictConfig | dict[str, Any],
    ) -> None:
        """Pre-populate replay buffer with SFT demonstrations.

        Args:
            loop: The LoopRunner instance with replay buffer
            warmstart_cfg: Configuration dict with:
                - data_path: Path to JSONL file
                - strip_thinking: Remove <think> content (default False)
                - dedupe: Deduplicate entries (default True)
                - compute_rewards: Compute real ELPD-based rewards (default False for speed)
                - log_reward_default: Default log reward if not computing (default -5.0)
        """
        from pathlib import Path

        data_path = warmstart_cfg.get("data_path")
        if not data_path:
            logger.warning("sft_warmstart.enabled=True but no data_path specified")
            return

        buffer = loop.gfn_replay_buffer
        if buffer is None:
            logger.warning("SFT warmstart requires GFN replay buffer (use_gfn_replay=True)")
            return

        from synthstats.data.sft_loader import (
            compute_sft_rewards,
            load_sft_jsonl,
            sft_to_buffer_entry,
        )

        path = Path(data_path).expanduser()
        # Default must preserve <think> to maintain the CoT-as-latent-variable
        # structure required by TB/SubTB training.
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

        # compute rewards if requested (slow but correct)
        if compute_rewards:
            collector = getattr(loop, "collector", None)
            env = getattr(collector, "env", None)
            reward_fn = getattr(env, "score_program", None)
            if not callable(reward_fn):
                raise ValueError(
                    "sft_warmstart.compute_rewards=True requires env.score_program(program: str) "
                    "-> float reward (supported by BoxingEnv)"
                )

            # Default to the reward pipeline clamp range (-700, 700) to avoid overflow/underflow.
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
            log_rewards = compute_sft_rewards(
                examples,
                reward_fn,
                log_clamp=log_clamp,
                show_progress=show_progress,
            )
        else:
            log_rewards = [log_reward_default] * len(examples)

        # convert to BufferEntry
        entries = []
        for ex, log_r in zip(examples, log_rewards, strict=False):
            entry = sft_to_buffer_entry(
                ex,
                policy_version=0,
                log_reward=log_r,
                strip_thinking=strip_thinking,
            )
            entries.append(entry)

        # pre-populate buffer
        added = buffer.pre_populate(entries, dedupe=dedupe)
        logger.info(f"Warm-started replay buffer with {added} SFT demonstrations")

    def state_dict(self) -> dict[str, Any]:
        """Not implemented for runner (use checkpoint manager)."""
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Not implemented for runner (use checkpoint manager)."""
        pass
