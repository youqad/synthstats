"""Local PyTorch training runner.

Pure PyTorch training on a single node. Uses LoopRunner for the
training loop and builds all components from Hydra config.
"""

from __future__ import annotations

import logging
import signal
from typing import Any

from omegaconf import DictConfig, OmegaConf

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

            env = self._build_env()
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

    def _build_env(self) -> Any:
        """Build environment from config."""
        from synthstats.envs.builders import build_env

        return build_env(self.cfg)

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
            beta=obj_cfg.get("beta", 1.0),
            loss_type=obj_cfg.get("loss_type", "tb"),
            subtb_lambda=obj_cfg.get("subtb_lambda", 0.9),
            tb_max_residual=obj_cfg.get("tb_max_residual", 100.0),
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
            logZ_lr=optim_cfg.get("logZ_lr", 1e-1),
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

            # check if WandbLogger - inject training config
            target = logging_cfg.get("_target_", "")
            if "WandbLogger" in target:
                return instantiate(
                    logging_cfg,
                    config=OmegaConf.to_container(self.cfg, resolve=True),
                )
            return instantiate(logging_cfg)

        # fallback for legacy configs without _target_
        return StdoutLogger()

    def state_dict(self) -> dict[str, Any]:
        """Not implemented for runner (use checkpoint manager)."""
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Not implemented for runner (use checkpoint manager)."""
        pass
