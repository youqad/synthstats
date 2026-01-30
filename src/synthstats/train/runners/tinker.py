"""Tinker API training runner.

Uses Tinker's API for model training. Local components (env, collector)
with API-based policy and training.
"""

from __future__ import annotations

import logging
from typing import Any

from omegaconf import DictConfig

from synthstats.train.checkpointing.minimal import MinimalCheckpoint
from synthstats.train.learners.subtb_tinker import SubTBTinkerConfig, SubTBTinkerLearner
from synthstats.train.logging.stdout import StdoutLogger
from synthstats.train.loop.batching import build_tinker_batch
from synthstats.train.loop.collectors import TrajectoryCollector
from synthstats.train.loop.loop_runner import LoopConfig, LoopRunner
from synthstats.train.runners.base import RunResult
from synthstats.train.utils.device import resolve_device
from synthstats.train.utils.seeding import seed_everything

logger = logging.getLogger(__name__)


class TinkerRunner:
    """Tinker API training runner.

    Uses Tinker for policy and training, local env and collection.

    Args:
        cfg: Hydra DictConfig with full configuration
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self._checkpoints: list[str] = []

    def run(self) -> RunResult:
        """Execute training.

        Returns:
            RunResult with final metrics
        """
        try:
            # seed
            seed = self.cfg.get("seed", 42)
            seed_everything(seed)

            device = resolve_device(self.cfg.get("device", "auto"))

            # build components
            env = self._build_env()
            policy = self._build_policy()
            learner = self._build_learner(device)
            collector = TrajectoryCollector(env=env, policy_fn=policy)
            checkpoint_mgr = self._build_checkpoint_manager()
            logger_sink = StdoutLogger()

            # loop config
            runner_cfg = self.cfg.get("runner", {})
            train_cfg = runner_cfg.get("train", {})

            loop_config = LoopConfig(
                num_steps=train_cfg.get("steps", 500),
                batch_size=train_cfg.get("batch_size", 4),
                temperature=train_cfg.get("temperature", 0.7),
                device=device,
            )

            loop = LoopRunner(
                collector=collector,
                learner=learner,
                logger_sink=logger_sink,
                checkpoint_manager=checkpoint_mgr,
                config=loop_config,
                batch_builder=build_tinker_batch,
            )

            # run
            logger.info(f"Starting Tinker training for {loop_config.num_steps} steps")
            all_metrics = loop.run()

            if logger_sink:
                logger_sink.close()

            final_metrics = all_metrics[-1] if all_metrics else {}
            return RunResult(
                metrics=final_metrics,
                checkpoints=self._checkpoints,
            )

        except Exception as e:
            logger.exception("Tinker training failed")
            return RunResult(error=str(e))

    def _build_env(self) -> Any:
        """Build environment."""
        from synthstats.envs.builders import build_env

        return build_env(self.cfg)

    def _build_policy(self) -> Any:
        """Build Tinker policy."""
        from synthstats.integrations.tinker.adapter import TinkerConfig, TinkerPolicy

        policy_cfg = self.cfg.get("policy", {}).get("config", {})
        config = TinkerConfig(
            api_key=policy_cfg.get("api_key"),
            model=policy_cfg.get("model", "Qwen/Qwen3-4B"),
            max_tokens=policy_cfg.get("max_tokens", 256),
            temperature=policy_cfg.get("temperature", 0.7),
        )
        return TinkerPolicy(config=config)

    def _build_learner(self, device: str) -> SubTBTinkerLearner:
        """Build Tinker learner."""
        learner_cfg = self.cfg.get("learner", {})
        tinker_cfg = learner_cfg.get("tinker", {})

        config = SubTBTinkerConfig(
            api_key=tinker_cfg.get("api_key"),
            model=tinker_cfg.get("model", "Qwen/Qwen3-4B"),
            lora_rank=tinker_cfg.get("lora_rank", 32),
            learning_rate=tinker_cfg.get("learning_rate", 1e-5),
        )
        return SubTBTinkerLearner(config=config, device=device)

    def _build_checkpoint_manager(self) -> MinimalCheckpoint | None:
        """Build checkpoint manager."""
        ckpt_cfg = self.cfg.get("checkpoint", {})
        if ckpt_cfg.get("every_steps", 0) <= 0:
            return None

        return MinimalCheckpoint(
            save_dir=ckpt_cfg.get("save_dir", "checkpoints"),
            every_steps=ckpt_cfg.get("every_steps", 0),
        )

    def state_dict(self) -> dict[str, Any]:
        """Not implemented."""
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Not implemented."""
        pass
