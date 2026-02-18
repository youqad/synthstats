#!/usr/bin/env python
"""SynthStats training CLI.

Unified entrypoint that dispatches to runners based on config.

Usage:
    # Local PyTorch training
    synthstats-train runner=local env=dugongs policy=hf_qwen3_0_6b

    # Tinker API backend
    synthstats-train runner=tinker env=dugongs policy=tinker

    # Or via module:
    uv run python -m synthstats.cli.train runner=local
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

load_dotenv(override=True)  # before Hydra resolves ${oc.env:...}

if TYPE_CHECKING:
    from synthstats.train.runners.base import Runner

logger = logging.getLogger(__name__)


def get_runner(cfg: DictConfig) -> Runner:
    """Create runner from config."""
    runner_cfg = cfg.get("runner", {})
    runner_type = runner_cfg.get("type", "local")

    if runner_type == "local":
        from synthstats.train.runners.local import LocalRunner

        return LocalRunner(cfg)

    elif runner_type == "tinker":
        from synthstats.train.runners.tinker import TinkerRunner

        return TinkerRunner(cfg)

    elif runner_type == "runpod":
        from synthstats.train.runners.local import LocalRunner

        return LocalRunner(cfg)

    else:
        raise ValueError(
            f"Unknown runner type: {runner_type}. Valid options: local, tinker, runpod"
        )


@hydra.main(version_base=None, config_path="../../../configs", config_name="train")
def main(cfg: DictConfig) -> float | None:
    """Training entrypoint; returns final loss for HPO or None on error."""
    logger.info("SynthStats Training")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    try:
        runner = get_runner(cfg)
        result = runner.run()

        if result.error:
            logger.error(f"Training failed: {result.error}")
            return None

        if result.interrupted:
            logger.info("Training was interrupted")

        if result.checkpoints:
            logger.info(f"Checkpoints: {result.checkpoints}")

        logger.info(f"Final metrics: {result.metrics}")

        return result.metrics.get("loss")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return None

    except Exception as e:
        logger.exception(f"Training failed with exception: {e}")
        return None


if __name__ == "__main__":
    main()
