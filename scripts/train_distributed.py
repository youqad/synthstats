#!/usr/bin/env python
"""Distributed GFlowNet training with SkyRL.

This script runs distributed SubTB training using Ray and SkyRL infrastructure.
For single-GPU training, use train_skyrl.py instead.

Usage:
    # Single-node, 4 GPUs
    uv run python scripts/train_distributed.py

    # Custom model
    uv run python scripts/train_distributed.py \\
        trainer.policy.model.path="Qwen/Qwen3-4B-Instruct-2507"

    # Multi-node (requires Ray cluster)
    uv run python scripts/train_distributed.py \\
        ray=multi_node \\
        ray.address="ray://<head-ip>:10001"

    # With WandB logging
    uv run python scripts/train_distributed.py \\
        wandb.enabled=true \\
        wandb.project="synthstats-distributed"

Requirements:
    - SkyRL (skyrl-train, skyrl-gym): pip install skyrl-train
    - Ray: pip install ray[default]
    - vLLM (optional, for faster generation): pip install vllm
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def check_skyrl_available() -> bool:
    """Check if SkyRL is installed."""
    try:
        import skyrl_train  # noqa: F401

        return True
    except ImportError:
        return False


def check_ray_available() -> bool:
    """Check if Ray is installed."""
    try:
        import ray  # noqa: F401

        return True
    except ImportError:
        return False


def init_ray(cfg: DictConfig) -> None:
    """Initialize Ray cluster.

    Args:
        cfg: Config with ray settings
    """
    import ray

    ray_cfg = cfg.ray

    # check if already initialized
    if ray.is_initialized():
        logger.info("Ray already initialized")
        return

    # init options
    init_kwargs: dict[str, Any] = {}

    if ray_cfg.address:
        # connect to existing cluster
        init_kwargs["address"] = ray_cfg.address
        logger.info(f"Connecting to Ray cluster at {ray_cfg.address}")
    else:
        # start local cluster
        num_gpus = ray_cfg.gpus_per_node
        num_cpus = ray_cfg.resources.cpu_per_gpu * num_gpus

        init_kwargs["num_cpus"] = num_cpus
        init_kwargs["num_gpus"] = num_gpus

        # object store memory
        total_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        object_store_memory = int(
            total_memory * ray_cfg.runtime.object_store_memory_fraction
        )
        init_kwargs["object_store_memory"] = object_store_memory

        logger.info(
            f"Starting local Ray cluster: {num_cpus} CPUs, {num_gpus} GPUs, "
            f"{object_store_memory / 1e9:.1f}GB object store"
        )

    # dashboard
    if ray_cfg.dashboard_port:
        init_kwargs["dashboard_port"] = ray_cfg.dashboard_port

    ray.init(**init_kwargs)

    # log cluster info
    resources = ray.cluster_resources()
    logger.info(f"Ray cluster resources: {resources}")


def sync_registries() -> None:
    """Sync SkyRL registries to Ray workers."""
    try:
        import ray

        if not ray.is_initialized():
            return

        from skyrl_train.utils.ppo_utils import sync_registries

        sync_registries()
        logger.info("SkyRL registries synced with Ray workers")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to sync registries: {e}")


def create_experiment(cfg: DictConfig) -> Any:
    """Create GFlowNetExp from config.

    Args:
        cfg: Hydra config

    Returns:
        GFlowNetExp instance
    """
    from synthstats.distributed.gfn_exp import GFlowNetExp

    return GFlowNetExp(cfg)


def setup_wandb(cfg: DictConfig) -> None:
    """Initialize WandB if enabled."""
    if not cfg.wandb.enabled:
        return

    try:
        import wandb

        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.get("name"),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        logger.info(f"WandB initialized: {wandb.run.url}")
    except ImportError:
        logger.warning("wandb not installed, disabling")
        cfg.wandb.enabled = False
    except Exception as e:
        logger.warning(f"Failed to init WandB: {e}")
        cfg.wandb.enabled = False


def run_training(cfg: DictConfig) -> dict[str, Any]:
    """Run distributed training.

    Args:
        cfg: Hydra config

    Returns:
        Final training metrics
    """
    # create experiment
    logger.info("Creating GFlowNetExp...")
    exp = create_experiment(cfg)

    # sync registries
    sync_registries()

    # run training
    logger.info("Starting distributed training...")

    try:
        # the trainer.train() method handles the training loop
        exp.trainer.train()

        # collect final metrics
        metrics = {
            "status": "completed",
            "logZ": exp.trainer.logZ.item(),
        }

        # get buffer stats
        buffer_stats = exp.trainer.replay_buffer.get_stats()
        metrics.update({f"buffer_{k}": v for k, v in buffer_stats.items()})

        return metrics

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return {"status": "interrupted"}

    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return {"status": "failed", "error": str(e)}


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="distributed/gfn_base",
)
def main(cfg: DictConfig) -> None:
    """Main entry point for distributed training."""
    logger.info("=" * 60)
    logger.info("SynthStats Distributed GFlowNet Training")
    logger.info("=" * 60)

    # check dependencies
    if not check_skyrl_available():
        logger.error(
            "SkyRL not installed. Install with:\n"
            "  pip install skyrl-train\n"
            "Or for development:\n"
            "  pip install -e path/to/SkyRL/skyrl-train"
        )
        sys.exit(1)

    if not check_ray_available():
        logger.error("Ray not installed. Install with: pip install ray[default]")
        sys.exit(1)

    # log config
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # set seeds
    seed = cfg.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # init Ray
    init_ray(cfg)

    # setup WandB
    setup_wandb(cfg)

    try:
        # run training
        metrics = run_training(cfg)

        logger.info("=" * 60)
        logger.info("Training completed")
        logger.info(f"Final metrics: {metrics}")
        logger.info("=" * 60)

    finally:
        # cleanup
        import ray

        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray shutdown complete")

        if cfg.wandb.enabled:
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass

if __name__ == "__main__":
    main()
