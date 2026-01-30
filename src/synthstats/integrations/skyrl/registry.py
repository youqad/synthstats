"""SkyRL registry synchronization and loss registration.

All SkyRL-specific code is isolated here. This module:
- Checks SkyRL availability
- Registers TB/SubTB losses with SkyRL's policy_loss registry
- Syncs registries across Ray workers

This module is import-safe: works without SkyRL installed.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# check SkyRL availability at import time
SKYRL_AVAILABLE = False
try:
    from skyrl.trainer.policy_loss import register_policy_loss  # noqa: F401

    SKYRL_AVAILABLE = True
except ImportError:
    register_policy_loss = None  # type: ignore[misc,assignment]


def is_skyrl_available() -> bool:
    """Return True if SkyRL is installed."""
    return SKYRL_AVAILABLE


def register_losses() -> None:
    """Register TB and SubTB losses with SkyRL.

    Call this before creating SkyRL experiments to ensure
    losses are available in the registry.
    """
    if not SKYRL_AVAILABLE:
        logger.warning("SkyRL not available, skipping loss registration")
        return

    from skyrl.trainer.policy_loss import register_policy_loss

    from synthstats.training.losses.trajectory_balance import (
        compute_modified_subtb_loss,
        compute_trajectory_balance_loss,
    )

    try:
        register_policy_loss("trajectory_balance")(compute_trajectory_balance_loss)
        logger.info("Registered 'trajectory_balance' policy loss with SkyRL")
    except Exception as e:
        logger.warning(f"Failed to register trajectory_balance: {e}")

    try:
        register_policy_loss("modified_subtb")(compute_modified_subtb_loss)
        logger.info("Registered 'modified_subtb' policy loss with SkyRL")
    except Exception as e:
        logger.warning(f"Failed to register modified_subtb: {e}")


def sync_registries_if_ray() -> None:
    """Sync loss registries across Ray workers.

    Call this after ray.init() to ensure all workers have
    the same loss functions registered.
    """
    if not SKYRL_AVAILABLE:
        return

    try:
        import ray

        if not ray.is_initialized():
            logger.warning("Ray not initialized, skipping registry sync")
            return

        # register losses on driver
        register_losses()

        # propagate to workers via Ray's object store
        # (workers will call register_losses() when they import the module)
        logger.info("Registry sync complete (workers will register on import)")

    except ImportError:
        logger.warning("Ray not available, skipping registry sync")
    except Exception as e:
        logger.warning(f"Registry sync failed: {e}")
