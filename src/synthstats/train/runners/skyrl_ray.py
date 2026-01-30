"""Ray + SkyRL distributed training runner.

Initializes Ray, syncs SkyRL registries, and runs distributed training.
All SkyRL-specific code is isolated here and in integrations/skyrl/.
"""

from __future__ import annotations

import logging
from typing import Any

from omegaconf import DictConfig

from synthstats.train.runners.base import RunResult

logger = logging.getLogger(__name__)


class SkyRLRayRunner:
    """Ray + SkyRL distributed training runner.

    EXPERIMENTAL: Not fully implemented. Use LocalRunner for single-node training.

    Initializes Ray cluster and runs training via SkyRL's BasePPOExp.

    Args:
        cfg: Hydra DictConfig with full configuration
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        # warn at construction time
        if cfg.get("runner", {}).get("experimental", False):
            logger.warning(
                "SkyRLRayRunner is experimental and not fully implemented. "
                "Use 'runner=local' for single-node training."
            )

    def run(self) -> RunResult:
        """Execute distributed training.

        Returns:
            RunResult with final metrics

        Raises:
            NotImplementedError: Full distributed training is not yet available.
        """
        raise NotImplementedError(
            "SkyRLRayRunner is experimental and not fully implemented. "
            "Use 'synthstats-train runner=local' for single-node training. "
            "See CLAUDE.md for available runners."
        )

    def state_dict(self) -> dict[str, Any]:
        """Not implemented (SkyRL manages state)."""
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Not implemented (SkyRL manages state)."""
        pass
