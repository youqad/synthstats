"""Weights & Biases logging sink."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class WandbLogger:
    """Weights & Biases logging implementation.

    Args:
        project: W&B project name
        entity: W&B entity (username or team)
        name: Run name (auto-generated if None)
        tags: Run tags
        config: Config dict to log
    """

    def __init__(
        self,
        project: str = "synthstats",
        entity: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.project = project
        self.entity = entity
        self.name = name
        self.tags = tags or []
        self.config = config or {}
        self._run = None

    def _ensure_init(self) -> None:
        """Lazy initialize W&B."""
        if self._run is not None:
            return

        try:
            import wandb

            self._run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=self.name,
                tags=self.tags,
                config=self.config,
            )
            run_name = self._run.name if self._run else "unknown"
            logger.info(f"Initialized W&B run: {run_name}")
        except ImportError:
            logger.warning("wandb not installed, logging disabled")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        """Log metrics to W&B."""
        self._ensure_init()
        if self._run is None:
            return

        try:
            import wandb

            wandb.log(metrics, step=step)
        except Exception as e:
            logger.warning(f"W&B log failed: {e}")

    def close(self) -> None:
        """Finish W&B run."""
        if self._run is not None:
            try:
                import wandb

                wandb.finish()
            except Exception as e:
                logger.warning(f"W&B finish failed: {e}")
            self._run = None
