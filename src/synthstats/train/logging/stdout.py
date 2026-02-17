"""Console logging sink."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class StdoutLogger:
    """Console logging with configurable formatting."""

    def __init__(
        self,
        log_interval: int = 1,
        metrics_format: dict[str, str] | None = None,
    ) -> None:
        self.log_interval = log_interval
        self.metrics_format = metrics_format or {
            "loss": ".4f",
            "tb_loss": ".4f",
            "logZ": ".4f",
            "avg_reward": ".4f",
            "entropy": ".4f",
        }

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        if self.log_interval > 1 and step % self.log_interval != 0:
            return

        parts = [f"step={step}"]
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                fmt = self.metrics_format.get(key, ".4f")
                parts.append(f"{key}={value:{fmt}}")

        logger.info(" | ".join(parts))

    def close(self) -> None:
        pass
