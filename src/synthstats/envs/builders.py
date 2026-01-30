"""Environment builders for training runners.

This module exists to preserve dependency inversion:
- `synthstats.train.*` must not import `synthstats.tasks.*` directly.

Training runners should call `build_env(cfg)` and treat the returned object as an
opaque environment that satisfies the collector's expectations.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_env(cfg: Any) -> Any:
    """Build an environment instance from config.

    If `cfg.env` contains a Hydra `_target_`, it is instantiated directly.
    Otherwise, we fall back to the default BoxingGym environment wrapper.
    """
    env_cfg = cfg.get("env", {}) if hasattr(cfg, "get") else {}

    target = env_cfg.get("_target_", None) if hasattr(env_cfg, "get") else None
    if target is not None:
        from hydra.utils import instantiate

        return instantiate(env_cfg)

    # Default path for current training configs: BoxingGym wrapper with an env name.
    return build_boxing_env(env_cfg)


def build_boxing_env(env_cfg: Any) -> Any:
    """Build the default BoxingGym environment (BoxingEnv + BoxingTask)."""
    from synthstats.envs.boxing_env import BoxingEnv, BoxingEnvConfig
    from synthstats.judges.composite import CompositeJudge
    from synthstats.judges.likelihood import LikelihoodJudge
    from synthstats.tasks.boxing import BoxingCodec
    from synthstats.tasks.boxing.task import BoxingTask

    env_name = env_cfg.get("name", "dugongs") if hasattr(env_cfg, "get") else "dugongs"
    max_steps = env_cfg.get("max_steps", 20) if hasattr(env_cfg, "get") else 20

    task = BoxingTask(
        env_name=env_name,
        max_steps=max_steps,
    )
    codec = BoxingCodec()
    judge = CompositeJudge([(LikelihoodJudge(beta=1.0), 1.0)])

    executors: dict[str, Any] = {}
    try:
        from synthstats.executors.pymc_sandbox import PyMCExecutor

        executors["pymc"] = PyMCExecutor()
    except Exception as exc:
        logger.warning("Failed to init PyMCExecutor: %s", exc)

    config = BoxingEnvConfig(max_turns=max_steps)
    return BoxingEnv(task=task, codec=codec, executors=executors, judge=judge, config=config)
