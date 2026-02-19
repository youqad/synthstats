"""Environment builders for training runners.

Preserves dependency inversion: train/ never imports tasks/ directly.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def build_env(cfg: Any) -> Any:
    """Build an environment from config. Requires cfg.env._target_."""
    env_cfg = deepcopy(cfg.get("env", {}) if hasattr(cfg, "get") else {})
    target = env_cfg.get("_target_", None) if hasattr(env_cfg, "get") else None
    if target is None:
        raise ValueError(
            "cfg.env must define _target_ for explicit environment instantiation "
            "(e.g., configs/env/*.yaml with _target_: synthstats.envs.boxing_env.BoxingEnv)"
        )

    judge_cfg = cfg.get("judge", None) if hasattr(cfg, "get") else None
    if judge_cfg is not None:
        judge_target = judge_cfg.get("_target_", None) if hasattr(judge_cfg, "get") else None
    else:
        judge_target = None
    if judge_target is not None:
        env_cfg["judge"] = deepcopy(judge_cfg)

    from hydra.utils import instantiate

    return instantiate(env_cfg)
