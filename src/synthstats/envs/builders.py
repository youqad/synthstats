"""Environment builders for training runners.

Preserves dependency inversion: train/ never imports tasks/ directly.
"""

from __future__ import annotations

from typing import Any


def build_env(cfg: Any) -> Any:
    """Build an environment from config. Requires cfg.env._target_."""
    env_cfg = cfg.get("env", {}) if hasattr(cfg, "get") else {}
    target = env_cfg.get("_target_", None) if hasattr(env_cfg, "get") else None
    if target is None:
        raise ValueError(
            "cfg.env must define _target_ for explicit environment instantiation "
            "(e.g., configs/env/*.yaml with _target_: synthstats.envs.boxing_env.BoxingEnv)"
        )

    from hydra.utils import instantiate

    return instantiate(env_cfg)
