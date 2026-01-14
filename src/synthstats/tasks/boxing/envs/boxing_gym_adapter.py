"""Adapter for boxing-gym environments.

This module provides a thin wrapper that lets SynthStats tasks use
boxing_gym environments when available. It is import-safe: if the
boxing_gym package (or local repo) cannot be imported, callers can
fall back to the built-in stub environments.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Optional alias map for environment names.
_ENV_ALIASES: dict[str, str] = {
    "survival_analysis": "survival",
    "moral_machines": "morals",
}

# Query key hints used to strip "key=" prefixes.
_QUERY_KEYS: dict[str, tuple[str, ...]] = {
    "dugongs": ("age",),
    "peregrines": ("year", "time", "ti"),
}

# Result keys for formatting outputs.
_RESULT_KEYS: dict[str, str] = {
    "dugongs": "length",
    "peregrines": "population",
}


def _find_boxing_gym_src() -> Path | None:
    """Locate a local boxing-gym repo (boxing-gym-wip) if present."""
    env_path = os.environ.get("BOXING_GYM_SRC")
    if env_path:
        candidate = Path(env_path).expanduser().resolve()
        if candidate.exists():
            return candidate

    # Walk up the filesystem and look for a sibling repo.
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "boxing-gym-wip" / "src"
        if candidate.exists():
            return candidate

    return None


def _import_boxing_gym() -> Any | None:
    """Import boxing_gym, optionally adding a local repo to sys.path."""
    try:
        return importlib.import_module("boxing_gym")
    except Exception:
        pass

    local_src = _find_boxing_gym_src()
    if local_src is None:
        return None

    local_src_str = str(local_src)
    if local_src_str not in sys.path:
        sys.path.insert(0, local_src_str)

    try:
        return importlib.import_module("boxing_gym")
    except Exception as exc:
        logger.debug("Failed to import boxing_gym from %s: %s", local_src, exc)
        return None


class BoxingGymAdapter:
    """Adapter exposing a query/reset interface over boxing_gym envs."""

    def __init__(self, env_name: str, env: Any):
        self.env_name = env_name
        self._env = env
        self.backend = "boxing_gym"
        self._query_keys = _QUERY_KEYS.get(env_name, ())
        self._result_key = _RESULT_KEYS.get(env_name)

        # Some envs use include_prior to format descriptions.
        if hasattr(self._env, "include_prior"):
            try:
                self._env.include_prior = True
            except Exception:
                pass

    def reset(self, seed: int | None = None) -> None:
        """Reset the underlying environment."""
        if seed is not None:
            try:
                import random

                random.seed(seed)
            except Exception:
                pass
            try:
                import numpy as np

                np.random.seed(seed)
            except Exception:
                pass

        reset_fn = getattr(self._env, "reset", None)
        if not callable(reset_fn):
            return

        try:
            params = inspect.signature(reset_fn).parameters
            if "seed" in params:
                reset_fn(seed=seed)
            else:
                reset_fn()
        except (TypeError, ValueError):
            reset_fn()

    def query(self, query: str) -> str:
        """Run a single experiment query and format the response."""
        input_string = self._normalize_query(query)

        if not hasattr(self._env, "run_experiment"):
            return "Environment does not support run_experiment()."

        try:
            result = self._env.run_experiment(input_string)
        except Exception as exc:
            return f"Error running experiment: {exc}"

        # Normalize result shape
        success = True
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], bool):
            result, success = result

        if not success:
            return str(result)

        return self._format_result(result)

    def _normalize_query(self, query: str) -> str:
        text = query.strip()
        for key in self._query_keys:
            marker = f"{key}="
            if marker in text:
                return text.split(marker, 1)[1].strip()
        return text

    def _format_result(self, result: Any) -> str:
        if self._result_key is None:
            return str(result)

        # Match stub formatting (float rounding for dugongs).
        if isinstance(result, float):
            value = f"{result:.3f}"
        else:
            value = str(result)
        return f"{self._result_key}={value}"

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the underlying env."""
        return getattr(self._env, name)


def load_boxing_gym_env(env_name: str) -> BoxingGymAdapter | None:
    """Create a boxing_gym environment adapter if available."""
    boxing_gym = _import_boxing_gym()
    if boxing_gym is None:
        return None

    try:
        from boxing_gym.envs.registry import get_environment_registry
    except Exception as exc:
        logger.debug("boxing_gym registry unavailable: %s", exc)
        return None

    resolved_name = _ENV_ALIASES.get(env_name, env_name)
    try:
        name_to_env, _ = get_environment_registry()
    except Exception as exc:
        logger.debug("Failed to load boxing_gym registry: %s", exc)
        return None

    env_cls = name_to_env.get(resolved_name)
    if env_cls is None:
        return None

    try:
        env = env_cls()
    except Exception as exc:
        logger.debug("Failed to instantiate boxing_gym env %s: %s", resolved_name, exc)
        return None

    return BoxingGymAdapter(resolved_name, env)
