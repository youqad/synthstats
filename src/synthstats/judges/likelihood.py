"""LikelihoodJudge - ELPD-based reward computation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from synthstats.core.types import Reward, Trajectory

if TYPE_CHECKING:
    import arviz as az


class LikelihoodJudge:
    """Reward = exp(beta * ELPD-LOO). Uses ArviZ when idata is available."""

    def __init__(self, beta: float = 0.1, clip_range: tuple[float, float] = (-700, 700)):
        self.beta = beta
        self.clip_range = clip_range

    def _compute_elpd_loo(self, idata: az.InferenceData) -> float:
        try:
            import arviz as az
        except ImportError as e:
            raise ImportError(
                "ArviZ required for ELPD-LOO computation. Install with: pip install arviz"
            ) from e

        loo_result = az.loo(idata, pointwise=True)
        return float(loo_result.elpd_loo)

    def score(self, *, task_name: str, trajectory: Trajectory, artifacts: dict) -> Reward:
        elpd = 0.0
        elpd_source = "default"

        if "idata" in artifacts:
            try:
                elpd = self._compute_elpd_loo(artifacts["idata"])
                elpd_source = "loo_cv"
            except Exception as e:
                # fall back to pre-computed if LOO fails
                if "elpd" in artifacts:
                    elpd = artifacts["elpd"]
                    elpd_source = "precomputed_fallback"
                else:
                    elpd_source = f"error: {e}"
        elif "elpd" in artifacts:
            elpd = artifacts["elpd"]
            elpd_source = "precomputed"

        log_reward = self.beta * elpd
        log_reward = max(self.clip_range[0], min(self.clip_range[1], log_reward))
        total = math.exp(log_reward)

        return Reward(
            total=total,
            components={"elpd": elpd},
            info={"log_reward": log_reward, "elpd_source": elpd_source},
        )
