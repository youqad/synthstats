"""LikelihoodJudge - ELPD-based reward computation.

Computes rewards based on Expected Log Pointwise Predictive Density (ELPD-LOO).
Uses ArviZ for LOO-CV computation when PyMC inference data is available.
"""

import math
from typing import TYPE_CHECKING

from synthstats.core.types import Reward, Trajectory

if TYPE_CHECKING:
    import arviz as az


class LikelihoodJudge:
    """Computes reward from ELPD-LOO.

    Supports two modes:
    1. Pre-computed: reads artifacts["elpd"] if already computed
    2. Full computation: uses ArviZ to compute ELPD-LOO from artifacts["idata"]

    Args:
        beta: Scaling factor for ELPD in the exponent. Default 0.1.
        clip_range: (min, max) for log_reward before exp. Prevents overflow.
    """

    def __init__(
        self, beta: float = 0.1, clip_range: tuple[float, float] = (-700, 700)
    ):
        self.beta = beta
        self.clip_range = clip_range

    def _compute_elpd_loo(self, idata: "az.InferenceData") -> float:
        """Compute ELPD-LOO using ArviZ.

        Args:
            idata: ArviZ InferenceData with posterior_predictive and observed_data.

        Returns:
            ELPD-LOO estimate (sum of pointwise elpd values).
        """
        try:
            import arviz as az
        except ImportError as e:
            raise ImportError(
                "ArviZ required for ELPD-LOO computation. "
                "Install with: pip install arviz"
            ) from e

        loo_result = az.loo(idata, pointwise=True)
        return float(loo_result.elpd_loo)

    def score(
        self, *, task_name: str, trajectory: Trajectory, artifacts: dict
    ) -> Reward:
        """Compute reward from ELPD.

        Checks for:
        1. artifacts["idata"] - ArviZ InferenceData for full LOO computation
        2. artifacts["elpd"] - Pre-computed ELPD value

        Reward = exp(beta * elpd), clipped to prevent overflow.

        Args:
            task_name: Name of the task (unused in this judge).
            trajectory: Complete episode trajectory (unused in this judge).
            artifacts: Should contain "idata" or "elpd" for meaningful reward.

        Returns:
            Reward with exp(beta * elpd) as total.
        """
        elpd = 0.0
        elpd_source = "default"

        # prefer full computation from InferenceData
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
