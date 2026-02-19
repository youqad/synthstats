"""Weighted combination of multiple judges."""

from __future__ import annotations

import math
from typing import Any

from omegaconf import DictConfig, ListConfig

from synthstats.core.judge import Judge
from synthstats.core.types import Reward, Trajectory


class CompositeJudge:
    """Combines multiple judges with configurable weights.

    Each sub-judge contributes its score multiplied by its weight to the
    total. Individual scores are tracked in components for analysis.

    Example:
        judge = CompositeJudge([
            (LikelihoodJudge(), 0.7),
            (FormattingJudge(), 0.3),
        ])
        reward = judge.score(task_name="test", trajectory=traj, artifacts={})
        # reward.total = 0.7 * likelihood_score + 0.3 * formatting_score

    Args:
        judges: List of (judge, weight) tuples.
    """

    def __init__(
        self,
        judges: list[tuple[Judge, float]] | list[dict[str, Any]],
        *,
        scalarization_mode: str = "weighted_log_product",
        min_factor: float = 1e-12,
    ):
        self.judges = self._normalize_judges(judges)
        self.scalarization_mode = scalarization_mode
        self.min_factor = min_factor
        supported = {"weighted_sum", "weighted_log_product"}
        if self.scalarization_mode not in supported:
            raise ValueError(
                f"Unsupported scalarization_mode={scalarization_mode!r}; "
                "expected 'weighted_sum' or 'weighted_log_product'"
            )

    def _normalize_judges(
        self,
        judges: list[tuple[Judge, float]] | list[dict[str, Any]],
    ) -> list[tuple[Judge, float]]:
        normalized: list[tuple[Judge, float]] = []
        candidates = list(judges) if isinstance(judges, ListConfig) else judges
        for spec in candidates:
            judge_obj: Any
            weight_val: Any

            if isinstance(spec, tuple | list):
                if len(spec) != 2:
                    raise ValueError(
                        "Tuple/list judge specs must be (judge, weight) pairs in CompositeJudge"
                    )
                judge_obj, weight_val = spec
            elif isinstance(spec, dict) or isinstance(spec, DictConfig):
                judge_obj = spec.get("judge")
                weight_val = spec.get("weight", 1.0)
            else:
                raise TypeError(
                    "CompositeJudge judges entries must be (judge, weight) pairs "
                    "or {'judge': ..., 'weight': ...} dicts"
                )

            if judge_obj is None:
                raise ValueError("CompositeJudge received a judge spec with judge=None")
            normalized.append((judge_obj, float(weight_val)))
        return normalized

    def score(self, *, task_name: str, trajectory: Trajectory, artifacts: dict) -> Reward:
        """Compute weighted combination of sub-judge scores.

        Args:
            task_name: Passed to each sub-judge.
            trajectory: Passed to each sub-judge.
            artifacts: Passed to each sub-judge.

        Returns:
            Reward with weighted total and individual scores in components.
        """
        if not self.judges:
            return Reward(
                total=0.0,
                components={},
                factors={},
                scalarization=self.scalarization_mode,
                info={"scalarization_mode": self.scalarization_mode, "factor_weights": {}},
            )

        total = 0.0
        weighted_log_total = 0.0
        components: dict[str, float] = {}
        factors: dict[str, float] = {}
        factor_weights: dict[str, float] = {}
        label_counts: dict[str, int] = {}

        for judge, weight in self.judges:
            sub_reward = judge.score(
                task_name=task_name, trajectory=trajectory, artifacts=artifacts
            )
            judge_name = type(judge).__name__
            seen = label_counts.get(judge_name, 0)
            label_counts[judge_name] = seen + 1
            label = judge_name if seen == 0 else f"{judge_name}#{seen + 1}"

            raw_total = float(sub_reward.total)
            components[label] = raw_total
            factors[label] = raw_total
            factor_weights[label] = float(weight)

            if self.scalarization_mode == "weighted_sum":
                total += weight * raw_total
            else:
                weighted_log_total += weight * math.log(max(raw_total, self.min_factor))

        if self.scalarization_mode == "weighted_log_product":
            total = math.exp(weighted_log_total)

        return Reward(
            total=total,
            components=components,
            factors=factors,
            scalarization=self.scalarization_mode,
            info={
                "scalarization_mode": self.scalarization_mode,
                "factor_weights": factor_weights,
            },
        )
