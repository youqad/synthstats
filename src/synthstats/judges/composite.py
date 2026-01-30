"""CompositeJudge - combines multiple judges with configurable weights."""

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

    def __init__(self, judges: list[tuple[Judge, float]]):
        self.judges = judges

    def score(self, *, task_name: str, trajectory: Trajectory, artifacts: dict) -> Reward:
        """Compute weighted combination of sub-judge scores.

        Args:
            task_name: Passed to each sub-judge.
            trajectory: Passed to each sub-judge.
            artifacts: Passed to each sub-judge.

        Returns:
            Reward with weighted total and individual scores in components.
        """
        total = 0.0
        components: dict[str, float] = {}

        for judge, weight in self.judges:
            sub_reward = judge.score(
                task_name=task_name, trajectory=trajectory, artifacts=artifacts
            )
            total += weight * sub_reward.total
            components[type(judge).__name__] = sub_reward.total

        return Reward(total=total, components=components, info={})
