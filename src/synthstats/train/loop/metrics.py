"""Shared metric utilities for training loops."""

from __future__ import annotations


def summarize_eval_metrics(
    rewards: list[float],
    *,
    episodes: int,
    logZ: float,
) -> dict[str, float]:
    """Compute standard eval metrics from per-episode rewards."""
    if not rewards:
        return {
            "eval_avg_reward": 0.0,
            "eval_max_reward": 0.0,
            "eval_min_reward": 0.0,
            "eval_success_rate": 0.0,
            "eval_episodes": episodes,
            "logZ": logZ,
        }

    avg_reward = sum(rewards) / len(rewards)
    max_reward = max(rewards)
    min_reward = min(rewards)
    success_rate = sum(1 for r in rewards if r > 0) / len(rewards)

    return {
        "eval_avg_reward": avg_reward,
        "eval_max_reward": max_reward,
        "eval_min_reward": min_reward,
        "eval_success_rate": success_rate,
        "eval_episodes": episodes,
        "logZ": logZ,
    }
