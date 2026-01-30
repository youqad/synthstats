"""SkyRL shape and format adapters.

Adapters for converting between SynthStats and SkyRL data formats.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor


def batch_to_skyrl_format(batch: dict[str, Any]) -> dict[str, Any]:
    """Convert SynthStats batch to SkyRL's expected format.

    SkyRL expects certain tensor shapes and naming conventions.
    This adapter handles the translation.

    Args:
        batch: SynthStats batch with keys like log_probs, loss_mask, log_reward

    Returns:
        Batch in SkyRL's expected format
    """
    result = {}

    # map log_probs -> log_probs (same)
    if "log_probs" in batch:
        result["log_probs"] = batch["log_probs"]

    # map loss_mask -> response_mask (SkyRL naming)
    if "loss_mask" in batch:
        result["response_mask"] = batch["loss_mask"]

    # map log_reward -> advantages (via tb_identity estimator)
    if "log_reward" in batch:
        log_rewards = batch["log_reward"]
        # expand to sequence length if needed
        if "log_probs" in batch and log_rewards.dim() == 1:
            seq_len = batch["log_probs"].shape[-1]
            log_rewards = log_rewards.unsqueeze(-1).expand(-1, seq_len)
        result["advantages"] = log_rewards

    # pass through other keys
    for key in batch:
        if key not in result:
            result[key] = batch[key]

    return result


def skyrl_result_to_metrics(result: Any) -> dict[str, float]:
    """Convert SkyRL training result to metrics dict.

    Args:
        result: SkyRL's training result object

    Returns:
        Dict of metrics suitable for logging
    """
    metrics: dict[str, float] = {}

    if hasattr(result, "loss"):
        loss = result.loss
        metrics["loss"] = loss.item() if isinstance(loss, Tensor) else float(loss)

    if hasattr(result, "clip_ratio"):
        metrics["clip_ratio"] = float(result.clip_ratio)

    if hasattr(result, "metrics") and isinstance(result.metrics, dict):
        for k, v in result.metrics.items():
            if isinstance(v, (int, float)):
                metrics[k] = float(v)
            elif isinstance(v, Tensor):
                metrics[k] = v.item()

    return metrics
