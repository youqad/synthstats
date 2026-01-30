"""Checkpointing utilities for training state persistence.

Provides complete training state serialization including:
- Model and optimizer state dicts
- Learnable parameters (logZ)
- RNG states for reproducibility
- Replay buffer contents
- Metrics history

Usage:
    from synthstats.training.checkpointing import (
        CheckpointState, save_checkpoint, load_checkpoint
    )

    # save
    state = CheckpointState(
        step_count=100,
        logZ=trainer.logZ.item(),
        model_state_dict=policy.model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        rng_states=get_rng_states(),
        replay_buffer=buffer.state_dict() if buffer else None,
        config=asdict(config),
        metrics_history=metrics,
    )
    save_checkpoint(Path("checkpoint.pt"), state)

    # load
    state = load_checkpoint(Path("checkpoint.pt"))
    policy.model.load_state_dict(state.model_state_dict)
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    """Complete training state for checkpoint/resume.

    All fields are serializable via torch.save.
    """

    step_count: int
    logZ: float
    model_state_dict: dict[str, Any] | None
    optimizer_state_dict: dict[str, Any] | None
    rng_states: dict[str, Any]
    replay_buffer: dict[str, Any] | None
    config: dict[str, Any]
    metrics_history: list[dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        # NOTE: dataclasses.asdict() deep-copies values; for torch tensors in
        # state_dicts this can clone the full model weights and blow up peak RAM.
        # We only need a shallow mapping for torch.save().
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointState:
        """Create CheckpointState from dictionary."""
        return cls(**data)


def get_rng_states() -> dict[str, Any]:
    """Capture all RNG states for reproducibility.

    Captures states from:
    - torch (CPU and CUDA if available)
    - numpy
    - Python's random module

    Returns:
        Dict with keys "torch", "torch_cuda", "numpy", "random"
    """
    states: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "random": random.getstate(),
    }

    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()

    return states


def set_rng_states(states: dict[str, Any]) -> None:
    """Restore all RNG states from checkpoint.

    Args:
        states: Dict from get_rng_states() with torch, numpy, random states

    Note:
        CUDA states are only restored if device count matches. A warning is
        logged if checkpoint was saved with different GPU count.
    """
    if "torch" in states:
        torch.set_rng_state(states["torch"])

    if "numpy" in states:
        np.random.set_state(states["numpy"])

    if "random" in states:
        random.setstate(states["random"])

    if "torch_cuda" in states and torch.cuda.is_available():
        saved_states = states["torch_cuda"]
        current_device_count = torch.cuda.device_count()
        saved_device_count = len(saved_states)

        if saved_device_count == current_device_count:
            torch.cuda.set_rng_state_all(saved_states)
        else:
            logger.warning(
                f"CUDA device count mismatch: checkpoint has {saved_device_count}, "
                f"current has {current_device_count}. CUDA RNG state not restored."
            )


def save_checkpoint(path: Path, state: CheckpointState) -> None:
    """Save checkpoint to disk.

    Creates parent directories if needed. Uses torch.save for serialization.

    Args:
        path: Destination path for checkpoint file
        state: CheckpointState with complete training state
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_dict = state.to_dict()
    torch.save(checkpoint_dict, path)

    logger.info(f"Saved checkpoint to {path} (step {state.step_count})")


def load_checkpoint(path: Path) -> CheckpointState:
    """Load checkpoint from disk.

    Args:
        path: Path to checkpoint file

    Returns:
        CheckpointState with restored training state

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint_dict = torch.load(path, weights_only=False)
    state = CheckpointState.from_dict(checkpoint_dict)

    logger.info(f"Loaded checkpoint from {path} (step {state.step_count})")

    return state


def cleanup_old_checkpoints(
    checkpoint_dir: Path,
    keep_last_n: int,
    pattern: str = "checkpoint_*.pt",
) -> list[Path]:
    """Remove old checkpoints, keeping only the most recent N.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of checkpoints to keep
        pattern: Glob pattern for checkpoint files

    Returns:
        List of removed checkpoint paths
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []

    checkpoints = sorted(checkpoint_dir.glob(pattern), key=lambda p: p.stat().st_mtime)

    if keep_last_n <= 0 or len(checkpoints) <= keep_last_n:
        return []

    to_remove = checkpoints[:-keep_last_n]
    removed = []

    for ckpt in to_remove:
        try:
            ckpt.unlink()
            removed.append(ckpt)
            logger.debug(f"Removed old checkpoint: {ckpt}")
        except OSError as e:
            logger.warning(f"Failed to remove checkpoint {ckpt}: {e}")

    if removed:
        logger.info(f"Cleaned up {len(removed)} old checkpoints")

    return removed
