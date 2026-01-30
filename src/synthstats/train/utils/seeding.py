"""RNG state management for reproducibility.

Provides utilities for seeding and state capture/restore across:
- PyTorch (CPU and CUDA)
- NumPy
- Python's random module
"""

from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Seed all random number generators.

    Args:
        seed: Seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Seeded all RNGs with {seed}")


def get_rng_states() -> dict[str, Any]:
    """Capture all RNG states for checkpointing.

    Returns:
        Dict with torch, numpy, random states (and torch_cuda if available)
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
    """Restore RNG states from checkpoint.

    Args:
        states: Dict from get_rng_states()
    """
    if "torch" in states:
        torch.set_rng_state(states["torch"])
    if "numpy" in states:
        np.random.set_state(states["numpy"])
    if "random" in states:
        random.setstate(states["random"])

    if "torch_cuda" in states and torch.cuda.is_available():
        saved_states = states["torch_cuda"]
        current_count = torch.cuda.device_count()
        saved_count = len(saved_states)

        if saved_count == current_count:
            torch.cuda.set_rng_state_all(saved_states)
        else:
            logger.warning(
                f"CUDA device count mismatch: checkpoint has {saved_count}, "
                f"current has {current_count}. CUDA RNG state not restored."
            )
