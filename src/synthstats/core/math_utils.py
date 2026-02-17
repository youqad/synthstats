"""Tensor math utilities."""

from __future__ import annotations

import torch
from torch import Tensor


def sanitize_finite(tensor: Tensor, fallback: float) -> Tensor:
    """Replace NaN/Inf values with a fallback constant."""
    return torch.where(torch.isfinite(tensor), tensor, torch.full_like(tensor, fallback))
