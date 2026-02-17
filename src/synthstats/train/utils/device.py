"""Device resolution utilities."""

from __future__ import annotations

import torch


def normalize_device(device: str | torch.device) -> torch.device:
    """Normalize a device specification to a torch.device object.

    Args:
        device: Device as string or torch.device

    Returns:
        torch.device instance
    """
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def resolve_device(device_str: str) -> str:
    """Resolve device string.

    Args:
        device_str: Device specification ("auto", "cpu", "cuda", "cuda:0", etc.)

    Returns:
        Resolved device string
    """
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str
