"""Device resolution utilities."""

from __future__ import annotations

from typing import Any

import torch


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


def get_device_info() -> dict[str, Any]:
    """Get information about available devices.

    Returns:
        Dict with device availability and properties
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        info["cuda_devices"] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["cuda_devices"].append(
                {
                    "name": props.name,
                    "total_memory_gb": props.total_memory / (1024**3),
                    "major": props.major,
                    "minor": props.minor,
                }
            )

    return info
