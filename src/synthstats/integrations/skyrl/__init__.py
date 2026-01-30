"""SkyRL integration layer.

All SkyRL-specific code is isolated here:
- registry: SkyRL loss registration and Ray sync
- adapters: Shape/format adapters for SkyRL interfaces
"""

from synthstats.integrations.skyrl.registry import (
    SKYRL_AVAILABLE,
    register_losses,
    sync_registries_if_ray,
)

__all__ = [
    "SKYRL_AVAILABLE",
    "sync_registries_if_ray",
    "register_losses",
]
