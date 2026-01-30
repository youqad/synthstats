"""External framework integrations.

Submodules:
- skyrl: SkyRL/Ray distributed training (registry, adapters)
- tinker: Tinker API training backend (policy, trainer, adapters)

Usage:
    from synthstats.integrations.skyrl import sync_registries_if_ray, register_losses
    from synthstats.integrations.tinker import TinkerPolicy, TinkerTrainer
"""

from synthstats.integrations import skyrl, tinker

__all__ = ["skyrl", "tinker"]
