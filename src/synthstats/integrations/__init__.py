"""External framework integrations.

The SkyRL integration module has been archived (January 2026) in favor of
native SkyRL integration using @register_policy_loss.

See:
- synthstats.training.losses.trajectory_balance for TB loss registration
- synthstats.training.tb_trainer for TBTrainer
- synthstats.envs.boxing_env for native BaseTextEnv

Archived code: _archive/skyrl_integration_2026-01/
"""

from synthstats.integrations import tinker_adapter

__all__ = ["tinker_adapter"]
