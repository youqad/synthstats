"""Loss functions for GFlowNet training.

Two implementations are available:
1. Standalone: `subtb_loss()` - for non-SkyRL training loops
2. SkyRL-integrated: TB and SubTB losses registered with SkyRL's PolicyLossRegistry

SkyRL-Integrated Losses:
- `trajectory_balance`: Vanilla TB loss (whole-trajectory matching)
- `modified_subtb`: SubTB loss with lambda-weighted sub-trajectory matching

Use `SKYRL_REGISTERED` to check if SkyRL integration is available.
"""

from synthstats.training.losses.tb_loss import subtb_loss

# register TB/SubTB losses with SkyRL on import
try:
    from synthstats.training.losses.trajectory_balance import (
        SKYRL_REGISTERED,
        compute_modified_subtb_loss,
        compute_tb_identity_advantage,
        compute_trajectory_balance_loss,
    )
except ImportError:
    SKYRL_REGISTERED = False
    compute_trajectory_balance_loss = None
    compute_modified_subtb_loss = None
    compute_tb_identity_advantage = None

__all__ = [
    # standalone loss
    "subtb_loss",
    # SkyRL-integrated losses
    "compute_trajectory_balance_loss",
    "compute_modified_subtb_loss",
    "compute_tb_identity_advantage",
    "SKYRL_REGISTERED",
]
