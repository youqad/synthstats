"""
Trajectory Balance (TB) and Sub-Trajectory Balance (SubTB) trainers for SkyRL.

These trainers extend SkyRL's RayPPOTrainer to support GFlowNet training
with TB and SubTB objectives.

Architecture:
1. logZ: Learned log partition function, injected via config before loss
2. tb_identity estimator: Passes log_rewards through as "advantages"
3. Config injection: logZ (and eos_logprobs for SubTB) travel via config

Available Trainers:
- TBTrainer: Vanilla TB - whole trajectory flow matching
- SubTBTrainer: SubTB - sub-trajectory flow matching with lambda weighting

Usage:
    # Vanilla TB
    trainer = TBTrainer(config)
    trainer.train()

    # SubTB (requires eos_logprobs in batch data)
    trainer = SubTBTrainer(config)
    trainer.train()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class LogZModule(nn.Module):
    """Learnable log partition function for GFlowNets.

    This is a simple module that holds logZ as a parameter,
    allowing it to be optimized alongside model weights.
    """

    def __init__(self, init_value: float = 0.0) -> None:
        super().__init__()
        self.logZ = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        """Return current logZ value."""
        return self.logZ


class TBTrainerMixin:
    """Mixin that adds TB-specific functionality to any trainer.

    This mixin provides:
    1. logZ parameter management
    2. Config injection for logZ
    3. logZ optimization step
    """

    def __init__(
        self,
        *args: Any,
        logZ_init: float = 0.0,
        logZ_lr: float = 1e-2,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.logZ_module = LogZModule(logZ_init)
        self.logZ_optimizer = torch.optim.Adam(
            self.logZ_module.parameters(),
            lr=logZ_lr,
        )
        self._tb_step_count = 0

    @property
    def logZ(self) -> torch.Tensor:
        """Current logZ value."""
        return self.logZ_module.logZ

    def inject_logZ_into_config(self, config: DictConfig | dict[str, Any]) -> None:
        """Inject current logZ value into config for loss computation.

        This must be called before each training step to ensure the
        TB loss has access to the current logZ value.

        The tensor is injected via _logZ_tensor attribute to preserve
        gradients for logZ learning. The float value is also set for
        compatibility with config serialization.

        Args:
            config: Training config (will be modified in-place)
        """
        logZ_val = self.logZ.item()

        # inject tensor for gradient flow (set as Python attribute, not OmegaConf key)
        # this preserves the computational graph so logZ can learn
        # use object.__setattr__ to bypass OmegaConf's type validation
        try:
            object.__setattr__(config, "_logZ_tensor", self.logZ)
        except (AttributeError, TypeError):
            # plain dict or frozen config - tensor injection not possible
            # logZ will be created as a new tensor in the loss function
            logger.debug("Could not inject logZ tensor - using float fallback")

        # also inject float for compatibility/serialization
        if hasattr(config, "algorithm"):
            # OmegaConf DictConfig with algorithm section
            config.algorithm.logZ = logZ_val
        elif isinstance(config, dict):
            # plain dict
            if "algorithm" in config:
                config["algorithm"]["logZ"] = logZ_val
            else:
                config["logZ"] = logZ_val
        else:
            # try direct attribute assignment
            config.logZ = logZ_val

        logger.debug(f"Injected logZ={logZ_val:.4f} into config (with tensor)")

    def tb_optimizer_step(self, loss: torch.Tensor | None = None) -> dict[str, float]:
        """Perform logZ optimizer step.

        If loss is provided, computes gradients for logZ.
        Should be called after the main optimizer step.

        Args:
            loss: Optional loss tensor to backprop for logZ gradients

        Returns:
            Metrics dict with logZ value
        """
        if loss is not None and loss.requires_grad:
            # compute logZ gradients
            # note: loss should already have been backward'd for main optimizer,
            # but logZ might need its own gradient computation
            self.logZ_optimizer.zero_grad()
            # recompute gradients for logZ only
            loss.backward(retain_graph=True)

        self.logZ_optimizer.step()
        self.logZ_optimizer.zero_grad()
        self._tb_step_count += 1

        return {
            "logZ": self.logZ.item(),
            "tb_step": self._tb_step_count,
        }


# try to import SkyRL trainer for native integration
try:
    from skyrl_train.trainer import RayPPOTrainer

    class TBTrainer(TBTrainerMixin, RayPPOTrainer):
        """Trajectory Balance trainer extending SkyRL's RayPPOTrainer.

        This trainer:
        1. Uses the registered trajectory_balance loss + tb_identity estimator
        2. Manages logZ as a separate learnable parameter
        3. Injects logZ into config before each loss computation

        Config requirements:
            trainer.algorithm.policy_loss_type: trajectory_balance
            trainer.algorithm.advantage_estimator: tb_identity
        """

        def __init__(
            self,
            config: DictConfig,
            *args: Any,
            logZ_init: float | None = None,
            logZ_lr: float | None = None,
            **kwargs: Any,
        ) -> None:
            # extract TB params from config if not provided
            if logZ_init is None:
                logZ_init = config.get("logZ_init", 0.0)
            if logZ_lr is None:
                logZ_lr = config.get("logZ_lr", 1e-2)

            # initialize mixin first (sets up logZ)
            TBTrainerMixin.__init__(
                self,
                logZ_init=logZ_init,
                logZ_lr=logZ_lr,
            )
            # then initialize SkyRL trainer
            RayPPOTrainer.__init__(self, config, *args, **kwargs)

            # verify config
            self._verify_tb_config(config)

        def _verify_tb_config(self, config: DictConfig) -> None:
            """Verify config has correct TB settings."""
            algorithm = config.trainer.algorithm

            loss_type = algorithm.get("policy_loss_type", "regular")
            if loss_type != "trajectory_balance":
                logger.warning(
                    f"TBTrainer expects policy_loss_type='trajectory_balance', "
                    f"got '{loss_type}'. TB training may not work correctly."
                )

            adv_estimator = algorithm.get("advantage_estimator", "gae")
            if adv_estimator != "tb_identity":
                logger.warning(
                    f"TBTrainer expects advantage_estimator='tb_identity', "
                    f"got '{adv_estimator}'. Log rewards may not be passed correctly."
                )

        def train_critic_and_policy(self, data: Any) -> Any:
            """Train step with TB-specific handling.

            Injects logZ into config before training, then updates logZ
            after the main optimizer step.
            """
            # inject current logZ into config before loss computation
            self.inject_logZ_into_config(self.cfg)

            # run standard training step
            result = super().train_critic_and_policy(data)

            # perform logZ optimizer step
            # note: we don't pass loss here because SkyRL already did backward
            # and cleared gradients. logZ gets its gradient from being in the
            # computation graph during the loss calculation.
            tb_metrics = self.tb_optimizer_step()

            # add TB metrics to training output
            if hasattr(self, "all_metrics"):
                self.all_metrics.update(tb_metrics)

            return result

    SKYRL_AVAILABLE = True

except ImportError:
    # SkyRL not available, provide a fallback class
    class TBTrainer(TBTrainerMixin):  # type: ignore[no-redef]
        """Standalone TB trainer when SkyRL is not available.

        This provides the same interface but requires manual
        implementation of the training loop.
        """

        def __init__(
            self,
            *args: Any,
            logZ_init: float = 0.0,
            logZ_lr: float = 1e-2,
            **kwargs: Any,
        ) -> None:
            super().__init__(logZ_init=logZ_init, logZ_lr=logZ_lr)
            logger.warning(
                "SkyRL not available. TBTrainer provides logZ management only. "
                "Use TBTrainerMixin with your own trainer base class."
            )

    SKYRL_AVAILABLE = False


# -----------------------------------------------------------------------------
# SubTB Trainer Mixin (extends TBTrainerMixin for EOS logprobs injection)
# -----------------------------------------------------------------------------


class SubTBTrainerMixin(TBTrainerMixin):
    """Mixin that adds SubTB-specific functionality to any trainer.

    Extends TBTrainerMixin with:
    1. EOS logprobs tensor injection into config
    2. SubTB-specific config injection helper

    The EOS logprobs are needed for SubTB's sub-trajectory flow matching.
    """

    def inject_subtb_data(
        self,
        config: DictConfig | dict[str, Any],
        eos_logprobs: torch.Tensor,
    ) -> None:
        """Inject SubTB-specific data into config.

        This method injects both logZ (via parent class) and EOS logprobs
        needed for the SubTB loss computation.

        Args:
            config: Training config (will be modified in-place)
            eos_logprobs: EOS log probabilities tensor [B, T]
        """
        # first inject logZ (from parent)
        self.inject_logZ_into_config(config)

        # inject EOS logprobs tensor for SubTB flow matching
        try:
            object.__setattr__(config, "_eos_logprobs", eos_logprobs)
            logger.debug(f"Injected eos_logprobs shape={eos_logprobs.shape} into config")
        except (AttributeError, TypeError):
            # plain dict or frozen config - tensor injection not possible
            logger.warning(
                "Could not inject eos_logprobs tensor. SubTB will fall back to vanilla TB."
            )


# try to define SkyRL-based SubTBTrainer
try:
    from skyrl_train.trainer import RayPPOTrainer

    class SubTBTrainer(SubTBTrainerMixin, RayPPOTrainer):
        """Sub-Trajectory Balance trainer extending SkyRL's RayPPOTrainer.

        This trainer:
        1. Uses the registered modified_subtb loss + tb_identity estimator
        2. Manages logZ as a separate learnable parameter
        3. Injects logZ AND eos_logprobs into config before each loss computation

        Config requirements:
            trainer.algorithm.policy_loss_type: modified_subtb
            trainer.algorithm.advantage_estimator: tb_identity
            trainer.algorithm.subtb_lambda: 0.9  (decay factor)
        """

        def __init__(
            self,
            config: DictConfig,
            *args: Any,
            logZ_init: float | None = None,
            logZ_lr: float | None = None,
            **kwargs: Any,
        ) -> None:
            # extract params from config if not provided
            if logZ_init is None:
                logZ_init = config.get("logZ_init", 0.0)
            if logZ_lr is None:
                logZ_lr = config.get("logZ_lr", 1e-2)

            # initialize mixin first (sets up logZ)
            SubTBTrainerMixin.__init__(
                self,
                logZ_init=logZ_init,
                logZ_lr=logZ_lr,
            )
            # then initialize SkyRL trainer
            RayPPOTrainer.__init__(self, config, *args, **kwargs)

            # verify config
            self._verify_subtb_config(config)

        def _verify_subtb_config(self, config: DictConfig) -> None:
            """Verify config has correct SubTB settings."""
            algorithm = config.trainer.algorithm

            loss_type = algorithm.get("policy_loss_type", "regular")
            if loss_type != "modified_subtb":
                logger.warning(
                    f"SubTBTrainer expects policy_loss_type='modified_subtb', "
                    f"got '{loss_type}'. SubTB training may not work correctly."
                )

            adv_estimator = algorithm.get("advantage_estimator", "gae")
            if adv_estimator != "tb_identity":
                logger.warning(
                    f"SubTBTrainer expects advantage_estimator='tb_identity', "
                    f"got '{adv_estimator}'. Log rewards may not be passed correctly."
                )

            subtb_lambda = algorithm.get("subtb_lambda", None)
            if subtb_lambda is None:
                logger.info(
                    "subtb_lambda not set in config, using default 0.9. "
                    "Set algorithm.subtb_lambda to customize."
                )

        def train_critic_and_policy(self, data: Any) -> Any:
            """Train step with SubTB-specific handling.

            Injects logZ and EOS logprobs into config before training.
            """
            # get eos_logprobs from batch data if available
            eos_logprobs = getattr(data, "eos_logprobs", None)

            if eos_logprobs is not None:
                # full SubTB mode
                self.inject_subtb_data(self.cfg, eos_logprobs)
            else:
                # fallback to vanilla TB (just inject logZ)
                self.inject_logZ_into_config(self.cfg)
                logger.debug("No eos_logprobs in batch, using vanilla TB fallback")

            # run standard training step
            result = super().train_critic_and_policy(data)

            # perform logZ optimizer step
            tb_metrics = self.tb_optimizer_step()

            # add TB metrics to training output
            if hasattr(self, "all_metrics"):
                self.all_metrics.update(tb_metrics)

            return result

except ImportError:
    # SkyRL not available, provide a fallback class
    class SubTBTrainer(SubTBTrainerMixin):  # type: ignore[no-redef]
        """Standalone SubTB trainer when SkyRL is not available."""

        def __init__(
            self,
            *args: Any,
            logZ_init: float = 0.0,
            logZ_lr: float = 1e-2,
            **kwargs: Any,
        ) -> None:
            super().__init__(logZ_init=logZ_init, logZ_lr=logZ_lr)
            logger.warning(
                "SkyRL not available. SubTBTrainer provides logZ management only. "
                "Use SubTBTrainerMixin with your own trainer base class."
            )


__all__ = [
    "LogZModule",
    "TBTrainerMixin",
    "TBTrainer",
    "SubTBTrainerMixin",
    "SubTBTrainer",
    "SKYRL_AVAILABLE",
]
