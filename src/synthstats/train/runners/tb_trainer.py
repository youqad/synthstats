"""TB and SubTB trainers for SkyRL.

Extend RayPPOTrainer with learned logZ and config injection.
logZ (and eos_logprobs for SubTB) injected via config before each loss step.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def _resolve_logz_params(
    config: DictConfig,
    *,
    logZ_init: float | None,
    logZ_lr: float | None,
) -> tuple[float, float]:
    if logZ_init is None:
        logZ_init = config.get("logZ_init", 0.0)
    if logZ_lr is None:
        logZ_lr = config.get("logZ_lr", 1e-2)
    return float(logZ_init), float(logZ_lr)


def _warn_algorithm_mismatch(
    algorithm: Any,
    *,
    trainer_name: str,
    expected_loss: str,
    expected_advantage: str,
) -> None:
    loss_type = algorithm.get("policy_loss_type", "regular")
    if loss_type != expected_loss:
        logger.warning(
            f"{trainer_name} expects policy_loss_type='{expected_loss}', "
            f"got '{loss_type}'. {trainer_name} training may not work correctly."
        )

    adv_estimator = algorithm.get("advantage_estimator", "gae")
    if adv_estimator != expected_advantage:
        logger.warning(
            f"{trainer_name} expects advantage_estimator='{expected_advantage}', "
            f"got '{adv_estimator}'. Log rewards may not be passed correctly."
        )


class LogZModule(nn.Module):
    """Learnable log partition function for GFlowNets."""

    def __init__(self, init_value: float = 0.0) -> None:
        super().__init__()
        self.logZ = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return self.logZ


class TBTrainerMixin:
    """Mixin for logZ parameter management, config injection, and optimization."""

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
        return self.logZ_module.logZ

    def inject_logZ_into_config(self, config: DictConfig | dict[str, Any]) -> None:
        """Inject current logZ into config (tensor for gradient flow, float for compat)."""
        logZ_val = self.logZ.item()

        # bypass OmegaConf's type validation to preserve computational graph
        try:
            object.__setattr__(config, "_logZ_tensor", self.logZ)
        except (AttributeError, TypeError):
            logger.debug("Could not inject logZ tensor - using float fallback")

        if hasattr(config, "algorithm"):
            config.algorithm.logZ = logZ_val
        elif isinstance(config, dict):
            if "algorithm" in config:
                config["algorithm"]["logZ"] = logZ_val
            else:
                config["logZ"] = logZ_val
        else:
            config.logZ = logZ_val

        logger.debug(f"Injected logZ={logZ_val:.4f} into config (with tensor)")

    def tb_optimizer_step(self, loss: torch.Tensor | None = None) -> dict[str, float]:
        """Perform logZ optimizer step. Call after main optimizer step."""
        if loss is not None and loss.requires_grad:
            self.logZ_optimizer.zero_grad()
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
        """TB trainer extending RayPPOTrainer with learned logZ."""

        def __init__(
            self,
            config: DictConfig,
            *args: Any,
            logZ_init: float | None = None,
            logZ_lr: float | None = None,
            **kwargs: Any,
        ) -> None:
            logZ_init, logZ_lr = _resolve_logz_params(
                config,
                logZ_init=logZ_init,
                logZ_lr=logZ_lr,
            )

            TBTrainerMixin.__init__(
                self,
                logZ_init=logZ_init,
                logZ_lr=logZ_lr,
            )
            RayPPOTrainer.__init__(self, config, *args, **kwargs)
            self._verify_tb_config(config)

        def _verify_tb_config(self, config: DictConfig) -> None:
            algorithm = config.trainer.algorithm
            _warn_algorithm_mismatch(
                algorithm,
                trainer_name="TBTrainer",
                expected_loss="trajectory_balance",
                expected_advantage="tb_identity",
            )

        def train_critic_and_policy(self, data: Any) -> Any:
            self.inject_logZ_into_config(self.cfg)
            result = super().train_critic_and_policy(data)

            # logZ gets gradient from being in the computation graph
            tb_metrics = self.tb_optimizer_step()

            if hasattr(self, "all_metrics"):
                self.all_metrics.update(tb_metrics)

            return result

    SKYRL_AVAILABLE = True

except ImportError:
    class TBTrainer(TBTrainerMixin):  # type: ignore[no-redef]
        """Standalone TB trainer (logZ management only, no SkyRL)."""

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


class SubTBTrainerMixin(TBTrainerMixin):
    """Extends TBTrainerMixin with EOS logprobs injection for SubTB."""

    def inject_subtb_data(
        self,
        config: DictConfig | dict[str, Any],
        eos_logprobs: torch.Tensor,
    ) -> None:
        """Inject logZ and EOS logprobs into config for SubTB loss."""
        self.inject_logZ_into_config(config)

        try:
            object.__setattr__(config, "_eos_logprobs", eos_logprobs)
            logger.debug(f"Injected eos_logprobs shape={eos_logprobs.shape} into config")
        except (AttributeError, TypeError):
            logger.warning(
                "Could not inject eos_logprobs tensor. SubTB will fall back to vanilla TB."
            )


# try to define SkyRL-based SubTBTrainer
try:
    from skyrl_train.trainer import RayPPOTrainer

    class SubTBTrainer(SubTBTrainerMixin, RayPPOTrainer):
        """SubTB trainer extending RayPPOTrainer with learned logZ and EOS injection."""

        def __init__(
            self,
            config: DictConfig,
            *args: Any,
            logZ_init: float | None = None,
            logZ_lr: float | None = None,
            **kwargs: Any,
        ) -> None:
            logZ_init, logZ_lr = _resolve_logz_params(
                config,
                logZ_init=logZ_init,
                logZ_lr=logZ_lr,
            )

            SubTBTrainerMixin.__init__(
                self,
                logZ_init=logZ_init,
                logZ_lr=logZ_lr,
            )
            RayPPOTrainer.__init__(self, config, *args, **kwargs)
            self._verify_subtb_config(config)

        def _verify_subtb_config(self, config: DictConfig) -> None:
            algorithm = config.trainer.algorithm
            _warn_algorithm_mismatch(
                algorithm,
                trainer_name="SubTBTrainer",
                expected_loss="modified_subtb",
                expected_advantage="tb_identity",
            )

            subtb_lambda = algorithm.get("subtb_lambda", None)
            if subtb_lambda is None:
                logger.info(
                    "subtb_lambda not set in config, using default 0.9. "
                    "Set algorithm.subtb_lambda to customize."
                )

        def train_critic_and_policy(self, data: Any) -> Any:
            eos_logprobs = getattr(data, "eos_logprobs", None)

            if eos_logprobs is not None:
                self.inject_subtb_data(self.cfg, eos_logprobs)
            else:
                self.inject_logZ_into_config(self.cfg)
                logger.debug("No eos_logprobs in batch, using vanilla TB fallback")

            result = super().train_critic_and_policy(data)
            tb_metrics = self.tb_optimizer_step()

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
