#!/usr/bin/env python
"""Native SkyRL training script with Trajectory Balance loss.

This script uses SkyRL's native training infrastructure with our registered
trajectory_balance loss. For custom training workflows, use SkyRL's entrypoint
directly with appropriate config.

Usage:
    # smoke test with mock policy
    uv run python scripts/train.py trainer=skyrl_tb +env=dugongs +dry_run=true

    # real training (requires GPU and SkyRL fully installed)
    python -m skyrl_train.entrypoints.main \\
        trainer.algorithm.policy_loss_type=trajectory_balance \\
        trainer.algorithm.advantage_estimator=tb_identity \\
        env=boxing_dugongs

Environment Variables:
    WANDB_PROJECT: W&B project name (optional)
    CUDA_VISIBLE_DEVICES: GPU selection

Note: Full SkyRL training requires:
- Linux with CUDA
- vLLM for inference
- Sufficient GPU memory for the model

For local development/testing without full SkyRL infrastructure,
this script provides a simplified training loop.
"""

from __future__ import annotations

import logging

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from synthstats.envs.boxing_env import BoxingEnv, BoxingEnvConfig

# register TB loss on import (SkyRL registry-based)
from synthstats.training.losses import trajectory_balance as _trajectory_balance  # noqa: F401
from synthstats.training.tb_trainer import SKYRL_AVAILABLE, LogZModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_tb_loss_registered() -> bool:
    """Verify that trajectory_balance loss is registered with SkyRL."""
    try:
        from skyrl_train.utils.ppo_utils import PolicyLossRegistry

        available = PolicyLossRegistry.list_available()
        if "trajectory_balance" in available:
            logger.info("trajectory_balance loss registered with SkyRL")
            return True
        else:
            logger.error(f"trajectory_balance not in registry: {available}")
            return False
    except ImportError:
        logger.warning("SkyRL not available - running in local mode")
        return False


def create_boxing_env(cfg: DictConfig) -> BoxingEnv:
    """Create BoxingEnv from Hydra config."""
    from synthstats.executors.pymc_sandbox import PyMCExecutor
    from synthstats.judges.likelihood import LikelihoodJudge
    from synthstats.tasks.boxing import BoxingTask
    from synthstats.tasks.boxing.codecs import BoxingCodec

    task_name = cfg.get("env", {}).get("name", "dugongs")
    task = BoxingTask(env_name=task_name)

    codec = BoxingCodec()
    executors = {"pymc": PyMCExecutor()}
    judge = LikelihoodJudge()

    env_config = BoxingEnvConfig(
        max_turns=cfg.get("env", {}).get("max_turns", 20),
        reward_floor=cfg.get("trainer", {}).get("reward_floor", 1e-4),
    )

    return BoxingEnv(
        task=task,
        codec=codec,
        executors=executors,
        judge=judge,
        config=env_config,
    )


def run_local_training(cfg: DictConfig) -> dict[str, float]:
    """Run simplified local training loop for development/testing.

    This is NOT the full SkyRL training - it's a simplified loop for
    testing the TB loss and environment without vLLM/Ray infrastructure.
    """
    from synthstats.training.losses.tb_loss import subtb_loss

    logger.info("Running local training mode (simplified, no vLLM/Ray)")

    env = create_boxing_env(cfg)
    logger.info(f"Created env: {env.task.name}")

    logZ_init = cfg.get("trainer", {}).get("logZ_init", 0.0)
    logZ_lr = cfg.get("trainer", {}).get("logZ_lr", 0.01)
    logZ_module = LogZModule(init_value=logZ_init)
    logZ_optimizer = torch.optim.Adam(logZ_module.parameters(), lr=logZ_lr)

    num_episodes = cfg.get("trainer", {}).get("num_episodes", 10)

    metrics: dict[str, list[float]] = {
        "loss": [],
        "logZ": [],
        "reward": [],
    }

    for ep in range(num_episodes):
        # collect episode (simplified - no actual generation)
        obs, info = env.init()
        logger.debug(f"Episode {ep+1}: initialized env")

        # simulate step (in real training, this would be policy generation)
        # For now, just use a dummy action to test the flow
        dummy_action = '{"type": "submit_program", "payload": "# dummy program"}'
        result = env.step(dummy_action)

        reward = result["reward"]
        metrics["reward"].append(reward)

        # compute dummy loss (for testing gradient flow)
        log_reward = torch.log(torch.tensor(max(reward, 1e-4)))
        dummy_log_probs = torch.randn(1, 10) * 0.1  # dummy

        loss = subtb_loss(
            log_probs=dummy_log_probs,
            loss_mask=torch.ones_like(dummy_log_probs),
            log_rewards=log_reward.unsqueeze(0),
            logZ=logZ_module.logZ,
        )

        loss.backward()
        logZ_optimizer.step()
        logZ_optimizer.zero_grad()

        metrics["loss"].append(loss.item())
        metrics["logZ"].append(logZ_module.logZ.item())

        if (ep + 1) % max(1, num_episodes // 10) == 0:
            logger.info(
                f"Episode {ep+1}/{num_episodes}: "
                f"loss={loss.item():.4f}, logZ={logZ_module.logZ.item():.4f}, "
                f"reward={reward:.4f}"
            )

    return {
        "mean_loss": sum(metrics["loss"]) / len(metrics["loss"]),
        "final_logZ": metrics["logZ"][-1],
        "mean_reward": sum(metrics["reward"]) / len(metrics["reward"]),
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training entrypoint."""
    logger.info("=" * 60)
    logger.info("SynthStats Native SkyRL Training")
    logger.info("=" * 60)

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    tb_registered = verify_tb_loss_registered()

    if cfg.get("dry_run", False):
        logger.info("Dry run mode - skipping actual training")
        return

    if SKYRL_AVAILABLE and tb_registered:
        logger.info("SkyRL available - use skyrl_train.entrypoints.main for full training")
        logger.info("Example:")
        logger.info("  python -m skyrl_train.entrypoints.main \\")
        logger.info("      trainer.algorithm.policy_loss_type=trajectory_balance \\")
        logger.info("      trainer.algorithm.advantage_estimator=tb_identity \\")
        logger.info("      env=boxing_dugongs")
        logger.info("")
        logger.info("Running local training mode for testing...")

    results = run_local_training(cfg)

    logger.info("=" * 60)
    logger.info("Training Complete")
    logger.info(f"Results: {results}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
