#!/usr/bin/env python
"""Tinker API training with SubTB loss on BoxingGym environments.

Usage:
    TINKER_API_KEY=... uv run python scripts/train_tinker.py env=dugongs
    uv run python scripts/train_tinker.py env=peregrines trainer.config.model=Qwen/Qwen3-4B
    uv run python scripts/train_tinker.py env=dugongs +mock=true

Environment Variables:
    TINKER_API_KEY: Tinker API key (get from https://thinkingmachines.ai/tinker/)
    WANDB_PROJECT: W&B project name (optional)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_trajectory(
    env_name: str,
    reward: float = 0.5,
) -> Any:
    from synthstats.core.types import Message, Reward, Trajectory

    messages = [
        Message(role="system", content=f"You are solving the {env_name} environment."),
        Message(role="user", content="Here is the data: [1, 2, 3]"),
        Message(
            role="assistant",
            content='{"type": "submit_program", "payload": "# mock program"}',
        ),
    ]

    return Trajectory(
        messages=messages,
        token_ids=[[1, 2, 3, 4, 5]],
        token_logprobs=[[-0.1, -0.2, -0.15, -0.1, -0.2]],
        loss_mask=[[True, True, True, True, True]],
        reward=Reward(total=reward, components={"likelihood": reward}, info={}),
    )


def run_mock_training(cfg: DictConfig) -> dict[str, float]:
    from synthstats.integrations.tinker import (
        MockTinkerTrainingClient,
        TinkerConfig,
        TinkerTrainer,
        trajectories_to_tinker_batch,
    )

    logger.info("Mock mode (no API key needed)")

    tinker_cfg = TinkerConfig(
        model="mock-model",
        api_key="mock-key",
        lora_rank=cfg.trainer.config.lora_rank,
        learning_rate=cfg.trainer.config.learning_rate,
    )
    trainer = TinkerTrainer(config=tinker_cfg, logZ_init=cfg.trainer.logZ_init)

    trainer._training_client = MockTinkerTrainingClient()
    trainer._service_client = "mock"

    num_episodes = cfg.trainer.num_episodes
    batch_size = cfg.trainer.batch_size
    env_name = cfg.env.name

    metrics: dict[str, list[float]] = {
        "loss": [],
        "logZ": [],
        "reward": [],
    }

    logger.info(f"Training on {env_name} for {num_episodes} episodes (batch_size={batch_size})")

    for ep in range(num_episodes):
        trajectories = [
            create_mock_trajectory(env_name, reward=0.1 + 0.5 * (ep / num_episodes))
            for _ in range(batch_size)
        ]

        batch = trajectories_to_tinker_batch(trajectories, device="cpu")
        step_metrics = trainer.train_step(batch)

        metrics["loss"].append(step_metrics.get("subtb_loss", step_metrics.get("loss", 0.0)))
        metrics["logZ"].append(step_metrics.get("logZ", trainer.logZ.item()))
        metrics["reward"].append(batch["log_reward"].mean().item())

        if (ep + 1) % max(1, num_episodes // 10) == 0:
            logger.info(
                f"Episode {ep + 1}/{num_episodes}: "
                f"loss={metrics['loss'][-1]:.4f}, "
                f"logZ={metrics['logZ'][-1]:.4f}, "
                f"mean_log_R={metrics['reward'][-1]:.4f}"
            )

    mean_loss = sum(metrics["loss"]) / len(metrics["loss"]) if metrics["loss"] else 0.0
    final_logZ = metrics["logZ"][-1] if metrics["logZ"] else 0.0
    mean_log_reward = (
        sum(metrics["reward"]) / len(metrics["reward"]) if metrics["reward"] else 0.0
    )
    return {
        "mean_loss": mean_loss,
        "final_logZ": final_logZ,
        "mean_log_reward": mean_log_reward,
    }


def run_real_training(cfg: DictConfig) -> dict[str, float]:
    from synthstats.integrations.tinker import (
        TinkerConfig,
        TinkerPolicy,
        TinkerTrainer,
        is_tinker_available,
        trajectories_to_tinker_batch,
    )

    if not is_tinker_available():
        raise RuntimeError(
            "Tinker SDK not installed. Install with: pip install tinker\n"
            "Or run in mock mode: uv run python scripts/train_tinker.py +mock=true"
        )

    logger.info("Tinker API training")

    tinker_cfg = TinkerConfig(
        model=cfg.trainer.config.model,
        api_key=cfg.trainer.config.api_key,
        max_tokens=cfg.trainer.config.max_tokens,
        temperature=cfg.trainer.config.temperature,
        lora_rank=cfg.trainer.config.lora_rank,
        learning_rate=cfg.trainer.config.learning_rate,
    )

    policy = TinkerPolicy(config=tinker_cfg)
    trainer = TinkerTrainer(config=tinker_cfg, logZ_init=cfg.trainer.logZ_init)

    logger.info(f"Model: {tinker_cfg.model}")
    logger.info(f"LoRA rank: {tinker_cfg.lora_rank}")
    logger.info(f"Learning rate: {tinker_cfg.learning_rate}")

    from synthstats.envs.boxing_env import BoxingEnv, BoxingEnvConfig
    from synthstats.executors.pymc_sandbox import PyMCExecutor
    from synthstats.judges.likelihood import LikelihoodJudge
    from synthstats.tasks.boxing import BoxingTask
    from synthstats.tasks.boxing.codecs import BoxingCodec

    task = BoxingTask(env_name=cfg.env.name)
    codec = BoxingCodec()
    executors = {"pymc": PyMCExecutor()}
    judge = LikelihoodJudge()
    env_config = BoxingEnvConfig(max_turns=cfg.env.max_steps)

    env = BoxingEnv(
        task=task,
        codec=codec,
        executors=executors,
        judge=judge,
        config=env_config,
    )

    logger.info(f"Environment: {cfg.env.name}")

    num_episodes = cfg.trainer.num_episodes
    batch_size = cfg.trainer.batch_size
    checkpoint_interval = cfg.get("checkpoint_interval", 50)

    if cfg.wandb.enabled:
        import wandb
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    metrics: dict[str, list[float]] = {
        "loss": [],
        "logZ": [],
        "reward": [],
    }

    logger.info(f"Training: {num_episodes} episodes, batch_size={batch_size}")

    for ep in range(num_episodes):
        trajectories = []
        for _ in range(batch_size):
            chat_history, _ = env.init()
            action, logp, entropy = policy(chat_history)
            action_str = policy._render_action(action)
            result = env.step(action_str)

            from synthstats.core.types import Message, Reward, Trajectory

            messages = [Message(role=m["role"], content=m["content"]) for m in env.chat_history]

            traj = Trajectory(
                messages=messages,
                token_ids=[[]],
                token_logprobs=[[logp]],
                loss_mask=[[True]],
                reward=Reward(
                    total=result["reward"],
                    components=result.get("reward_components", {}),
                    info=result.get("info", {}),
                ),
            )
            trajectories.append(traj)

        batch = trajectories_to_tinker_batch(trajectories, device="cpu", multi_turn=True)
        step_metrics = trainer.train_step(batch)

        loss = step_metrics.get("subtb_loss", step_metrics.get("loss", 0.0))
        logZ = step_metrics.get("logZ", trainer.logZ.item())
        mean_reward = batch["log_reward"].mean().item()

        metrics["loss"].append(loss)
        metrics["logZ"].append(logZ)
        metrics["reward"].append(mean_reward)

        if cfg.wandb.enabled:
            wandb.log({
                "step": ep,
                "loss": loss,
                "logZ": logZ,
                "mean_log_reward": mean_reward,
            })

        if checkpoint_interval > 0 and (ep + 1) % checkpoint_interval == 0:
            ckpt_path = Path(cfg.output_dir) / f"checkpoint_ep{ep + 1}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(str(ckpt_path))
            logger.info(f"Saved checkpoint: {ckpt_path}")

        if (ep + 1) % max(1, num_episodes // 10) == 0:
            logger.info(
                f"Episode {ep + 1}/{num_episodes}: "
                f"loss={loss:.4f}, logZ={logZ:.4f}, mean_log_R={mean_reward:.4f}"
            )

    if cfg.wandb.enabled:
        wandb.finish()

    mean_loss = sum(metrics["loss"]) / len(metrics["loss"]) if metrics["loss"] else 0.0
    final_logZ = metrics["logZ"][-1] if metrics["logZ"] else 0.0
    mean_log_reward = (
        sum(metrics["reward"]) / len(metrics["reward"]) if metrics["reward"] else 0.0
    )
    return {
        "mean_loss": mean_loss,
        "final_logZ": final_logZ,
        "mean_log_reward": mean_log_reward,
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)

    trainer_target = cfg.get("trainer", {}).get("_target_", "")
    if "TinkerTrainer" not in trainer_target:
        logger.info("Using trainer=tinker config")
        tinker_trainer = OmegaConf.load(
            Path(__file__).parent.parent / "configs/trainer/tinker.yaml"
        )
        if "trainer" in cfg:
            cfg.trainer = OmegaConf.merge(cfg.trainer, tinker_trainer)
        else:
            cfg.trainer = tinker_trainer

    logger.info("=" * 60)
    logger.info("SynthStats Tinker Training")
    logger.info("=" * 60)

    logger.info(f"Environment: {cfg.get('env', {}).get('name', 'unknown')}")
    logger.info(f"Model: {cfg.trainer.config.model}")
    logger.info(f"Episodes: {cfg.trainer.num_episodes}")
    logger.info(f"Batch size: {cfg.trainer.batch_size}")

    mock_mode = cfg.get("mock", False)

    if mock_mode:
        results = run_mock_training(cfg)
    else:
        api_key = cfg.trainer.config.api_key or os.environ.get("TINKER_API_KEY")
        if not api_key:
            logger.warning(
                "No TINKER_API_KEY found. Running in mock mode.\n"
                "Set TINKER_API_KEY env var or run with +mock=true"
            )
            results = run_mock_training(cfg)
        else:
            results = run_real_training(cfg)

    logger.info("=" * 60)
    logger.info("Training Complete")
    logger.info(f"Results: {results}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
