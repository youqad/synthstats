#!/usr/bin/env python
"""SynthStats training CLI.

Usage:
    uv run python -m synthstats.cli.train
    uv run python -m synthstats.cli.train model=qwen3_4b trainer.batch_size=8
    uv run python -m synthstats.cli.train +wandb.enabled=true wandb.project=synthstats
"""

from __future__ import annotations

import logging
import signal
import sys
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def get_device(device_cfg: str) -> str:
    """Resolve device configuration to actual device string.

    Args:
        device_cfg: Device config value ("auto", "cpu", "cuda", "mps", etc.)

    Returns:
        Resolved device string.
    """
    if device_cfg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_cfg


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_policy(cfg: DictConfig, device: str) -> Any:
    """Instantiate the policy based on config.

    Args:
        cfg: Full Hydra config with model settings.
        device: Resolved device string.

    Returns:
        Policy instance (HFPolicy or MockPolicy).
    """
    model_cfg = cfg.model

    # check for mock policy (for testing)
    if getattr(model_cfg, "mock", False):
        from synthstats.policies.hf_policy import MockPolicy

        log.info("Using MockPolicy for testing")
        return MockPolicy(
            fixed_text=getattr(model_cfg, "fixed_text", '{"answer": "test"}'),
            fixed_token_ids=list(getattr(model_cfg, "fixed_token_ids", [1, 2, 3])),
            fixed_token_logprobs=list(
                getattr(model_cfg, "fixed_token_logprobs", [-0.5, -0.3, -0.2])
            ),
        )

    from synthstats.policies.hf_policy import HFPolicy

    log.info(f"Loading model: {model_cfg.name}")
    return HFPolicy(
        model_name=model_cfg.name,
        device=device,
        dtype=getattr(model_cfg, "dtype", "bfloat16"),
    )


def build_task(cfg: DictConfig) -> Any:
    """Instantiate the task based on config.

    Args:
        cfg: Full Hydra config with task settings.

    Returns:
        Task instance.
    """
    task_cfg = cfg.task

    if task_cfg.name == "toy":
        from synthstats.training.trainer import ToyTask

        log.info("Using ToyTask")
        return ToyTask()

    if task_cfg.name == "boxing":
        from synthstats.tasks.boxing.task import BoxingTask

        log.info(f"Using BoxingTask with env={task_cfg.env}")
        return BoxingTask(
            env_name=task_cfg.env,
            max_steps=getattr(task_cfg, "max_steps", 20),
        )

    raise ValueError(f"Unknown task: {task_cfg.name}")


def build_codec(cfg: DictConfig) -> Any:
    """Instantiate the codec based on config.

    Args:
        cfg: Full Hydra config with runtime settings.

    Returns:
        ActionCodec instance.
    """
    codec_name = cfg.runtime.codec

    if codec_name == "json":
        from synthstats.runtime.codecs import JSONToolCodec

        return JSONToolCodec()

    if codec_name == "xml":
        from synthstats.runtime.codecs import XMLToolCodec

        return XMLToolCodec()

    raise ValueError(f"Unknown codec: {codec_name}")


def build_judge(cfg: DictConfig) -> Any:
    """Instantiate the judge based on config.

    Args:
        cfg: Full Hydra config with judge settings.

    Returns:
        Judge instance.
    """
    judge_cfg = cfg.judge

    # build individual judges from config
    judges_with_weights: list[tuple[Any, float]] = []

    for judge_spec in judge_cfg.judges:
        judge_type = judge_spec["type"]
        weight = judge_spec.get("weight", 1.0)

        if judge_type == "likelihood":
            from synthstats.judges.likelihood import LikelihoodJudge

            judges_with_weights.append((LikelihoodJudge(), weight))
        elif judge_type == "formatting":
            from synthstats.judges.formatting import FormattingJudge

            judges_with_weights.append((FormattingJudge(), weight))
        else:
            raise ValueError(f"Unknown judge type: {judge_type}")

    from synthstats.judges.composite import CompositeJudge

    return CompositeJudge(judges_with_weights)


def build_trainer_config(cfg: DictConfig) -> Any:
    """Build TrainerConfig from Hydra config.

    Args:
        cfg: Full Hydra config with trainer settings.

    Returns:
        TrainerConfig instance.
    """
    from synthstats.training.trainer import TrainerConfig

    trainer_cfg = cfg.trainer

    return TrainerConfig(
        batch_size=getattr(trainer_cfg, "batch_size", 4),
        learning_rate=getattr(trainer_cfg, "learning_rate", 1e-4),
        logZ_learning_rate=getattr(trainer_cfg, "logZ_lr", 1e-2),
        max_episodes=getattr(trainer_cfg, "num_episodes", 1000),
        max_steps_per_episode=getattr(cfg.task, "max_steps", 10),
        max_grad_norm=getattr(trainer_cfg, "max_grad_norm", 1.0),
        seed=cfg.seed,
    )


def setup_wandb(cfg: DictConfig) -> bool:
    """Initialize WandB if enabled.

    Args:
        cfg: Full Hydra config.

    Returns:
        True if WandB was initialized, False otherwise.
    """
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return False

    try:
        import wandb

        config = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            project=wandb_cfg.get("project", "synthstats"),
            entity=wandb_cfg.get("entity"),
            config=cast(dict[str, Any], config),
        )
        run = wandb.run
        if run is not None:
            log.info(f"WandB initialized: {run.name}")
        else:
            log.info("WandB initialized")
        return True
    except ImportError:
        log.warning("wandb not installed, skipping WandB logging")
        return False
    except Exception as e:
        log.warning(f"Failed to initialize WandB: {e}")
        return False


def create_log_callback(use_wandb: bool):
    """Create a logging callback for training metrics.

    Args:
        use_wandb: Whether to log to WandB.

    Returns:
        Callback function for trainer.
    """

    def callback(metrics, step: int) -> None:
        log.info(
            f"Step {step}: loss={metrics.loss:.4f}, "
            f"logZ={metrics.logZ:.4f}, "
            f"avg_reward={metrics.avg_reward:.4f}"
        )

        if use_wandb:
            try:
                import wandb

                wandb.log(
                    {
                        "loss": metrics.loss,
                        "logZ": metrics.logZ,
                        "avg_reward": metrics.avg_reward,
                        "replay_fraction": metrics.replay_fraction,
                    },
                    step=step,
                )
            except Exception:
                pass

    return callback


def save_checkpoint(
    trainer: Any,
    output_dir: Path,
    step: int,
    is_final: bool = False,
) -> Path:
    """Save a training checkpoint.

    Args:
        trainer: Trainer instance.
        output_dir: Directory to save checkpoint.
        step: Current training step.
        is_final: Whether this is the final checkpoint.

    Returns:
        Path to saved checkpoint.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_final:
        ckpt_path = output_dir / "checkpoint_final.pt"
    else:
        ckpt_path = output_dir / f"checkpoint_{step:06d}.pt"

    checkpoint = {
        "step": step,
        "logZ": trainer.logZ,
        "optimizer_state_dict": trainer.optimizer.state_dict(),
    }

    torch.save(checkpoint, ckpt_path)
    log.info(f"Saved checkpoint: {ckpt_path}")
    return ckpt_path


def load_checkpoint(trainer: Any, checkpoint_path: Path) -> int:
    """Load a training checkpoint.

    Args:
        trainer: Trainer instance to restore.
        checkpoint_path: Path to checkpoint file.

    Returns:
        Step number from checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # restore logZ
    trainer._logZ.data.fill_(checkpoint["logZ"])

    # restore optimizer
    trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    step = checkpoint["step"]
    log.info(f"Loaded checkpoint from step {step}: {checkpoint_path}")
    return step


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    """Main training entry point.

    Args:
        cfg: Hydra configuration.

    Returns:
        Final loss value (for hyperparameter optimization).
    """
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # resolve device
    device = get_device(cfg.device)
    log.info(f"Using device: {device}")

    # set seed
    set_seed(cfg.seed)
    log.info(f"Random seed: {cfg.seed}")

    # build components
    policy = build_policy(cfg, device)
    task = build_task(cfg)
    codec = build_codec(cfg)
    judge = build_judge(cfg)
    trainer_config = build_trainer_config(cfg)

    # setup wandb
    use_wandb = setup_wandb(cfg)

    # create trainer
    from synthstats.training.trainer import Trainer

    trainer = Trainer(
        config=trainer_config,
        policy=policy,
        task=task,
        codec=codec,
        judge=judge,
        device=device,
        log_callback=create_log_callback(use_wandb),
    )

    # resolve output directory
    output_dir = Path(cfg.output_dir)
    log.info(f"Output directory: {output_dir}")

    # check for resume checkpoint
    start_step = 0
    resume_path = cfg.get("resume_from")
    if resume_path:
        resume_path = Path(resume_path)
        if resume_path.exists():
            start_step = load_checkpoint(trainer, resume_path)

    # setup graceful shutdown
    shutdown_requested = False

    def handle_signal(signum, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            log.warning("Force shutdown requested")
            sys.exit(1)
        log.info("Shutdown requested, finishing current step...")
        shutdown_requested = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # training loop
    num_steps = trainer_config.max_episodes
    final_loss = 0.0

    try:
        for step in range(start_step, num_steps):
            if shutdown_requested:
                log.info("Shutting down gracefully...")
                save_checkpoint(trainer, output_dir, step, is_final=False)
                break

            metrics = trainer.train_step()
            final_loss = metrics.loss

            # periodic checkpoint
            checkpoint_interval = cfg.get("checkpoint_interval", 100)
            if (step + 1) % checkpoint_interval == 0:
                save_checkpoint(trainer, output_dir, step + 1)

        else:
            # training completed normally
            log.info("Training completed!")
            save_checkpoint(trainer, output_dir, num_steps, is_final=True)

    except Exception as e:
        log.error(f"Training failed: {e}")
        raise

    finally:
        # cleanup wandb
        if use_wandb:
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass

    # final evaluation
    log.info("Running final evaluation...")
    eval_results = trainer.evaluate(num_episodes=10)
    log.info(f"Final evaluation: {eval_results}")

    return final_loss


if __name__ == "__main__":
    main()
