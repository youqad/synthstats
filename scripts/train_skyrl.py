#!/usr/bin/env python
"""SkyRL-integrated training script.

Canonical entrypoint for training with the SkyRL integration layer.

Usage:
    uv run python scripts/train_skyrl.py
    uv run python scripts/train_skyrl.py model=qwen3_0.6b
    uv run python scripts/train_skyrl.py +trainer.batch_size=8 +wandb.enabled=true

ARC/SLURM usage:
    sbatch scripts/arc/train_0.6b.slurm
    # Supports automatic resume from checkpoint via resume_from config
"""

import logging
import random
import signal
import sys
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from synthstats.envs.boxing_env import BoxingEnv, BoxingEnvConfig
from synthstats.envs.skyrl_text_env import SynthStatsTextEnv
from synthstats.policies.hf_policy import HFPolicy, MockHFPolicy
from synthstats.trainers.skyrl_subtb import SkyRLSubTBTrainer, SubTBConfig
from synthstats.training.train_loop import TrainingConfig, TrainingLoop

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_requested = False


def handle_sigterm(signum, frame):
    """Handle SIGTERM for graceful SLURM preemption."""
    global shutdown_requested
    logger.info(f"Received signal {signum}, requesting graceful shutdown...")
    shutdown_requested = True


def save_checkpoint(
    trainer: Any,
    policy: Any,
    step: int,
    output_dir: str | Path,
    metrics: dict | None = None,
) -> Path:
    """Save training checkpoint.

    Args:
        trainer: SkyRL trainer or TinkerTrainer with logZ parameter
        policy: HFPolicy or TinkerPolicy with model state
        step: Current training step
        output_dir: Directory to save checkpoint
        metrics: Optional metrics to include

    Returns:
        Path to saved checkpoint
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / f"checkpoint_{step:06d}.pt"

    # check if trainer has its own save_checkpoint method (TinkerTrainer)
    if hasattr(trainer, "save_checkpoint"):
        trainer.save_checkpoint(str(checkpoint_path))
        # append additional metadata
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        checkpoint["step"] = step
        checkpoint["metrics"] = metrics or {}
        if hasattr(trainer, "optimizer") and trainer.optimizer:
            checkpoint["optimizer_state"] = trainer.optimizer.state_dict()
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    # default checkpoint format (SkyRLSubTBTrainer)
    checkpoint = {
        "step": step,
        "logZ": trainer.logZ.item(),
        "optimizer_state": trainer.optimizer.state_dict() if hasattr(trainer, "optimizer") and trainer.optimizer else None,
        "metrics": metrics or {},
    }

    # save policy model if it has state
    if hasattr(policy, "model") and hasattr(policy.model, "state_dict"):
        checkpoint["model_state"] = policy.model.state_dict()

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str | Path,
    trainer: Any,
    policy: Any,
) -> int:
    """Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        trainer: SkyRL trainer or TinkerTrainer to restore
        policy: HFPolicy or TinkerPolicy to restore

    Returns:
        Step number to resume from
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # check if trainer has its own load_checkpoint method (TinkerTrainer)
    if hasattr(trainer, "load_checkpoint"):
        checkpoint = trainer.load_checkpoint(str(checkpoint_path), strict=False)
        # restore optimizer state if available
        if hasattr(trainer, "optimizer") and trainer.optimizer and checkpoint.get("optimizer_state"):
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])
        step = checkpoint.get("step", 0)
        logger.info(f"Resumed TinkerTrainer from step {step}")
        return step

    # default load (SkyRLSubTBTrainer)
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    # restore logZ
    with torch.no_grad():
        trainer.logZ.fill_(checkpoint["logZ"])

    # restore optimizer
    if hasattr(trainer, "optimizer") and trainer.optimizer and checkpoint.get("optimizer_state"):
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])

    # restore model if available
    if checkpoint.get("model_state") and hasattr(policy, "model"):
        policy.model.load_state_dict(checkpoint["model_state"])

    step = checkpoint["step"]
    logger.info(f"Resumed from step {step}, logZ={checkpoint['logZ']:.4f}")

    return step



def resolve_device(device_str: str) -> str:
    """Resolve 'auto' to actual device."""
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def create_policy(cfg: DictConfig, device: str) -> Any:
    """Create policy from config."""
    # hydra _target_ takes precedence
    if "_target_" in cfg.model:
        logger.info("Using Hydra instantiation for policy")
        try:
            from hydra.utils import instantiate
            return instantiate(cfg.model)
        except Exception as e:
            logger.warning(f"Hydra instantiation failed: {e}, falling back to manual")
            from synthstats.integrations.tinker_adapter import TinkerConfig, TinkerPolicy
            model_cfg = cfg.model.get("config", {})
            tinker_config = TinkerConfig(
                model=model_cfg.get("model", "Qwen/Qwen3-4B"),
                api_key=model_cfg.get("api_key"),
                max_tokens=model_cfg.get("max_tokens", 256),
                temperature=model_cfg.get("temperature", 0.7),
            )
            return TinkerPolicy(config=tinker_config)

    # simple name-based config
    model_name = cfg.model.name

    if model_name == "mock":
        logger.info("Using MockHFPolicy for testing")
        return MockHFPolicy(device=device)

    logger.info(f"Loading HFPolicy: {model_name}")
    return HFPolicy(
        model_name=model_name,
        device=device,
        require_grad_logp=cfg.model.get("require_grad_logp", False),
    )


def create_trainer(cfg: DictConfig, device: str, policy: Any | None = None) -> Any:
    """Create trainer from config."""
    if "_target_" in cfg.trainer:
        logger.info("Using Hydra instantiation for trainer")
        from synthstats.integrations.tinker_adapter import TinkerConfig, TinkerTrainer
        from hydra.utils import instantiate

        config_obj = cfg.trainer.get("config", {})
        if "_target_" in config_obj:
            tinker_config = instantiate(config_obj)
        else:
            tinker_config = TinkerConfig(
                model=config_obj.get("model", "Qwen/Qwen3-4B"),
                api_key=config_obj.get("api_key"),
                lora_rank=config_obj.get("lora_rank", 32),
                learning_rate=config_obj.get("learning_rate", 1e-5),
                max_tokens=config_obj.get("max_tokens", 256),
                temperature=config_obj.get("temperature", 0.7),
            )

        trainer = TinkerTrainer(
            config=tinker_config,
            logZ_init=cfg.trainer.get("logZ_init", 0.0),
        )

        # local optimizer for logZ updates
        trainer.optimizer = torch.optim.Adam(
            [{"params": trainer.parameters(), "lr": cfg.trainer.get("logZ_lr", 1e-3)}]
        )
        return trainer

    # default: SkyRLSubTBTrainer
    subtb_config = SubTBConfig(
        logZ_init=cfg.trainer.get("logZ_init", 0.0),
        use_ref_policy=cfg.trainer.get("use_ref_policy", False),
        ref_weight=cfg.trainer.get("ref_weight", 1.0),
        normalize_by_length=cfg.trainer.get("normalize_by_length", False),
        allow_mismatched_tokenizer=cfg.trainer.get(
            "allow_mismatched_tokenizer", False
        ),
    )
    trainer = SkyRLSubTBTrainer(
        config=subtb_config,
        device=device,
    )

    # setup optimizer
    params = [{"params": [trainer.logZ], "lr": cfg.trainer.get("logZ_lr", 0.1)}]

    if policy is not None and hasattr(policy, "parameters"):
        try:
            policy_params = [p for p in policy.parameters() if p.requires_grad]
        except Exception:
            policy_params = []
        if policy_params:
            params.insert(
                0,
                {
                    "params": policy_params,
                    "lr": cfg.trainer.get("learning_rate", 1e-5),
                },
            )

    trainer.optimizer = torch.optim.Adam(params)

    return trainer


def create_ref_policy(cfg: DictConfig, device: str) -> Any | None:
    """Create optional reference policy for ref-policy correction."""
    if not cfg.trainer.get("use_ref_policy", False):
        return None

    ref_name = cfg.trainer.get("ref_model_name")
    if ref_name is None:
        raise ValueError(
            "use_ref_policy=True requires trainer.ref_model_name to be set"
        )
    if ref_name == "mock":
        logger.info("Using MockHFPolicy as reference policy")
        return MockHFPolicy(device=device)

    logger.info(f"Loading reference HFPolicy: {ref_name}")
    return HFPolicy(
        model_name=ref_name,
        device=device,
    )


def build_task(cfg: DictConfig) -> Any:
    """Instantiate the task based on config."""
    task_cfg = cfg.task

    if task_cfg.name == "toy":
        from synthstats.training.trainer import ToyTask

        logger.info("Using ToyTask")
        return ToyTask()

    if task_cfg.name == "boxing":
        from synthstats.tasks.boxing.task import BoxingTask

        logger.info(f"Using BoxingTask with env={task_cfg.env}")
        return BoxingTask(
            env_name=task_cfg.env,
            max_steps=getattr(task_cfg, "max_steps", 20),
        )

    raise ValueError(f"Unknown task: {task_cfg.name}")


def build_codec(cfg: DictConfig) -> Any:
    """Instantiate the codec based on config."""
    codec_name = cfg.runtime.codec

    if codec_name == "boxing":
        from synthstats.tasks.boxing import BoxingCodec

        return BoxingCodec()

    if codec_name == "json":
        from synthstats.runtime.codecs import JSONToolCodec

        return JSONToolCodec()

    if codec_name == "xml":
        from synthstats.runtime.codecs import XMLToolCodec

        return XMLToolCodec()

    raise ValueError(f"Unknown codec: {codec_name}")


def build_judge(cfg: DictConfig) -> Any:
    """Instantiate judge(s) based on config."""
    judge_cfg = cfg.judge
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


def create_env(cfg: DictConfig) -> Any:
    """Create SkyRL text environment."""
    task = build_task(cfg)
    codec = build_codec(cfg)
    max_turns = getattr(cfg.task, "max_steps", 20)

    if cfg.task.name == "boxing":
        judge = build_judge(cfg)
        executors: dict[str, Any] = {}
        try:
            from synthstats.executors.pymc_sandbox import PyMCExecutor

            executors["pymc"] = PyMCExecutor()
        except Exception as e:
            logger.warning(f"Failed to init PyMCExecutor: {e}")

        env_config = BoxingEnvConfig(max_turns=max_turns)
        return BoxingEnv(
            task=task,
            codec=codec,
            executors=executors,
            judge=judge,
            config=env_config,
        )

    return SynthStatsTextEnv(
        task=task,
        codec=codec,
        executors={},
        max_turns=max_turns,
    )


def create_wandb_callback(cfg: DictConfig):
    """Create WandB logging callback if enabled."""
    if not cfg.wandb.enabled:
        return None

    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed, disabling logging")
        return None

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.get("run_name"),
        tags=cfg.wandb.get("tags", []),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    def callback(metrics: dict, step: int):
        wandb.log(metrics, step=step)

    return callback


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training entrypoint."""
    global shutdown_requested

    # register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    logger.info("Starting SkyRL training")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # set seed
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # resolve device
    device = resolve_device(cfg.device)
    logger.info(f"Using device: {device}")

    # create components
    policy = create_policy(cfg, device)
    trainer = create_trainer(cfg, device, policy)
    ref_policy = create_ref_policy(cfg, device)
    env = create_env(cfg)

    # handle resume from checkpoint
    start_step = 0
    resume_from = cfg.get("resume_from")
    if resume_from:
        start_step = load_checkpoint(resume_from, trainer, policy)
        logger.info(f"Resuming training from step {start_step}")

    # get output directory and checkpoint interval
    output_dir = cfg.get("output_dir", "checkpoints")
    checkpoint_interval = cfg.get("checkpoint_interval", 100)

    # create training loop - handle both _target_ and simple configs
    # for _target_ configs, params may be under 'config' subkey or at root
    trainer_cfg = cfg.trainer
    if "_target_" in trainer_cfg:
        batch_size = trainer_cfg.get("batch_size", trainer_cfg.get("config", {}).get("batch_size", 4))
        learning_rate = trainer_cfg.get("config", {}).get("learning_rate", 1e-5)
    else:
        batch_size = trainer_cfg.get("batch_size", 4)
        learning_rate = trainer_cfg.get("learning_rate", 1e-5)

    training_config = TrainingConfig(
        batch_size=batch_size,
        learning_rate=learning_rate,
        temperature=0.7,
        device=device,
        log_interval=cfg.get("log_interval", 10),
        eval_interval=cfg.get("eval_interval", 50),
        replay_buffer_size=trainer_cfg.get("replay_buffer_size", 0),
        replay_ratio=trainer_cfg.get("replay_ratio", 0.5),
        replay_prioritized=trainer_cfg.get("replay_prioritized", False),
        replay_alpha=trainer_cfg.get("replay_alpha", 1.0),
    )
    loop = TrainingLoop(config=training_config)

    # setup with optional WandB callback
    callback = create_wandb_callback(cfg)
    loop.setup(
        policy=policy,
        trainer=trainer,
        env=env,
        ref_policy=ref_policy,
        log_callback=callback,
    )

    # run training with checkpointing
    num_steps = trainer_cfg.get("num_episodes", 100)
    remaining_steps = num_steps - start_step
    logger.info(f"Starting training for {remaining_steps} steps (total: {num_steps})")
    logger.info(f"Batch size: {batch_size}, Checkpoint interval: {checkpoint_interval}")
    import sys
    sys.stdout.flush()  # Force flush

    all_metrics = []
    current_step = start_step

    try:
        # run in chunks to allow checkpointing
        while current_step < num_steps and not shutdown_requested:
            # calculate steps for this chunk
            steps_until_checkpoint = checkpoint_interval - (current_step % checkpoint_interval)
            steps_until_end = num_steps - current_step
            chunk_steps = min(steps_until_checkpoint, steps_until_end)

            # run chunk
            logger.info(f"Running chunk: steps {current_step} to {current_step + chunk_steps}")
            chunk_metrics = loop.run(steps=chunk_steps)
            all_metrics.extend(chunk_metrics)
            current_step += chunk_steps

            # log progress
            if chunk_metrics:
                latest = chunk_metrics[-1]
                logger.info(
                    f"Step {current_step}/{num_steps} | "
                    f"loss={latest.get('loss', 0):.4f} | "
                    f"logZ={latest.get('logZ', 0):.4f} | "
                    f"reward={latest.get('avg_reward', 0):.4f}"
                )

            # save checkpoint at intervals
            if current_step % checkpoint_interval == 0 or current_step >= num_steps:
                save_checkpoint(
                    trainer=trainer,
                    policy=policy,
                    step=current_step,
                    output_dir=output_dir,
                    metrics={"last_loss": chunk_metrics[-1].get("loss", 0)} if chunk_metrics else None,
                )

            # check for shutdown request
            if shutdown_requested:
                logger.info("Shutdown requested, saving final checkpoint...")
                save_checkpoint(
                    trainer=trainer,
                    policy=policy,
                    step=current_step,
                    output_dir=output_dir,
                    metrics={"interrupted": True},
                )
                break

        if all_metrics:
            logger.info(f"Training complete. Final metrics: {all_metrics[-1]}")
        else:
            logger.info("Training complete. No metrics recorded.")

    except KeyboardInterrupt:
        logger.info("Training interrupted, saving checkpoint...")
        save_checkpoint(
            trainer=trainer,
            policy=policy,
            step=current_step,
            output_dir=output_dir,
            metrics={"interrupted": True},
        )
    finally:
        # cleanup
        if callback:
            import wandb
            wandb.finish()


if __name__ == "__main__":
    main()
