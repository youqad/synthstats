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
from synthstats.training.checkpointing import cleanup_old_checkpoints
from synthstats.training.train_loop import TrainingConfig, TrainingLoop

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _sync_skyrl_registries() -> None:
    """Sync SkyRL registries if Ray is initialized.

    This ensures registered losses (tb_identity, trajectory_balance, modified_subtb)
    are available to Ray workers for distributed training.
    """
    try:
        import ray
        if not ray.is_initialized():
            return
        from skyrl_train.utils.ppo_utils import sync_registries
        sync_registries()
        logger.info("SkyRL registries synced with Ray workers")
    except ImportError:
        pass  # SkyRL or Ray not available

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
    loop: TrainingLoop | None = None,
) -> Path:
    """Save training checkpoint.

    For SkyRLSubTBTrainer with a TrainingLoop, uses the module's CheckpointState
    format which includes RNG states and replay buffer for full reproducibility.
    For TinkerTrainer, uses the trainer's own checkpoint method.

    Args:
        trainer: SkyRL trainer or TinkerTrainer with logZ parameter
        policy: HFPolicy or TinkerPolicy with model state
        step: Current training step
        output_dir: Directory to save checkpoint
        metrics: Optional metrics to include
        loop: TrainingLoop instance for full state checkpointing

    Returns:
        Path to saved checkpoint
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"checkpoint_{step:06d}.pt"

    # TinkerTrainer has its own checkpoint method
    if hasattr(trainer, "save_checkpoint") and not hasattr(trainer, "logZ"):
        trainer.save_checkpoint(str(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        checkpoint["step"] = step
        checkpoint["metrics"] = metrics or {}
        if hasattr(trainer, "optimizer") and trainer.optimizer:
            checkpoint["optimizer_state"] = trainer.optimizer.state_dict()
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    # SkyRLSubTBTrainer: use loop's checkpoint if available (full state)
    if loop is not None:
        loop.step_count = step  # sync step count
        return loop.save_checkpoint(checkpoint_path)

    # fallback: simple format (for backward compat or when loop unavailable)
    optimizer_state = (
        trainer.optimizer.state_dict()
        if hasattr(trainer, "optimizer") and trainer.optimizer
        else None
    )
    checkpoint = {
        "step": step,
        "logZ": trainer.logZ.item(),
        "optimizer_state": optimizer_state,
        "metrics": metrics or {},
    }
    if hasattr(policy, "model") and hasattr(policy.model, "state_dict"):
        checkpoint["model_state"] = policy.model.state_dict()

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str | Path,
    trainer: Any,
    policy: Any,
    loop: TrainingLoop | None = None,
) -> int:
    """Load training checkpoint.

    Handles both old (simple) and new (CheckpointState) formats.

    Args:
        checkpoint_path: Path to checkpoint file
        trainer: SkyRL trainer or TinkerTrainer to restore
        policy: HFPolicy or TinkerPolicy to restore
        loop: TrainingLoop instance for full state restoration

    Returns:
        Step number to resume from
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # TinkerTrainer has its own load_checkpoint method
    if hasattr(trainer, "load_checkpoint") and not hasattr(trainer, "logZ"):
        checkpoint = trainer.load_checkpoint(str(checkpoint_path), strict=False)
        if (
            hasattr(trainer, "optimizer")
            and trainer.optimizer
            and checkpoint.get("optimizer_state")
        ):
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])
        step = checkpoint.get("step", 0)
        logger.info(f"Resumed TinkerTrainer from step {step}")
        return step

    checkpoint = torch.load(checkpoint_path, weights_only=False)

    # detect format: new CheckpointState has step_count, old has step
    is_new_format = "step_count" in checkpoint

    if is_new_format and loop is not None:
        # full restoration via TrainingLoop
        loop.load_checkpoint(checkpoint_path)
        return loop.step_count

    # old format or no loop: manual restoration
    step_key = "step_count" if is_new_format else "step"
    step = checkpoint.get(step_key, 0)

    with torch.no_grad():
        trainer.logZ.fill_(checkpoint["logZ"])

    # optimizer state location differs by format
    opt_state = checkpoint.get("optimizer_state_dict") or checkpoint.get("optimizer_state")
    if opt_state and hasattr(trainer, "optimizer") and trainer.optimizer:
        trainer.optimizer.load_state_dict(opt_state)

    # model state location differs by format
    model_state = checkpoint.get("model_state_dict") or checkpoint.get("model_state")
    if model_state and hasattr(policy, "model"):
        policy.model.load_state_dict(model_state)

    # restore RNG states for new format (critical for reproducibility)
    if is_new_format and "rng_states" in checkpoint:
        from synthstats.training.checkpointing import set_rng_states
        set_rng_states(checkpoint["rng_states"])
        logger.info("Restored RNG states from checkpoint")

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
            from synthstats.integrations.tinker import TinkerConfig, TinkerPolicy
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

    max_new_tokens = cfg.model.get("max_new_tokens", 300)
    dtype_str = cfg.model.get("dtype", "bfloat16")
    gradient_checkpointing = cfg.model.get("gradient_checkpointing", False)

    # LoRA config from Hydra (e.g., ++model.lora.r=16 ++model.lora.alpha=32)
    lora_cfg = cfg.model.get("lora", None)
    lora_config = OmegaConf.to_container(lora_cfg, resolve=True) if lora_cfg else None

    logger.info(f"Loading HFPolicy: {model_name} (max_new_tokens={max_new_tokens}, dtype={dtype_str})")
    if lora_config:
        logger.info(f"LoRA config: r={lora_config.get('r')}, alpha={lora_config.get('alpha')}")
    return HFPolicy(
        model_name=model_name,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=cfg.model.get("temperature", 0.7),
        require_grad_logp=cfg.model.get("require_grad_logp", False),
        dtype=dtype_str,
        lora_config=lora_config,
        gradient_checkpointing=gradient_checkpointing,
    )


def create_trainer(cfg: DictConfig, device: str, policy: Any | None = None) -> Any:
    """Create trainer from config."""
    if "_target_" in cfg.trainer:
        logger.info("Using Hydra instantiation for trainer")
        from hydra.utils import instantiate

        from synthstats.integrations.tinker import TinkerConfig, TinkerTrainer

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
        beta=cfg.trainer.get("beta", 1.0),  # reward scaling: exp(beta * ELPD)
        use_ref_policy=cfg.trainer.get("use_ref_policy", False),
        ref_weight=cfg.trainer.get("ref_weight", 1.0),
        normalize_by_length=cfg.trainer.get("normalize_by_length", False),
        tb_max_residual=cfg.trainer.get("tb_max_residual", 100.0),
        max_grad_norm=cfg.trainer.get("max_grad_norm", None),  # gradient clipping
        allow_mismatched_tokenizer=cfg.trainer.get(
            "allow_mismatched_tokenizer", False
        ),
    )
    trainer = SkyRLSubTBTrainer(
        config=subtb_config,
        device=device,
    )

    params = [{"params": [trainer.logZ], "lr": cfg.trainer.get("logZ_lr", 0.1)}]

    # only train policy params when explicitly requested (require_grad_logp=True)
    if (
        policy is not None
        and hasattr(policy, "parameters")
        and cfg.model.get("require_grad_logp", False)
    ):
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
            logger.info(f"Policy params added to optimizer ({len(policy_params)} tensors)")
        else:
            logger.info("No trainable policy params found")
    else:
        logger.info("Policy frozen — only training logZ")

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

            beta = judge_spec.get("beta", 1.0)
            judges_with_weights.append((LikelihoodJudge(beta=beta), weight))
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

    # sync SkyRL registries if Ray is running (for distributed training)
    _sync_skyrl_registries()

    logger.info("Starting SkyRL training")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = resolve_device(cfg.device)
    logger.info(f"Using device: {device}")

    policy = create_policy(cfg, device)
    trainer = create_trainer(cfg, device, policy)
    ref_policy = create_ref_policy(cfg, device)
    env = create_env(cfg)

    # get checkpoint config
    checkpoint_cfg = cfg.get("checkpoint", {})
    output_dir = checkpoint_cfg.get("save_path") or cfg.get("output_dir", "checkpoints")
    checkpoint_interval = cfg.get("checkpoint_interval") if cfg.get("checkpoint_interval") is not None else checkpoint_cfg.get("save_interval", 100)
    keep_last_n = checkpoint_cfg.get("keep_last_n", 3)
    checkpointing_enabled = checkpoint_interval > 0 and output_dir is not None

    # create training loop
    trainer_cfg = cfg.trainer
    if "_target_" in trainer_cfg:
        batch_size = trainer_cfg.get(
            "batch_size",
            trainer_cfg.get("config", {}).get("batch_size", 4),
        )
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

    callback = create_wandb_callback(cfg)
    loop.setup(
        policy=policy,
        trainer=trainer,
        env=env,
        ref_policy=ref_policy,
        log_callback=callback,
    )

    # handle resume from checkpoint (after loop setup for full state restoration)
    start_step = 0
    resume_from = cfg.get("resume_from") or checkpoint_cfg.get("resume_from")
    if resume_from:
        start_step = load_checkpoint(resume_from, trainer, policy, loop=loop)
        logger.info(f"Resuming training from step {start_step}")

    # run training with checkpointing
    num_steps = trainer_cfg.get("num_episodes", 100)
    remaining_steps = num_steps - start_step
    logger.info(f"Starting training for {remaining_steps} steps (total: {num_steps})")
    logger.info(f"Batch size: {batch_size}, Checkpoint interval: {checkpoint_interval}")
    if not checkpointing_enabled:
        logger.info("Checkpointing disabled (checkpoint_interval <= 0)")
    sys.stdout.flush()  # Force flush

    all_metrics = []
    current_step = start_step

    try:
        # run in chunks to allow checkpointing
        while current_step < num_steps and not shutdown_requested:
            # calculate steps for this chunk
            steps_until_end = num_steps - current_step
            if checkpointing_enabled:
                steps_until_checkpoint = checkpoint_interval - (current_step % checkpoint_interval)
                chunk_steps = min(steps_until_checkpoint, steps_until_end)
            else:
                chunk_steps = steps_until_end

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
            if checkpointing_enabled and (
                current_step % checkpoint_interval == 0 or current_step >= num_steps
            ):
                last_loss = chunk_metrics[-1].get("loss", 0) if chunk_metrics else None
                save_checkpoint(
                    trainer=trainer,
                    policy=policy,
                    step=current_step,
                    output_dir=output_dir,
                    metrics={"last_loss": last_loss} if last_loss is not None else None,
                    loop=loop,
                )
                # clean up old checkpoints
                if keep_last_n > 0:
                    cleanup_old_checkpoints(Path(output_dir), keep_last_n)

            # check for shutdown request
            if shutdown_requested:
                logger.info("Shutdown requested, saving final checkpoint...")
                save_checkpoint(
                    trainer=trainer,
                    policy=policy,
                    step=current_step,
                    output_dir=output_dir,
                    metrics={"interrupted": True},
                    loop=loop,
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
            loop=loop,
        )
    finally:
        # cleanup
        if callback:
            import wandb
            wandb.finish()


if __name__ == "__main__":
    main()
