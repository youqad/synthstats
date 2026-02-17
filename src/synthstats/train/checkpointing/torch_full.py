"""Full state checkpointing for PyTorch training."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from synthstats.train.checkpointing.base import (
    CheckpointState,
    extract_logZ,
    find_latest_checkpoint,
    should_save,
)
from synthstats.train.utils.seeding import get_rng_states, set_rng_states

logger = logging.getLogger(__name__)


class FullStateCheckpoint:
    """Full state checkpoint manager for exact reproducibility."""

    def __init__(
        self,
        save_dir: str | Path = "checkpoints",
        every_steps: int = 100,
        keep_last: int = 3,
        resume_from: str | Path | None = None,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.every_steps = every_steps
        self.keep_last = keep_last
        self.resume_from = Path(resume_from) if resume_from else None

        self._config: dict[str, Any] = {}

    def set_config(self, config: dict[str, Any]) -> None:
        self._config = config

    def maybe_save(
        self,
        step: int,
        learner: Any,
        policy: Any | None = None,
        replay_buffer: Any | None = None,
        metrics_history: list[dict[str, float]] | None = None,
    ) -> Path | None:
        if not should_save(step, self.every_steps):
            return None
        return self.save(step, learner, policy, replay_buffer, metrics_history)

    def save(
        self,
        step: int,
        learner: Any,
        policy: Any | None = None,
        replay_buffer: Any | None = None,
        metrics_history: list[dict[str, float]] | None = None,
    ) -> Path:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / f"checkpoint_{step:06d}.pt"

        model_state = None
        if policy is not None and hasattr(policy, "model"):
            model = policy.model
            if hasattr(model, "state_dict"):
                model_state = model.state_dict()

        optimizer_state = None
        if hasattr(learner, "optimizer") and learner.optimizer is not None:
            optimizer_state = learner.optimizer.state_dict()

        replay_state = None
        if replay_buffer is not None and hasattr(replay_buffer, "state_dict"):
            replay_state = replay_buffer.state_dict()

        logZ = extract_logZ(learner)

        state = CheckpointState(
            step_count=step,
            logZ=logZ,
            model_state_dict=model_state,
            optimizer_state_dict=optimizer_state,
            rng_states=get_rng_states(),
            replay_buffer=replay_state,
            config=self._config,
            metrics_history=metrics_history or [],
        )

        torch.save(state.to_dict(), path)
        logger.info(f"Saved checkpoint to {path} (step {step})")

        self._cleanup_old()

        return path

    def load(self, path: str | Path) -> CheckpointState:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # weights_only=False: CheckpointState includes numpy/Python RNG states
        data = torch.load(path, weights_only=False)
        state = CheckpointState.from_dict(data)
        logger.info(f"Loaded checkpoint from {path} (step {state.step_count})")
        return state

    def find_latest(self) -> Path | None:
        return find_latest_checkpoint(self.save_dir)

    def restore(
        self,
        learner: Any,
        policy: Any | None = None,
        replay_buffer: Any | None = None,
        path: str | Path | None = None,
    ) -> int:
        """Restore from checkpoint. Returns step count."""
        if path is None:
            path = self.resume_from or self.find_latest()
        if path is None:
            return 0

        state = self.load(path)

        if hasattr(learner, "load_state_dict"):
            learner_state = {"objective": {"logZ": state.logZ}}
            if state.optimizer_state_dict is not None:
                learner_state["optimizer"] = state.optimizer_state_dict
            try:
                learner.load_state_dict(learner_state)
            except (KeyError, TypeError):
                if hasattr(learner, "objective"):
                    with torch.no_grad():
                        learner.objective.logZ.fill_(state.logZ)

        if policy is not None and state.model_state_dict is not None:
            if hasattr(policy, "model") and hasattr(policy.model, "load_state_dict"):
                policy.model.load_state_dict(state.model_state_dict)

        set_rng_states(state.rng_states)

        if replay_buffer is not None and state.replay_buffer is not None:
            if hasattr(replay_buffer, "load_state_dict"):
                replay_buffer.load_state_dict(state.replay_buffer)

        logger.info(f"Restored state from step {state.step_count}")
        return state.step_count

    def _cleanup_old(self) -> None:
        if self.keep_last <= 0:
            return

        checkpoints = sorted(
            self.save_dir.glob("checkpoint_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )

        if len(checkpoints) <= self.keep_last:
            return

        for ckpt in checkpoints[: -self.keep_last]:
            try:
                ckpt.unlink()
                logger.debug(f"Removed old checkpoint: {ckpt}")
            except OSError as e:
                logger.warning(f"Failed to remove {ckpt}: {e}")
