"""GFlowNet experiment class for SkyRL integration.

Extends BasePPOExp with GFlowNet-specific setup: no critic model
(GFlowNets don't use value functions), GFlowNetTrainer instead of
RayPPOTrainer, and SynthStats environment configuration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


# try to import SkyRL components
try:
    from skyrl_train.entrypoints.main_base import BasePPOExp

    SKYRL_AVAILABLE = True
except ImportError:
    BasePPOExp = object  # type: ignore[assignment,misc]
    SKYRL_AVAILABLE = False


class GFlowNetExp(BasePPOExp):  # type: ignore[misc]
    """GFlowNet experiment for distributed SubTB training.

    Extends BasePPOExp with GFlowNetTrainer (SubTB loss), no critic model,
    and SynthStats environment setup.
    """

    def __init__(self, cfg: DictConfig) -> None:
        if not SKYRL_AVAILABLE:
            raise ImportError(
                "SkyRL is required for GFlowNetExp. Install with: pip install skyrl-train"
            )

        # validate GFN config exists
        if not hasattr(cfg, "gfn"):
            logger.warning(
                "No gfn config found, using defaults. "
                "Add gfn section to config for GFlowNet settings."
            )

        super().__init__(cfg)

    def get_trainer(self) -> Any:
        """Return GFlowNetTrainer instead of RayPPOTrainer."""
        from synthstats.distributed.gfn_trainer import GFlowNetTrainer

        return GFlowNetTrainer(
            cfg=self.cfg,
            tracking=self.tracker,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            inference_engine_client=self.inference_engine_client,
            generator=self.generator,
            placement_group=getattr(self, "colocate_pg", None),
        )

    def _setup_trainer(self) -> None:
        """Setup trainer without critic."""
        # import strategy-specific workers
        if self.cfg.trainer.strategy == "fsdp2":
            from skyrl_train.actors.fsdp_workers import PolicyWorker
        else:
            from skyrl_train.actors.megatron_workers import PolicyWorker

        self._create_inference_engines()
        self.generator = self.get_generator()
        self.trainer = self.get_trainer()
        self._build_models_no_critic(PolicyWorker)

    def _build_models_no_critic(self, PolicyWorker: type) -> None:
        """Build policy and reference models without critic."""
        from skyrl_train.ray_utils import PPORayActorGroup

        # policy actor group
        policy_config = self.cfg.trainer.policy
        self.trainer.policy_actor_group = PPORayActorGroup(
            actor_cls=PolicyWorker,
            num_actors=policy_config.num_workers,
            num_gpus_per_actor=policy_config.num_gpus_per_worker,
            resources={"node": 1},
        )
        self.trainer.policy_actor_group.init_model(
            model_path=policy_config.model.path,
            tokenizer=self.tokenizer,
            config=self.cfg,
        )

        # reference model (for KL if configured)
        if getattr(self.cfg.trainer, "use_reference_model", False):
            ref_config = getattr(self.cfg.trainer, "reference", policy_config)
            self.trainer.ref_actor_group = PPORayActorGroup(
                actor_cls=PolicyWorker,
                num_actors=ref_config.num_workers,
                num_gpus_per_actor=ref_config.num_gpus_per_worker,
                resources={"node": 1},
            )
            self.trainer.ref_actor_group.init_model(
                model_path=ref_config.model.path,
                tokenizer=self.tokenizer,
                config=self.cfg,
            )
        else:
            self.trainer.ref_actor_group = None

        self.trainer.critic_actor_group = None

        logger.info(
            "GFlowNet models initialized: "
            f"policy={policy_config.model.path}, "
            f"reference={'yes' if self.trainer.ref_actor_group else 'no'}, "
            f"critic=no"
        )

    def get_train_dataset(self) -> Any:
        """Return training prompts from config.

        Prompts are loaded from config paths (data.train_files or
        task.prompt_file). Task-specific prompt generation should happen
        in the script/CLI layer, not here — this module must not import
        task plugins (dependency inversion).
        """
        # SkyRL dataset from file paths
        if hasattr(self.cfg, "data") and hasattr(self.cfg.data, "train_files"):
            return super().get_train_dataset()

        # pre-generated prompt file (one prompt per line)
        task_cfg = getattr(self.cfg, "task", {})
        prompt_file = task_cfg.get("prompt_file") if hasattr(task_cfg, "get") else None
        if prompt_file:
            return self._load_prompt_file(prompt_file)

        # no task-specific prompts configured — use placeholders
        num_prompts = task_cfg.get("num_prompts", 100) if hasattr(task_cfg, "get") else 100
        logger.warning(
            "No prompt source configured (data.train_files or task.prompt_file). "
            "Using generic placeholders. Generate task prompts via the script layer."
        )
        return self._make_placeholder_prompts(num_prompts)

    def _load_prompt_file(self, path: str) -> Any:
        """Load prompts from a text file (one per line or JSON lines)."""
        import json
        from pathlib import Path

        from skyrl_train.data.prompt_dataset import PromptDataset

        prompt_path = Path(path)
        if not prompt_path.exists():
            logger.warning(f"Prompt file {path} not found, using placeholders")
            return self._make_placeholder_prompts(100)

        prompts = []
        for line in prompt_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            # support JSON lines (each line is a JSON string)
            if line.startswith('"') or line.startswith("{"):
                try:
                    parsed = json.loads(line)
                    if isinstance(parsed, str):
                        prompts.append(parsed)
                    elif isinstance(parsed, dict):
                        prompts.append(parsed.get("prompt", json.dumps(parsed)))
                    else:
                        prompts.append(json.dumps(parsed))
                    continue
                except json.JSONDecodeError:
                    pass
            prompts.append(line)

        logger.info(f"Loaded {len(prompts)} prompts from {path}")
        return PromptDataset(prompts)

    def _make_placeholder_prompts(self, num_prompts: int) -> Any:
        """Generate generic placeholder prompts."""
        from skyrl_train.data.prompt_dataset import PromptDataset

        prompts = [
            f"Generate a PyMC program to model the data. [seed={i}]" for i in range(num_prompts)
        ]
        return PromptDataset(prompts)


def main() -> None:
    import hydra

    @hydra.main(
        config_path="../../configs",
        config_name="distributed/gfn_base",
        version_base=None,
    )
    def run(cfg: DictConfig) -> None:
        exp = GFlowNetExp(cfg)
        exp.trainer.train()

    run()


if __name__ == "__main__":
    main()
