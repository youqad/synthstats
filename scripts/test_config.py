import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(cfg)
    assert cfg.model.name == "Qwen/Qwen3-0.6B"
    assert cfg.trainer.algorithm == "subtb"
    print("Config loaded successfully!")

if __name__ == "__main__":
    main()
