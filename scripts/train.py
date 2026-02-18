import hydra
from omegaconf import DictConfig

from oxford_pet_model_comparison.cli import run_train



@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_train(cfg)

if __name__ == "__main__":
    main()