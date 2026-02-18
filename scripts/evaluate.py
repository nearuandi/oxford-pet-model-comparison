from omegaconf import DictConfig
import hydra

from oxford_pet_model_comparison.cli import run_eval


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_eval(cfg)


if __name__ == "__main__":
    main()