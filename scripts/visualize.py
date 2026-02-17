import torch
from pathlib import Path

from omegaconf import DictConfig
import hydra
from hydra.utils import get_original_cwd

from oxford_pet_model_comparison.visualize import loss_curves, acc_curves, best_val_acc, multi_model_loss, multi_model_acc

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # 파일에서 직접 실행할때 주석 해제
    # PROJECT_ROOT = Path(__file__).parent.parent
    # runs_dir = PROJECT_ROOT / "runs"

    # 프로젝트 루트에서 실행할때 주석 해제
    runs_dir = Path(get_original_cwd()) / cfg.paths.runs_dir

    exp_name = cfg.exp.name

    ckpt = torch.load(runs_dir/ exp_name / "history.pt")
    history = ckpt["history"]

    loss_curves(history, exp_name)
    acc_curves(history, exp_name)

    best_val_acc(history, exp_name)

    histories = {
        "simple_cnn": torch.load(runs_dir / "simple_cnn/history.pt")["history"],
        "mobilenet_v2": torch.load(runs_dir / "mobilenet_v2/history.pt")["history"],
        "resnet18": torch.load(runs_dir / "resnet18/history.pt")["history"],
        "resnet18_freeze": torch.load(runs_dir / "resnet18_freeze/history.pt")["history"],
    }

    multi_model_loss(histories, "Validation Loss")
    multi_model_acc(histories, "Validation Accuracy (%)")

if __name__ == "__main__":
    main()