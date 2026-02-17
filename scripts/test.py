import torch
import torch.nn as nn
from pathlib import Path

from omegaconf import DictConfig
import hydra
from hydra.utils import get_original_cwd

from oxford_pet_model_comparison.datasets import build_dataloaders
from oxford_pet_model_comparison.engine import evaluate_one_epoch
from oxford_pet_model_comparison.engine import load_best_model

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:

    runs_dir = Path(get_original_cwd()) / cfg.paths.runs_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"test device: {device}")

    _, _, test_loader = build_dataloaders(cfg)

    loss_fn = nn.CrossEntropyLoss()

    model_list = [
        ("simple_cnn", "simple_cnn", runs_dir / "simple_cnn/best.pt"),
        ("mobilenet_v2", "mobilenet_v2", runs_dir / "mobilenet_v2/best.pt"),
        ("resnet18", "resnet18", runs_dir / "resnet18/best.pt"),
        ("resnet18_freeze", "resnet18", runs_dir / "resnet18_freeze/best.pt"),
    ]

    for exp_name, model_name, path in model_list:
        print(f"{exp_name} 테스트 시작")
        model = load_best_model(model_name, path, cfg.dataset.num_classes, device=device)
        model.to(device)
        model.eval()

        test_loss, test_acc = evaluate_one_epoch(
            model=model,
            data_loader=test_loader,
            loss_fn=loss_fn,
            device=device,
        )
        print(f"{exp_name:<16} | test loss: {test_loss:.4f} | test acc: {test_acc:.2f}%\n")


if __name__ == "__main__":
    main()