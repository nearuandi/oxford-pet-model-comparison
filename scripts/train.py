from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from omegaconf import OmegaConf

import torch
import torch.nn as nn

from oxford_pet_model_comparison.engine import Trainer, build_training_components
from oxford_pet_model_comparison.models import build_model
from oxford_pet_model_comparison.datasets import build_dataloaders


def run_one_exp(
        cfg: DictConfig,
        device: torch.device
) -> None:

    runs_dir = Path(cfg.paths.runs_dir)

    exp_name = cfg.exp.name

    run_dir = runs_dir / exp_name
    run_dir.mkdir(exist_ok=True, parents=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")


    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    train_loader, val_loader, _ = build_dataloaders(cfg)

    model = build_model(
        model_name=cfg.model.name,
        num_classes=cfg.dataset.num_classes,
        pretrained=cfg.model.pretrained,
        freeze_backbone=cfg.model.freeze_backbone
    )
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    components = build_training_components(
        model=model,
        train=cfg.train,
        device=device,
    )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        device=device,
        optimizer=components.optimizer,
        scaler=components.scaler,
        scheduler=components.scheduler
    )

    trainer.fit(
        exp_name=exp_name,
        run_dir=run_dir,
        num_epochs=cfg.train.num_epochs,
        train_loader=train_loader,
        val_loader=val_loader
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(
        cfg: DictConfig
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"RUN: {cfg.exp.name}")
    run_one_exp(cfg, device=device)


if __name__ == "__main__":
    main()