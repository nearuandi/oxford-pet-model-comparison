from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

import torch
import torch.nn as nn

from oxford_pet_model_comparison.engine import fit, build_training_components
from oxford_pet_model_comparison.models import build_model
from oxford_pet_model_comparison.datasets import build_dataloaders


def run_one_exp(cfg: DictConfig, device: torch.device) -> None:
    run_name = cfg.exp.name

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    print(f"run_dir: {run_dir}")

    train_loader, val_loader, _ = build_dataloaders(cfg)

    model = build_model(
        cfg.model.name,
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

    fit(
        run_name=run_name,
        run_dir=run_dir,
        num_epochs=cfg.train.num_epochs,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        device=device,
        optimizer=components.optimizer,
        scaler=components.scaler,
        scheduler=components.scheduler,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"RUN: {cfg.exp.name}")
    run_one_exp(cfg, device=device)


if __name__ == "__main__":
    main()