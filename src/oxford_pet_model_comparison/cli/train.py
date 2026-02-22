from pathlib import Path
import torch

from oxford_pet_model_comparison.utils import seed_everything
from oxford_pet_model_comparison.engine import Trainer
from oxford_pet_model_comparison.models import build_model
from oxford_pet_model_comparison.datasets import build_datamodule
from oxford_pet_model_comparison.utils import ensure_dir


def run_train(cfg) -> None:
    seed_everything(int(cfg.train.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg.paths.out_dir) / cfg.exp.name
    ensure_dir(out_dir)

    datamodule = build_datamodule(cfg)

    model = build_model(
        cfg.model.name,
        num_classes=cfg.dataset.num_classes,
        pretrained=cfg.model.pretrained,
        freeze_backbone=cfg.model.freeze_backbone
    )

    trainer = Trainer(model=model, cfg=cfg, device=device)
    trainer.fit(
        out_dir=out_dir,
        train_loader=datamodule.train_loader,
        val_loader=datamodule.val_loader
    )