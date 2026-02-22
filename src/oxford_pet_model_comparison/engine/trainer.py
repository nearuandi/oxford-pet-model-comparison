from pathlib import Path
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler

from .loops.train_one_epoch import train_one_epoch
from .loops.evaluate_one_epoch import evaluate_one_epoch
from oxford_pet_model_comparison.utils import save_config, ensure_dir, save_history, save_best


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            cfg,
            device: torch.device
    ):
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)

        self.loss_fn = nn.CrossEntropyLoss().to(device)

        self.train = self.cfg.train
        self.dataset = self.cfg.dataset


        self.optimizer = AdamW(
            params=self.model.parameters(),
            lr=self.train.optimizer.lr,
            weight_decay=self.train.optimizer.weight_decay
        )
        self.scaler = GradScaler(
            enabled=self.train.amp and device.type == "cuda"
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode=self.train.scheduler.mode,
            factor=self.train.scheduler.factor,
            patience=self.train.scheduler.patience,
            min_lr=self.train.scheduler.min_lr
        )

    def fit(self, out_dir: str | Path, train_loader, val_loader):
        out_dir = Path(out_dir)

        save_config(out_dir / "config.yaml", self.cfg)

        history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
        }

        print(f"{self.cfg.exp.name} 훈련 시작")
        print(f"pretrained: {self.cfg.model.pretrained}")
        print(f"freeze_backbone: {self.cfg.model.freeze_backbone}")

        best_val_acc = 0.0
        start_time = time.time()
        for epoch in range(1, self.train.num_epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model=self.model,
                train_loader=train_loader,
                loss_fn=self.loss_fn,
                device=self.device,
                optimizer=self.optimizer,
                scaler=self.scaler
            )
            val_loss, val_acc = evaluate_one_epoch(
                model=self.model,
                val_loader=val_loader,
                loss_fn=self.loss_fn,
                device=self.device
            )
            self.scheduler.step(val_loss)

            print(
                f"[Epoch {epoch:02d}/{self.train.num_epochs}] {self.cfg.exp.name} | "
                f"Train: Loss {train_loss:.4f}, Acc {train_acc:.2f}% | "
                f"Val: Loss {val_loss:.4f}, Acc {val_acc:.2f}%"
            )

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_best(
                    out_dir=out_dir,
                    epoch=epoch,
                    model=self.model,
                    best_val_acc=best_val_acc,
                    exp_name=self.cfg.exp.name,
                    num_classes=self.cfg.dataset.num_classes
                )
                print(f"Best Updated: {best_val_acc:.2f}%")

        train_time = time.time() - start_time
        save_history(
            out_dir=out_dir,
            history=history,
            train_time=train_time,
            best_val_acc=best_val_acc
        )
        print(
            f"{self.cfg.exp.name} 훈련 완료, "
            f"train_time: {train_time / 60:.1f}분, "
            f"best_val_acc: {best_val_acc:.2f}%\n"
        )