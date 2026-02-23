from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .loops.train_one_epoch import train_one_epoch
from .loops.evaluate_one_epoch import evaluate_one_epoch
from oxford_pet_model_comparison.utils import save_config, ensure_dir, save_history, save_checkpoint


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        cfg,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)

        self.train = cfg.train

        self.loss_fn = nn.CrossEntropyLoss().to(device)

        self.optimizer = AdamW(
            params=self.model.parameters(),
            lr=self.train.optimizer.lr,
            weight_decay=self.train.optimizer.weight_decay,
            betas=self.train.optimizer.betas
        )

        self.amp_enabled = self.train.amp and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.amp_enabled)

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode=self.train.scheduler.mode,
            factor=self.train.scheduler.factor,
            patience=self.train.scheduler.patience,
            min_lr=self.train.scheduler.min_lr,
            threshold=self.train.scheduler.threshold
        )

        self.keep_last = self.train.save.keep_last

        self.best_metric = "val_acc"
        self.best_score = float("-inf")  # 0~1

    def _make_best_payload(self, epoch: int) -> dict:
        return {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "best_score": self.best_score,
            "best_metric": self.best_metric,
        }

    def _make_last_payload(self, epoch: int) -> dict:
        return {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler.is_enabled() else None,
            "best_score": self.best_score,
            "best_metric": self.best_metric
        }

    def fit(self, out_dir: str | Path, train_loader, val_loader) -> None:
        out_dir = Path(out_dir)
        ensure_dir(out_dir)
        save_config(out_dir / "config.yaml", self.cfg)

        history = {
            "train_loss": [],
            "train_acc": [],  # 0~100
            "val_loss": [],
            "val_acc": [],    # 0~100
        }

        print(f"{self.cfg.exp.name} 훈련 시작")
        print(f"pretrained: {self.cfg.model.pretrained}")
        print(f"freeze_backbone: {self.cfg.model.freeze_backbone}")

        start_time = time.time()
        for epoch in range(1, int(self.train.num_epochs) + 1):
            train_loss, train_acc = train_one_epoch(
                model=self.model,
                train_loader=train_loader,
                loss_fn=self.loss_fn,
                device=self.device,
                optimizer=self.optimizer,
                scaler=self.scaler,
                amp=self.amp_enabled,
            )

            val_loss, val_acc = evaluate_one_epoch(
                model=self.model,
                val_loader=val_loader,
                loss_fn=self.loss_fn,
                device=self.device,
                amp=self.amp_enabled,
            )

            self.scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(
                f"[Epoch {epoch:02d}/{self.train.num_epochs}] {self.cfg.exp.name} | "
                f"Train: Loss {train_loss:.4f}, Acc {train_acc:.2f}% | "
                f"Val: Loss {val_loss:.4f}, Acc {val_acc:.2f}%"
            )

            # best_score는 0~1 스케일로 통일
            score = val_acc / 100.0
            if score > self.best_score:
                self.best_score = score
                payload = self._make_best_payload(epoch=epoch)
                save_checkpoint(out_dir / "best.pt", payload)
                print(f"Best Updated: {self.best_metric}={self.best_score * 100:.2f}%")

            if self.keep_last:
                payload = self._make_last_payload(epoch=epoch)
                save_checkpoint(out_dir / "last.pt", payload)

        train_time = time.time() - start_time
        save_history(
            out_dir=out_dir,
            history=history,          # acc는 0~100
            train_time=train_time,
            best_score=self.best_score,  # 0~1
            best_metric=self.best_metric,
        )

        print(
            f"{self.cfg.exp.name} 훈련 완료, "
            f"train_time: {train_time / 60:.1f}분, "
            f"best_{self.best_metric}: {self.best_score * 100:.2f}%\n"
        )