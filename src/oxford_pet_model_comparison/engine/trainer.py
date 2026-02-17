import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .loops.train_one_epoch import train_one_epoch
from .loops.evaluate_one_epoch import evaluate_one_epoch
from .checkpoint import save_best, save_history


@dataclass(slots=True)
class TrainState:
    model: nn.Module
    optimizer: Optimizer
    scaler: GradScaler | None
    scheduler: ReduceLROnPlateau | None


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        device: torch.device,
        optimizer: Optimizer,
        scaler: GradScaler | None = None,
        scheduler: ReduceLROnPlateau | None = None
    ) -> None:
        self.device = device
        self.loss_fn = loss_fn.to(device)

        self.state = TrainState(
            model=model.to(device),
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
        )

    @property
    def model(self) -> nn.Module:
        return self.state.model

    @property
    def optimizer(self) -> Optimizer:
        return self.state.optimizer

    @property
    def scaler(self) -> GradScaler | None:
        return self.state.scaler

    @property
    def scheduler(self) -> ReduceLROnPlateau | None:
        return self.state.scheduler

    def fit(
        self,
        exp_name: str,
        run_dir: str | Path,
        num_epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        run_dir = Path(run_dir)

        history: dict[str, list[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
        }

        print(f"{exp_name} 훈련 시작")

        best_acc = 0.0
        start = time.time()

        for epoch in range(num_epochs):
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
                data_loader=val_loader,
                loss_fn=self.loss_fn,
                device=self.device,
            )

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            print(
                f"[Epoch {epoch + 1:02d}/{num_epochs}] {exp_name} | "
                f"Train: Loss {train_loss:.4f}, Acc {train_acc:.2f}% | "
                f"Val: Loss {val_loss:.4f}, Acc {val_acc:.2f}%"
            )

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                save_best(
                    run_dir=run_dir,
                    epoch=epoch + 1,
                    model=self.model,
                    optimizer=self.optimizer,
                    scaler=self.scaler,
                    scheduler=self.scheduler,
                    best_val_acc=best_acc,  # 기존 포맷 유지
                )
                print(f"Best Updated: {best_acc:.2f}%")

        train_time = time.time() - start
        save_history(
            run_dir=run_dir,
            history=history,
            train_time=train_time,
            best_val_acc=best_acc,
        )
        print(
            f"{exp_name} 훈련 완료, "
            f"train_time: {train_time / 60:.1f}분, "
            f"best_val_acc: {best_acc:.2f}\n"
        )
