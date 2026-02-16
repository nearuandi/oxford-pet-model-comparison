import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .checkpoint import save_best, save_history
from .train_one_epoch import train_one_epoch
from .evaluate_one_epoch import evaluate_one_epoch


def fit(
    run_name: str,
    run_dir: str | Path,
    num_epochs: int,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: Optimizer,
    scaler: GradScaler | None,
    scheduler: ReduceLROnPlateau | None,
) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }

    print(f"{run_name} 모델 훈련 시작")

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            device=device,
            optimizer=optimizer,
            scaler=scaler
        )
        val_loss, val_acc = evaluate_one_epoch(
            model=model,
            data_loader=val_loader,
            loss_fn=loss_fn,
            device=device
        )
        if scheduler is not None:
            scheduler.step(val_loss)

        print(
            f"[Epoch {epoch + 1:02d}/{num_epochs}] "
            f"{run_name} | "
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
                run_dir=run_dir,
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                best_val_acc=best_val_acc
            )
            print(f"Best Updated: {best_val_acc:.2f}%")

    train_time = time.time() - start_time
    save_history(
        run_dir=run_dir,
        history=history,
        train_time=train_time,
        best_val_acc=best_val_acc
    )
    print(f"{run_name} 모델 훈련 완료, train_time: {train_time / 60:.1f}분, best_val_acc: {best_val_acc:.2f}\n")