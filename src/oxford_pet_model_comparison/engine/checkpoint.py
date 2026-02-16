from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from oxford_pet_model_comparison.models import build_model


def save_best(
    run_dir: str | Path,
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    scaler: GradScaler | None,
    scheduler: ReduceLROnPlateau | None,
    best_val_acc: float
) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    best = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_val_acc": best_val_acc
    }
    torch.save(best, run_dir / "best.pt")

def save_history(
    run_dir: str | Path,
    history: dict[str, list],
    train_time: float,
    best_val_acc: float
) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    history_data = {
        "history": history,
        "train_time": train_time,
        "best_val_acc": best_val_acc
    }
    torch.save(history_data, run_dir / "history.pt")

def load_best_model(
    model_name: str,
    best_path: str | Path,
    num_classes: int,
    device: torch.device
) -> nn.Module:
    best_path = Path(best_path)

    best = torch.load(best_path, map_location=device, weights_only=True)
    model = build_model(model_name, num_classes=num_classes)
    model.load_state_dict(best["model_state_dict"])
    return model