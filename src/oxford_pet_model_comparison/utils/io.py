import torch.nn as nn
from pathlib import Path
import torch
from omegaconf import OmegaConf


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_config(path: str | Path, cfg) -> None:
    path = Path(path)
    path.write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")


def save_best(
    out_dir: str | Path,
    epoch: int,
    model: nn.Module,
    best_val_acc: float,
    exp_name: str,
    num_classes: int,
) -> None:
    out_dir = Path(out_dir)
    best = {
        "loops": epoch,
        "model_state_dict": model.state_dict(),
        "best_val_acc": best_val_acc,
        "exp_name": exp_name,
        "num_classes": num_classes,
    }
    torch.save(best, out_dir / "best.pt")

def save_history(
        out_dir: str | Path,
        history: dict[str, list],
        train_time: float,
        best_val_acc: float
) -> None:
    out_dir = Path(out_dir)
    history_data = {
        "history": history,
        "train_time": train_time,
        "best_val_acc": best_val_acc
    }
    torch.save(history_data, out_dir / "history.pt")