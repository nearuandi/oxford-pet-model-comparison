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

def save_checkpoint(path: Path, ckpt: dict) -> None:
    torch.save(ckpt, path)

def save_history(
        out_dir: str | Path,
        history: dict[str, list],
        train_time: float,
        best_score: float,
        best_metric: str
) -> None:
    out_dir = Path(out_dir)
    history_data = {
        "history": history,
        "train_time": train_time,
        "best_score": best_score,
        "best_metric": best_metric
    }
    torch.save(history_data, out_dir / "history.pt")