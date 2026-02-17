import torch
from pathlib import Path

from omegaconf import DictConfig
import hydra
from hydra.utils import get_original_cwd

from oxford_pet_model_comparison.datasets import build_transforms
from oxford_pet_model_comparison.engine import load_best_model
from oxford_pet_model_comparison.utils import make_batch_image_from_url

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    runs_dir = Path(get_original_cwd()) / cfg.paths.runs_dir
    run_dir = runs_dir / cfg.exp.name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"predict device: {device}")
    print(f"predict model: {cfg.model.name}")

    model_name = cfg.model.name

    model = load_best_model(
        model_name=model_name,
        best_path=run_dir / "best.pt",
        num_classes=cfg.dataset.num_classes,
        device=device
    )
    model.to(device)
    model.eval()

    url = "https://bowwowinsurance.com.au/wp-content/uploads/2020/09/shutterstock_48271828-keeshond-thumbnail.jpg"

    _, eval_transform, = build_transforms()
    img = make_batch_image_from_url(
        url=url, transform=eval_transform
    )
    img = img.to(device)

    with torch.no_grad():
        logits = model(img)
        pred = logits.argmax(dim=1)
        pred_idx = pred.item()

    print(f"pred_idx={pred_idx} | class={cfg.dataset.class_names[pred_idx]}")


if __name__ == "__main__":
    main()