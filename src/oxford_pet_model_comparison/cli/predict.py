from pathlib import Path
import torch

from oxford_pet_model_comparison.pipelines import Predictor


def run_predict(cfg) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.paths.out_dir) / cfg.exp.name
    best_path = out_dir / "best.pt"

    image_path = cfg.image_path

    predictor = Predictor(cfg, best_path=best_path, device=device)
    result = predictor.predict(image_path_or_url=image_path)

    print(f"cfg.model.name={cfg.model.name}\n"
          f"cfg.exp.name={cfg.exp.name}")
    print(f"class_id={result['class_id']}\n"
          f"class_name={result['class_name']}\n"
          f"prob={result['prob']*100:.2f}%")
