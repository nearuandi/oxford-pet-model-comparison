from pathlib import Path
import torch

from oxford_pet_model_comparison.pipelines import Predictor


def run_infer(cfg) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.paths.out_dir) / cfg.exp.name
    best_path = out_dir / "best.pt"

    ckpt = torch.load(best_path, map_location=device, weights_only=False)

    best_epoch = ckpt.get("epoch")
    best_metric = ckpt.get("best_metric")
    best_score = ckpt.get("best_score")

    image_path = cfg.image_path

    predictor = Predictor(cfg, ckpt=ckpt, device=device)
    result = predictor.predict(image_path_or_url=image_path)

    print(f"모델명={cfg.model.name}\n"
          f"실험명={cfg.exp.name}")
    print(f"최고 성능={best_epoch:02d} epoch | 기준 지표: 검증 정확도 | 점수={best_score * 100:.2f}%")
    print(f"예측 번호={result['class_id']}\n"
          f"예측 품종={result['class_name']}\n"
          f"정확도={result['prob']*100:.2f}%")
