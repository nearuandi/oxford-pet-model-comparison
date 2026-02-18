from pathlib import Path
import torch
from omegaconf import DictConfig

from oxford_pet_model_comparison.models import build_model
from oxford_pet_model_comparison.utils import load_image
from oxford_pet_model_comparison.datasets import build_eval_transform


class Predictor:
    def __init__(
            self,
            cfg: DictConfig,
            best_path: str | Path,
            device: torch.device
    ):
        self.cfg = cfg
        self.device = device

        # dict 형태로 저장했으니까 weights_only=False
        payload = torch.load(best_path, map_location=device, weights_only=False)

        self.model = build_model(
            model_name=cfg.model.name,
            num_classes=cfg.dataset.num_classes,
            pretrained=False,
            freeze_backbone=False
        )
        self.model.to(self.device)

        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()

        self.transform = build_eval_transform(cfg)

    @torch.inference_mode()
    def predict(self, image_path_or_url: str | Path) -> dict:
        img = load_image(image_path_or_url)
        x = self.transform(img).unsqueeze(0).to(self.device)
        logits = self.model(x)
        # 확률까지 반환 = softmax
        prob = torch.softmax(logits, dim=1)[0]
        idx = int(prob.argmax().item())

        return {
            "class_id": idx,
            "class_name": self.cfg.dataset.class_names[idx],
            "prob": float(prob[idx].item()),
        }