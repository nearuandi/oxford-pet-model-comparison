from pathlib import Path
import torch
from omegaconf import DictConfig

from oxford_pet_model_comparison.models import build_model
from oxford_pet_model_comparison.utils import load_image
from oxford_pet_model_comparison.data import build_eval_transform


class Predictor:
    def __init__(
            self,
            cfg: DictConfig,
            ckpt: dict,
            device: torch.device
    ):
        self.cfg = cfg
        self.device = device

        self.model = build_model(
            model_name=cfg.model.name,
            num_classes=cfg.dataset.num_classes,
            pretrained=False,
            freeze_backbone=False
        )
        self.model.to(self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.dataset= cfg.dataset

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
            "class_name": self.dataset.class_names[idx],
            "prob": float(prob[idx].item()),
        }