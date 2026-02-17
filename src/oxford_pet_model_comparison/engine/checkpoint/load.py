from pathlib import Path

import torch
import torch.nn as nn

from oxford_pet_model_comparison.models import build_model


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