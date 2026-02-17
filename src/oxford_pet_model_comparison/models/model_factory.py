from typing import Callable

import torch.nn as nn

from .simple_cnn import build_simple_cnn
from .resnet18 import build_resnet18
from .efficientnet_b0 import build_efficientnet_b0
from .mobilenet_v2 import build_mobilenet_v2


MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "simple_cnn": build_simple_cnn,
    "resnet18": build_resnet18,
    "efficientnet_b0": build_efficientnet_b0,
    "mobilenet_v2": build_mobilenet_v2,
}

def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:

    model_name = model_name.lower()

    build_fn = MODEL_REGISTRY[model_name]

    if model_name == "simple_cnn":
        return build_fn(num_classes=num_classes)

    return build_fn(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )