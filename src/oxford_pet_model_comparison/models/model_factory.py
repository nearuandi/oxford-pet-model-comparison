import torch.nn as nn

from .simple_cnn import build_simple_cnn
from .resnet18 import build_resnet18
from .efficientnet_b0 import build_efficientnet_b0
from .mobilenet_v2 import build_mobilenet_v2


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:

    model_name = model_name.lower()


    if model_name == "simple_cnn":
        return build_simple_cnn(num_classes=num_classes)
    if model_name == "mobilenet_v2":
        return build_mobilenet_v2(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
    if model_name == "efficientnet_b0":
        return build_efficientnet_b0(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
    if model_name == "resnet18":
        return build_resnet18(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)