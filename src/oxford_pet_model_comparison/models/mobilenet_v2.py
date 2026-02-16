import torch.nn as nn
from torchvision import models


def build_mobilenet_v2(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False
) -> nn.Module:
    # 가중치 설정
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    # classifier 교체
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    # freeze backbone
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False

    return model