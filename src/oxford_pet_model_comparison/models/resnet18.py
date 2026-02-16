import torch.nn as nn
from torchvision import models


def build_resnet18(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False
) -> nn.Module:
    # 가중치 설정
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # classifier 교체
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # freeze backbone
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False

    return model