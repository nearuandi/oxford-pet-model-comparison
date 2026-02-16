import torch
from torchvision.transforms import v2


def build_transforms(image_size: int = 224) -> tuple[v2.Compose, v2.Compose]:

    train_transform = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(0.2, 0.2, 0.2, 0.02),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(256),
        v2.CenterCrop(image_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, eval_transform