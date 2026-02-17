import torch
from torchvision.transforms import v2
from omegaconf import DictConfig


def build_transforms(
        cfg: DictConfig
) -> tuple[v2.Compose, v2.Compose]:

    dataset = cfg.dataset

    train_transform = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(dataset.image_size, scale=(0.8, 1.0)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(0.2, 0.2, 0.2, 0.02),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=dataset.image_mean,
            std=dataset.image_std
        ),
    ])

    eval_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(256),
        v2.CenterCrop(dataset.image_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=dataset.image_mean,
            std=dataset.image_std
        ),
    ])

    return train_transform, eval_transform