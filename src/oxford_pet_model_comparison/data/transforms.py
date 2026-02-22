from torchvision.transforms import v2 as T
from omegaconf import DictConfig
import torch

def build_train_transform(
        cfg: DictConfig
) -> T.Compose:
    dataset = cfg.dataset
    return T.Compose([
        T.ToImage(),
        T.Resize((dataset.image_size, dataset.image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.1),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=dataset.mean, std=dataset.std),
    ])


def build_eval_transform(
        cfg: DictConfig
) -> T.Compose:
    dataset = cfg.dataset
    return T.Compose([
        T.ToImage(),
        T.Resize((dataset.image_size, dataset.image_size)),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=dataset.mean, std=dataset.std),
    ])