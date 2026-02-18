from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

from .oxford_pet_dataset import OxfordPetDataset
from .transforms import build_train_transform, build_eval_transform


@dataclass(frozen=True, slots=True)
class DataModule:
    train_loader: DataLoader
    val_loader: DataLoader


def split_indices(
        n: int,
        train_ratio: float,
        seed: int
) -> tuple[list[int], list[int]]:
    train_size = int(n * train_ratio)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    return perm[:train_size], perm[train_size:]


def build_datamodule(cfg: DictConfig) -> DataModule:
    dataset = cfg.dataset
    train = cfg.train

    data_root = Path(dataset.root)

    seed = train.seed

    train_transform = build_train_transform(cfg)
    eval_transform = build_eval_transform(cfg)

    base_ds = OxfordPetDataset(
        root=data_root,
        split="trainval",
        transform=None,
        download=False,
    )

    train_idx, val_idx = split_indices(
        n=len(base_ds),
        train_ratio=dataset.train_ratio,
        seed=seed
    )

    train_ds_full = OxfordPetDataset(
        root=data_root,
        split="trainval",
        transform=train_transform,
        download=False,
    )
    val_ds_full = OxfordPetDataset(
        root=data_root,
        split="trainval",
        transform=eval_transform,
        download=False,
    )

    train_ds = Subset(train_ds_full, train_idx)
    val_ds = Subset(val_ds_full, val_idx)

    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=train.batch_size,
        shuffle=True,
        generator=g,
        num_workers=train.num_workers,
        pin_memory=train.pin_memory,
        persistent_workers=train.persistent_workers,
        drop_last=train.drop_last
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train.batch_size,
        shuffle=False,
        num_workers=train.num_workers,
        pin_memory=train.pin_memory,
        persistent_workers=train.persistent_workers,
        drop_last=False,
    )

    return DataModule(train_loader=train_loader, val_loader=val_loader)
