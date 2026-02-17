import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

from .oxford_pet_dataset import OxfordPetDataset
from .transforms import build_transforms


def build_dataloaders(
        cfg: DictConfig
) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset = cfg.dataset
    train = cfg.train

    data_root = dataset.root

    download = dataset.download

    train_ratio = dataset.train_ratio
    seed = cfg.seed

    train_transform, eval_transform = build_transforms()

    base_ds = OxfordPetDataset(
        root=data_root,
        split="trainval",
        transform=None,
        download=download
    )
    download = False

    base_len = len(base_ds)
    train_size = int(base_len * train_ratio)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(base_len, generator=g).tolist()
    train_indices = perm[:train_size]
    val_indices = perm[train_size:]

    train_ds_full = OxfordPetDataset(
        root=data_root,
        split="trainval",
        transform=train_transform,
        download=download
    )
    val_ds_full = OxfordPetDataset(
        root=data_root,
        split="trainval",
        transform=eval_transform,
        download=download
    )

    train_ds = Subset(train_ds_full, train_indices)
    val_ds = Subset(val_ds_full, val_indices)

    test_ds = OxfordPetDataset(
        root=data_root,
        split="test",
        transform=eval_transform,
        download=download
    )

    dl_g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=train.batch_size,
        shuffle=True,
        generator=dl_g,
        num_workers=train.num_workers,
        pin_memory=train.pin_memory,
        persistent_workers=train.persistent_workers,
        drop_last=train.drop_last,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train.batch_size,
        shuffle=False,
        num_workers=train.num_workers,
        pin_memory=train.pin_memory,
        persistent_workers=train.persistent_workers,
        drop_last=False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=train.batch_size,
        shuffle=False,
        num_workers=train.num_workers,
        pin_memory=train.pin_memory,
        persistent_workers=train.persistent_workers,
        drop_last=False
    )

    return train_loader, val_loader, test_loader