from pathlib import Path
from typing import Callable

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet

class OxfordPetDataset(Dataset):
    def __init__(
            self,
            root: str | Path,
            split: str,
            transform: Callable | None,
            download: bool = True
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.transform = transform

        self.dataset = OxfordIIITPet(
            root=self.root,
            split=split, 
            target_types="category",
            download=download
        )

    def __len__(self) -> int:
        return len(self.dataset)
    def __getitem__(self, idx) -> dict[str, Tensor]:
        image, label = self.dataset[idx]

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long)
        }