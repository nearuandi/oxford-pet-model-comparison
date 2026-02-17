from typing import cast, Literal
from dataclasses import dataclass

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau


def build_optimizer(model: nn.Module, train: DictConfig) -> Optimizer:
    params = (p for p in model.parameters() if p.requires_grad)
    betas = cast(tuple[float, float], tuple(train.optimizer.betas))

    return torch.optim.AdamW(
        params,
        lr=train.optimizer.lr,
        weight_decay=train.optimizer.weight_decay,
        betas=betas,
    )


def build_scaler(train: DictConfig, device: torch.device) -> GradScaler | None:
    use_amp = bool(train.amp) and device.type == "cuda"
    return GradScaler(enabled=True) if use_amp else None


def build_scheduler(optimizer: Optimizer, train: DictConfig) -> ReduceLROnPlateau:
    mode = cast(Literal["min", "max"], train.scheduler.mode)

    return ReduceLROnPlateau(
        optimizer=optimizer,
        mode=mode,
        factor=train.scheduler.factor,
        patience=train.scheduler.patience,
        min_lr=train.scheduler.min_lr,
    )


@dataclass(frozen=True, slots=True)
class TrainingComponents:
    optimizer: Optimizer
    scheduler: ReduceLROnPlateau | None
    scaler: GradScaler | None


def build_training_components(
    model: nn.Module,
    train: DictConfig,
    device: torch.device,
) -> TrainingComponents:
    optimizer = build_optimizer(model, train)
    scheduler = build_scheduler(optimizer, train) if train.get("scheduler") is not None else None
    scaler = build_scaler(train, device)
    return TrainingComponents(optimizer=optimizer, scheduler=scheduler, scaler=scaler)
