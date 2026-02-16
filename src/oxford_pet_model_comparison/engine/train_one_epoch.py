import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim import Optimizer
from torch.utils.data import DataLoader

def train_one_epoch(
        model: nn.Module,
        train_loader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
        optimizer: Optimizer,
        scaler: GradScaler
) -> tuple[float, float]:

    model.train()

    running_loss = 0.0
    num_correct = 0
    num_samples = 0

    use_amp = (device.type == "cuda")

    for batch in train_loader:
        images = batch["image"]
        labels = batch["label"]

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(dim=0)
        running_loss += loss.item() * batch_size

        preds = logits.argmax(dim=1)
        num_correct += preds.eq(labels).sum().item()
        num_samples += batch_size

    train_loss = running_loss / max(num_samples, 1)
    train_acc = 100.0 * num_correct / max(num_samples, 1)

    return train_loss, train_acc