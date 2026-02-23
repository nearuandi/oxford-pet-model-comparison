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
        scaler: GradScaler,
        amp: bool = True
) -> tuple[float, float]:

    model.train()

    sum_loss = 0.0
    sum_correct = 0
    sum_count = 0

    use_autocast = bool(amp) and (device.type == "cuda")
    use_scaler = use_autocast and (scaler is not None)

    for batch in train_loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_autocast):
            logits = model(x)
            loss = loss_fn(logits, y)

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = y.size(dim=0)
        sum_loss += loss.item() * bs
        sum_count += bs

        preds = logits.argmax(dim=1)
        sum_correct += preds.eq(y).sum().item()

    train_loss = sum_loss / max(1, sum_count)
    train_acc = 100.0 * sum_correct / max(1, sum_count)

    return train_loss, train_acc