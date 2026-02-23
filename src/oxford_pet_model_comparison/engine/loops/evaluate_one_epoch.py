import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader

@torch.inference_mode()
def evaluate_one_epoch(
        model: nn.Module,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device
) -> tuple[float, float]:

    model.eval()

    sum_loss = 0.0
    sum_correct = 0
    sum_count = 0

    use_amp = (device.type == "cuda")

    for batch in val_loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss = loss_fn(logits, y)

        bs = y.size(dim=0)
        sum_loss += loss.item() * bs
        sum_count += bs

        preds = logits.argmax(dim=1)
        sum_correct += preds.eq(y).sum().item()

    val_loss = sum_loss / max(1, sum_count)
    val_acc = 100.0 * sum_correct / max(1, sum_count)

    return val_loss, val_acc