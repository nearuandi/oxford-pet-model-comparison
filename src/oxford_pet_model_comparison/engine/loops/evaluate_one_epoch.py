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

    running_loss = 0.0
    num_correct = 0
    num_samples = 0

    use_amp = (device.type == "cuda")

    for batch in val_loader:
        images = batch["image"]
        labels = batch["label"]

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, labels)

        batch_size = labels.size(dim=0)
        running_loss += loss.item() * batch_size

        preds = logits.argmax(dim=1)
        num_correct += preds.eq(labels).sum().item()
        num_samples += batch_size

    val_loss = running_loss / max(num_samples, 1)
    val_acc = 100.0 * num_correct / max(num_samples, 1)

    return val_loss, val_acc