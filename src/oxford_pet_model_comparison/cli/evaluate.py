from pathlib import Path
import torch
import torch.nn as nn

from oxford_pet_model_comparison.data import build_datamodule
from oxford_pet_model_comparison.models import build_model
from oxford_pet_model_comparison.engine.loops import evaluate_one_epoch


def run_eval(cfg) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.paths.out_dir) / cfg.exp.name
    best_path = out_dir / "best.pt"

    # dict 형태로 저장했으니까 weights_only=False
    payload = torch.load(best_path, map_location=device, weights_only=False)

    # 이미 trained weight 있어서 pretrained 의미 없음
    # freeze 는 학습할때만 필요
    model = build_model(
        model_name=cfg.model.name,
        num_classes=cfg.dataset.num_classes,
        pretrained=False,
        freeze_backbone=False
    )
    model.to(device)

    model.load_state_dict(payload["model_state_dict"])

    datamodule = build_datamodule(cfg)
    loss_fn = nn.CrossEntropyLoss()

    # version counter 업데이트 안함, view tracking 안함, autograd metadata 최소화
    with torch.inference_mode():
        val_loss, val_acc = evaluate_one_epoch(
            model=model,
            val_loader=datamodule.val_loader,
            loss_fn=loss_fn,
            device=device
        )
    print(f"cfg.model.name: {cfg.model.name}")
    print(f"cfg.exp.name= {cfg.exp.name}")
    print(f"val_loss={val_loss:.4f} | val_acc={val_acc:.2f}%")