from io import BytesIO
from typing import Optional

import matplotlib.pyplot as plt
import requests
from PIL import Image
import torch
from torch import Tensor


_SESSION = requests.Session()

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,ko;q=0.8",
    "Connection": "keep-alive",
}


def pil_from_url(url: str, timeout: int = 15) -> Image.Image:
    headers = dict(_DEFAULT_HEADERS)
    headers["Referer"] = url

    resp = _SESSION.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    img = Image.open(BytesIO(resp.content))

    if img.mode != "RGB":
        img = img.convert("RGB")

    return img


def show_image_from_url(url: str) -> None:
    img_pil = pil_from_url(url)
    plt.imshow(img_pil)
    plt.axis("off")
    plt.show()


def make_batch_image_from_url(url: str, transform=None, device: Optional[torch.device] = None) -> Tensor:
    img_pil = pil_from_url(url)

    if transform is None:
        img = torch.from_numpy(__import__("numpy").array(img_pil)).permute(2, 0, 1).float() / 255.0
    else:
        img = transform(img_pil)  # (C,H,W)

    batch_img = img.unsqueeze(0)  # (1,C,H,W)

    if device is not None:
        batch_img = batch_img.to(device, non_blocking=True)

    return batch_img
