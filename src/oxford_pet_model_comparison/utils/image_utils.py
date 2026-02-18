from io import BytesIO
from pathlib import Path
from urllib.request import urlopen, Request
from PIL import Image


def load_image(path_or_url: str | Path) -> Image.Image:
    s = str(path_or_url)

    if s.startswith("http://") or s.startswith("https://"):
        req = Request(s, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=15) as r:
            data = r.read()
        return Image.open(BytesIO(data)).convert("RGB")

    return Image.open(s).convert("RGB")