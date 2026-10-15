"""Poster assets: a Sim10k frame before/after PASTA, and a QR code for the code repository.

Usage: .venv/bin/python poster/make_assets.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import segno
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pasta_gpu.pasta import PASTA  # noqa: E402

FIG = ROOT / "figures"
SIM10K = ROOT / "data/sim10k"


def pick_frame():
    """First image (sorted) among the 500 with the most car boxes: busy, but deterministic."""
    labels = sorted((SIM10K / "labels/train").glob("*.txt"))[:500]
    best = max(labels, key=lambda p: (len(p.read_text().splitlines()), -labels.index(p)))
    return SIM10K / "images/train" / f"{best.stem}.jpg"


def example(seed=3):
    src = pick_frame()
    img = Image.open(src).convert("RGB")
    x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float().div(255)[None]
    torch.manual_seed(seed)
    out = PASTA(alpha=3.0, beta=0.25, k=2)(x)[0]
    aug = Image.fromarray(out.mul(255).round().byte().permute(1, 2, 0).numpy())
    W, H = img.size
    box = (0, int(H * 0.12), W, int(H * 0.12) + int(W / 2.2))  # wide strip, aspect 2.2
    for im, name in [(img, "example_original.jpg"), (aug, "example_pasta.jpg")]:
        im.crop(box).resize((1100, 500), Image.LANCZOS).save(FIG / name, quality=90)
    print("example frame:", src.name)


def qr():
    url = json.loads((ROOT / "poster/authors.json").read_text()).get("code_url")
    if url:
        segno.make(url, error="m").save(FIG / "qr_code.svg", scale=10, border=2, dark="#0b0b0b", light="#ffffff")
        print("qr:", url)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    example()
    qr()
