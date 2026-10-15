"""Smoke-test stand-in for Sim10k: JPEGs with Sim10k's size (1914x1052) and ~6 boxes each.

Only for checking the pipeline before the real dataset is downloaded. Throughput
depends on image size, JPEG decode and box count, not content, but final poster
numbers must come from the real Sim10k.

Usage: python data/make_standin.py --src path/to/gtav_syn.png --n 500 --dst data/standin
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="a 1914x1052 synthetic frame, e.g. PASTA repo media/synthetic_images/gtav_syn.png")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--dst", default="data/standin")
    args = ap.parse_args()

    base = cv2.imread(args.src)
    H, W = base.shape[:2]
    dst = Path(args.dst).resolve()
    (dst / "images/train").mkdir(parents=True, exist_ok=True)
    (dst / "labels/train").mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    for i in range(args.n):
        img = np.roll(base, (rng.randrange(H), rng.randrange(W)), axis=(0, 1))
        if rng.random() < 0.5:
            img = img[:, ::-1]
        cv2.imwrite(str(dst / f"images/train/{i:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        boxes = []
        for _ in range(rng.randint(2, 10)):
            w, h = rng.uniform(0.02, 0.2), rng.uniform(0.02, 0.15)
            boxes.append(f"0 {rng.uniform(w / 2, 1 - w / 2):.6f} {rng.uniform(h / 2, 1 - h / 2):.6f} {w:.6f} {h:.6f}")
        (dst / f"labels/train/{i:05d}.txt").write_text("\n".join(boxes) + "\n")
    (dst.parent / "standin.yaml").write_text(f"path: {dst}\ntrain: images/train\nval: images/train\nnames:\n  0: car\n")
    print(f"{args.n} images -> {dst}")


if __name__ == "__main__":
    main()
