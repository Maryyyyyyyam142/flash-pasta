"""Input-pipeline throughput with PASTA on CPU workers vs on the GPU.

Each sample goes through: JPEG decode -> letterbox to --imgsz -> CHW uint8.
  none          no PASTA (upper bound of the loader)
  cpu_official  official PASTA inside each DataLoader worker
  cpu_ours      our implementation inside each worker (1 thread, rfft)
  gpu           batch moved to GPU, then our implementation on the whole batch

No model is trained: this measures how many images/sec the pipeline can deliver
for a given number of CPU workers, i.e. how many cores a GPU needs to stay fed.

Usage: CUDA_VISIBLE_DEVICES=2 python bench/bench_loader.py --images data/sim10k/images/train --out results/loader.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flashpasta.pasta import PASTA, pasta_uint8  # noqa: E402
from flashpasta.reference_official import PASTA as OfficialPASTA  # noqa: E402

ALPHA, BETA, K = 3.0, 0.25, 2


def letterbox(img, size):
    """Resize the long side to `size` and pad to size x size (YOLO-style). size=0 keeps native resolution."""
    if size == 0:
        return img
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = round(h * r), round(w * r)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    return cv2.copyMakeBorder(img, top, size - nh - top, left, size - nw - left, cv2.BORDER_CONSTANT, value=(114, 114, 114))


class ImageFolder(Dataset):
    def __init__(self, files, imgsz, mode):
        self.files, self.imgsz, self.mode = files, imgsz, mode
        self.official = OfficialPASTA(ALPHA, BETA, K)
        self.ours = PASTA(ALPHA, BETA, K, use_rfft=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = letterbox(cv2.imread(str(self.files[i])), self.imgsz)
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        if self.mode == "cpu_official":
            x = self.official(x.float().div(255)).mul(255).round().to(torch.uint8)
        elif self.mode == "cpu_ours":
            x = pasta_uint8(x, self.ours)
        return x


def run(files, imgsz, mode, workers, batch, device, warmup, min_seconds):
    ds = ImageFolder(files, imgsz, mode)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=True,
                    persistent_workers=False, prefetch_factor=4 if workers else None, drop_last=True)
    gpu_pasta = PASTA(ALPHA, BETA, K, use_rfft=True)
    it = iter(dl)
    # Workers deliver batches in rounds of `workers`; skip the first round and measure whole rounds.
    # Stop at a round boundary once min_seconds have passed (and >= 3 rounds), or when the epoch runs out.
    rnd = max(1, workers)
    warmup = max(warmup, rnd)
    assert warmup + 3 * rnd <= len(dl), f"dataset too small for {workers} workers"
    n_img, n_batches, gpu_aug_s, t0 = 0, 0, 0.0, None
    for step in range(len(dl)):
        x = next(it)
        x = x.to(device, non_blocking=True)
        if mode == "gpu":
            torch.cuda.synchronize(device)
            ta = time.perf_counter()
            x = pasta_uint8(x, gpu_pasta)
            torch.cuda.synchronize(device)
            if t0 is not None:
                gpu_aug_s += time.perf_counter() - ta
        else:
            torch.cuda.synchronize(device)
        if step == warmup - 1:
            t0 = time.perf_counter()
        elif step >= warmup:
            n_img += x.shape[0]
            n_batches += 1
            if n_batches % rnd == 0 and n_batches >= 3 * rnd and time.perf_counter() - t0 > min_seconds:
                break
    dt = time.perf_counter() - t0
    del it, dl
    return {"img_per_s": n_img / dt, "seconds": dt, "images": n_img, "gpu_aug_frac_of_time": gpu_aug_s / dt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder of training images")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--imgsz", default="640,0", help="0 = native resolution")
    ap.add_argument("--modes", default="none,cpu_official,cpu_ours,gpu")
    ap.add_argument("--workers", default="1,2,4,8,16,32")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=2, help="batches (at least one per worker)")
    ap.add_argument("--min-seconds", type=float, default=8.0, help="measure at least this long")
    ap.add_argument("--out", default="results/loader.json")
    args = ap.parse_args()

    files = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    device = torch.device(args.device)
    rows = []
    for imgsz in map(int, args.imgsz.split(",")):
        for mode in args.modes.split(","):
            for workers in map(int, args.workers.split(",")):
                r = run(files, imgsz, mode, workers, args.batch, device, args.warmup, args.min_seconds)
                row = {"imgsz": imgsz or "native", "mode": mode, "workers": workers, "batch": args.batch, **r}
                rows.append(row)
                print(json.dumps(row), flush=True)
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).write_text(json.dumps({"images_dir": args.images, "n_files": len(files),
                                                      "gpu": torch.cuda.get_device_name(device), "rows": rows}, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
