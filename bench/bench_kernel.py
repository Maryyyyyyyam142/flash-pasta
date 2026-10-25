"""Kernel-level PASTA throughput: images/sec, no data loading, no training.

Variants
  official_cpu   the authors' transform, one image per call (what a dataloader worker runs)
  ours_cpu       our batched rfft implementation on CPU, one image per call
  ours_gpu_fft2  our implementation on GPU, full complex FFT
  ours_gpu_rfft2 our implementation on GPU, real FFT (default)

CPU variants run in a pool of single-threaded processes (like DataLoader workers),
swept over process counts, so the result reads as "images/sec per N CPU cores".

Usage: CUDA_VISIBLE_DEVICES=2 python bench/bench_kernel.py --out results/kernel.json
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flashpasta.pasta import PASTA  # noqa: E402
from flashpasta.reference_official import PASTA as OfficialPASTA  # noqa: E402

ALPHA, BETA, K = 3.0, 0.25, 2
RESOLUTIONS = {"640x640": (640, 640), "1280x1280": (1280, 1280), "sim10k_1052x1914": (1052, 1914), "4k_2160x3840": (2160, 3840)}


def _cpu_worker(variant, H, W, seconds):
    torch.set_num_threads(1)
    img = torch.rand(3, H, W)
    fn = OfficialPASTA(ALPHA, BETA, K) if variant == "official_cpu" else None
    ours = PASTA(ALPHA, BETA, K, use_rfft=True)
    run = (lambda: fn(img)) if fn else (lambda: ours(img[None]))
    run()  # warm-up (mask cache, allocator)
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        run()
        n += 1
    return n, time.perf_counter() - t0


def cpu_throughput(variant, H, W, procs, seconds):
    with ProcessPoolExecutor(procs) as ex:
        res = list(ex.map(_cpu_worker, [variant] * procs, [H] * procs, [W] * procs, [seconds] * procs))
    return sum(n / dt for n, dt in res)


def gpu_throughput(H, W, batch, use_rfft, device, iters):
    p = PASTA(ALPHA, BETA, K, use_rfft=use_rfft)
    x = torch.rand(batch, 3, H, W, device=device)
    for _ in range(3):
        p(x)
    torch.cuda.synchronize(device)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        p(x)
    end.record()
    torch.cuda.synchronize(device)
    sec = start.elapsed_time(end) / 1000
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    torch.cuda.reset_peak_memory_stats(device)
    return batch * iters / sec, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--procs", default="1,4,8,16,32")
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--resolutions", default=",".join(RESOLUTIONS))
    ap.add_argument("--out", default="results/kernel.json")
    args = ap.parse_args()
    device = torch.device(args.device)
    rows = []

    def log(row):
        rows.append(row)
        print(json.dumps(row), flush=True)

    for res in args.resolutions.split(","):
        H, W = RESOLUTIONS[res]
        for variant in ["official_cpu", "ours_cpu"]:
            for procs in map(int, args.procs.split(",")):
                ips = cpu_throughput(variant, H, W, procs, args.seconds)
                log({"res": res, "H": H, "W": W, "variant": variant, "cpu_procs": procs, "batch": 1, "img_per_s": ips})
        for variant, rfft in [("ours_gpu_fft2", False), ("ours_gpu_rfft2", True)]:
            for batch in map(int, args.batches.split(",")):
                try:
                    iters = max(5, int(200 * 640 * 640 / (H * W * batch)))
                    ips, peak = gpu_throughput(H, W, batch, rfft, device, iters)
                    log({"res": res, "H": H, "W": W, "variant": variant, "batch": batch, "img_per_s": ips, "peak_mem_gib": peak})
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    log({"res": res, "H": H, "W": W, "variant": variant, "batch": batch, "img_per_s": None, "oom": True})

    meta = {
        "gpu": torch.cuda.get_device_name(device),
        "cpu": next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?"),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "alpha": ALPHA, "beta": BETA, "k": K,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
