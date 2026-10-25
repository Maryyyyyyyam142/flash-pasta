"""End-to-end YOLO training on Sim10k with PASTA on CPU workers vs on the GPU.

Writes per-epoch timing (train-loop time, time waiting for data, GPU utilisation,
images/sec) to --out. Validation runs on a tiny split so it does not dominate.

Example (one GPU, physical id 2):
  python bench/train_yolo.py --mode gpu --workers 4 --device 2 --imgsz 640 --out results/train/gpu_w4_640.json
Two GPUs (DDP):
  python bench/train_yolo.py --mode gpu --workers 4 --device 1,2 --batch 32 ...

--device takes torch device indices, which are the nvidia-smi ids when
CUDA_VISIBLE_DEVICES is unset. Only those GPUs are made visible to the run, because
Ultralytics caps data-loader workers at cpu_count // visible GPUs (64 // 4 = 16 on our node).
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["none", "cpu_official", "cpu_ours", "gpu"], required=True)
    ap.add_argument("--data", default=str(ROOT / "data/sim10k.yaml"))
    ap.add_argument("--model", default=str(ROOT / "weights/yolo26s.pt"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--device", default="2")
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ.update({
        "PASTA_MODE": args.mode, "PASTA_ALPHA": str(args.alpha), "PASTA_BETA": str(args.beta),
        "PASTA_K": str(args.k), "PASTA_TIMING_OUT": str(Path(args.out).resolve()),
    })
    # Hide the GPUs we don't train on (before CUDA initialises) so the worker cap doesn't apply
    ids = [int(i) for i in args.device.replace("cuda:", "").split(",")]
    visible = os.getenv("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible.split(",")[i] if visible else str(i) for i in ids)
    device = ",".join(str(i) for i in range(len(ids)))

    from ultralytics import YOLO

    from flashpasta.yolo_trainer import PastaDetectionTrainer

    name = Path(args.out).stem
    YOLO(args.model).train(
        trainer=PastaDetectionTrainer,
        data=args.data, imgsz=args.imgsz, batch=args.batch, epochs=args.epochs, workers=args.workers,
        device=device, fraction=args.fraction,
        mosaic=0.0, close_mosaic=0,  # identical per-image inputs for cpu and gpu PASTA
        cache=False, plots=False, save=False, val=False, amp=True, deterministic=False,
        project=str(ROOT / "runs"), name=name, exist_ok=True, verbose=False,
    )


if __name__ == "__main__":
    main()
