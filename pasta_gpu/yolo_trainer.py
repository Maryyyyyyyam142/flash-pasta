"""Ultralytics DetectionTrainer with PASTA on CPU workers or on the GPU, plus timing.

Configured through environment variables so the settings reach DDP worker processes:
  PASTA_MODE      none | cpu_official | cpu_ours | gpu   (default none)
  PASTA_ALPHA/BETA/K                 (default 3.0 / 0.25 / 2)
  PASTA_TIMING_OUT                   JSON path for per-epoch timing (rank 0 only)

All modes apply PASTA at the same point: per image, after all other augmentations
(letterbox, perspective, HSV, flip). cpu_* = last step of each DataLoader worker's
transform pipeline (official code, or our implementation on 1 CPU thread);
gpu = first step after the batch is moved to the device.
Run with mosaic=0 so all modes see identical per-image inputs.
"""
import json
import os
import threading
import time
from pathlib import Path

import torch
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import RANK

from pasta_gpu.pasta import PASTA, pasta_uint8
from pasta_gpu.reference_official import PASTA as OfficialPASTA


def _pasta_from_env():
    return PASTA(float(os.getenv("PASTA_ALPHA", 3.0)), float(os.getenv("PASTA_BETA", 0.25)),
                 int(os.getenv("PASTA_K", 2)), use_rfft=True)


class CPUPastaStep:
    """Last transform in a worker: uint8 CHW tensor -> PASTA -> uint8 CHW tensor."""

    def __init__(self, impl):
        self.impl = impl
        if impl == "official":
            self.pasta = OfficialPASTA(float(os.getenv("PASTA_ALPHA", 3.0)), float(os.getenv("PASTA_BETA", 0.25)),
                                       int(os.getenv("PASTA_K", 2)))
        else:
            self.pasta = _pasta_from_env()

    def __call__(self, labels):
        img = labels["img"]
        if self.impl == "official":
            labels["img"] = self.pasta(img.float().div(255)).mul(255).round().to(torch.uint8)
        else:
            labels["img"] = pasta_uint8(img, self.pasta)
        return labels


class PastaYOLODataset(YOLODataset):
    def build_transforms(self, hyp=None):
        transforms = super().build_transforms(hyp)
        if self.augment:
            transforms.append(CPUPastaStep(os.getenv("PASTA_MODE").removeprefix("cpu_")))
        return transforms


class GPUUtilSampler(threading.Thread):
    """Samples NVML utilisation of the training GPUs every `period` seconds."""

    def __init__(self, device_arg, period=0.1):
        super().__init__(daemon=True)
        import pynvml

        pynvml.nvmlInit()
        self.nvml = pynvml
        # --device holds torch indices; map them to NVML (physical) ids through CUDA_VISIBLE_DEVICES if set
        torch_ids = [int(i) for i in str(device_arg).replace("cuda:", "").split(",") if i.strip().isdigit()] or [0]
        visible = os.getenv("CUDA_VISIBLE_DEVICES")
        physical = [int(visible.split(",")[i]) for i in torch_ids] if visible else torch_ids
        self.physical_ids = physical
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical]
        self.period, self.samples, self.running = period, [], True

    def run(self):
        while self.running:
            self.samples.append([self.nvml.nvmlDeviceGetUtilizationRates(h).gpu for h in self.handles])
            time.sleep(self.period)

    def pop_mean(self):
        s, self.samples = self.samples, []
        return [sum(col) / len(col) for col in zip(*s)] if s else []


class PastaDetectionTrainer(DetectionTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pasta_mode = os.getenv("PASTA_MODE", "none")
        assert self.pasta_mode in {"none", "cpu_official", "cpu_ours", "gpu"}, self.pasta_mode
        self.gpu_pasta = _pasta_from_env() if self.pasta_mode == "gpu" else None
        self._t = {}
        self._epochs = []
        self._pasta_events = []
        for event in ["on_train_epoch_start", "on_train_batch_start", "on_train_batch_end",
                      "on_train_epoch_end", "on_train_end"]:
            self.add_callback(event, getattr(self, f"_{event}"))

    def build_dataset(self, img_path, mode="train", batch=None):
        dataset = super().build_dataset(img_path, mode, batch)
        if mode == "train" and self.pasta_mode.startswith("cpu_"):
            assert type(dataset) is YOLODataset, type(dataset)
            dataset.__class__ = PastaYOLODataset
            dataset.transforms = dataset.build_transforms(hyp=self.args)
        return dataset

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        if self.gpu_pasta is not None:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.autocast(device_type="cuda", enabled=False):
                batch["img"] = self.gpu_pasta(batch["img"].float())
            end.record()
            self._pasta_events.append((start, end))
        return batch

    # --- timing callbacks (each receives the trainer, i.e. self) ---
    def _on_train_epoch_start(self, _):
        if RANK in {-1, 0} and not hasattr(self, "_sampler"):
            self._sampler = GPUUtilSampler(self.args.device)
            self._sampler.start()
        if hasattr(self, "_sampler"):
            self._sampler.pop_mean()
        torch.cuda.synchronize(self.device)
        now = time.perf_counter()
        self._t = {"epoch_start": now, "last_end": now, "wait": 0.0, "step": 0.0, "batches": 0, "images": 0}

    def _on_train_batch_start(self, _):
        now = time.perf_counter()
        self._t["wait"] += now - self._t["last_end"]
        self._t["batch_start"] = now

    def _on_train_batch_end(self, _):
        torch.cuda.synchronize(self.device)
        now = time.perf_counter()
        self._t["step"] += now - self._t["batch_start"]
        self._t["last_end"] = now
        self._t["batches"] += 1
        self._t["images"] += self.batch_size

    def _on_train_epoch_end(self, _):
        if RANK not in {-1, 0}:
            return
        torch.cuda.synchronize(self.device)
        t = self._t
        train_s = t["last_end"] - t["epoch_start"]
        pasta_ms = sum(s.elapsed_time(e) for s, e in self._pasta_events)
        self._pasta_events = []
        world = max(1, int(os.getenv("WORLD_SIZE", 1)))
        self._epochs.append({
            "epoch": self.epoch + 1,
            "train_loop_s": train_s,
            "data_wait_s": t["wait"],
            "step_s": t["step"],
            "data_wait_frac": t["wait"] / train_s,
            "batches": t["batches"],
            "img_per_s_global": t["images"] * world / train_s,
            "gpu_pasta_s_rank0": pasta_ms / 1000,
            "gpu_util_mean": self._sampler.pop_mean() if hasattr(self, "_sampler") else [],
        })
        print("PASTA_TIMING", json.dumps(self._epochs[-1]), flush=True)

    def _on_train_end(self, _):
        if RANK not in {-1, 0}:
            return
        if hasattr(self, "_sampler"):
            self._sampler.running = False
        out = os.getenv("PASTA_TIMING_OUT")
        if out:
            meta = {
                "pasta_mode": self.pasta_mode,
                "workers": self.args.workers, "batch": self.batch_size, "imgsz": self.args.imgsz,
                "model": Path(str(self.args.model)).name, "device": str(self.args.device),
                "world_size": max(1, int(os.getenv("WORLD_SIZE", 1))), "fraction": self.args.fraction,
                "gpu": torch.cuda.get_device_name(self.device),
                "gpu_physical_ids": getattr(getattr(self, "_sampler", None), "physical_ids", None), "alpha": os.getenv("PASTA_ALPHA", "3.0"),
                "beta": os.getenv("PASTA_BETA", "0.25"), "k": os.getenv("PASTA_K", "2"),
            }
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(json.dumps({"meta": meta, "epochs": self._epochs}, indent=2))
