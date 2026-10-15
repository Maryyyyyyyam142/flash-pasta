# FlashPASTA

**GPU-accelerated PASTA augmentation for synthetic-to-real training.**

[PASTA](https://github.com/prithv1/PASTA) (Chattopadhyay et al., ICCV 2023) improves
synthetic-to-real generalization by randomly perturbing the Fourier amplitude spectrum
of each training image, more strongly at high frequencies. The reference implementation
runs per image on the CPU inside data-loader workers, which makes the augmentation itself
the bottleneck of the input pipeline at high resolution.

FlashPASTA moves PASTA to the GPU as a batched operation and produces the same output as
the official implementation up to floating-point rounding.

## How it works

| Change | Effect |
|---|---|
| Batched on the GPU | One cuFFT call per batch instead of one CPU FFT per image |
| Cached noise mask | The per-frequency noise level is built once per image size |
| No amplitude/phase split | `abs(F) * eps * exp(i*angle(F)) == F * eps`, so `abs`, `angle` and `exp` are skipped |
| Real FFT | `Re(IFFT(F * eps)) == IRFFT(F * (eps(f) + eps(-f)) / 2)`, so half the spectrum and half the random numbers suffice |

The real-FFT path draws its noise with the variance of the symmetrised reference noise
(and keeps it Hermitian-symmetric in the self-conjugate columns), so the output
distribution matches the reference exactly. `tests/test_equivalence.py` verifies this
against the unmodified official code, both with injected noise and statistically.

## Results

Measured on an NVIDIA RTX A5000 and an AMD Threadripper PRO 3975WX (32 cores), at
Sim10k resolution (1914×1052):

| Implementation | Images / s |
|---|---|
| Official, 1 CPU core | 0.7 |
| Official, 32 CPU cores | 16 |
| FlashPASTA, 32 CPU cores | 116 |
| FlashPASTA, 1 GPU | 682 |

End-to-end data-loader and YOLO training results on Sim10k are in `results/` and on the
poster (`poster/poster.pdf`).

## Usage

```python
import torch
from pasta_gpu.pasta import PASTA

pasta = PASTA(alpha=3.0, beta=0.25, k=2)          # paper defaults
images = torch.rand(16, 3, 1052, 1914, device="cuda")  # float in [0, 1]
augmented = pasta(images)
```

For uint8 batches, `pasta_gpu.pasta.pasta_uint8(batch, pasta)` converts, augments and
converts back.

## Repository layout

| Path | Contents |
|---|---|
| `pasta_gpu/pasta.py` | Batched implementation (GPU or CPU), `fft2` and `rfft2` paths |
| `pasta_gpu/reference_official.py` | Official transform, verbatim (MIT), used as ground truth and CPU baseline |
| `pasta_gpu/yolo_trainer.py` | Ultralytics trainer with PASTA in CPU workers or on the GPU, plus timing |
| `tests/test_equivalence.py` | Exact and distributional checks against the official implementation |
| `bench/bench_kernel.py` | Throughput of the transform alone: CPU process pool vs GPU |
| `bench/bench_loader.py` | Images/s a DataLoader delivers vs number of CPU workers |
| `bench/train_yolo.py` | End-to-end YOLO training with per-epoch timing and GPU utilisation |
| `bench/plot.py`, `poster/build_poster.py` | Figures and the A0 poster, generated from `results/` |
| `data/prepare_sim10k.py` | Sim10k (Pascal VOC) to YOLO format, car class only |
| `data/make_standin.py` | Same-size placeholder images for smoke-testing the pipeline |
| `run_all.sh` | Full benchmark: loader sweep and training sweep |

## Reproducing

```bash
pip install -r requirements.txt

# 1. Correctness against the official implementation
python tests/test_equivalence.py --image path/to/gtav_syn.png --device cuda:0

# 2. Kernel throughput
python bench/bench_kernel.py --device cuda:0

# 3. Data: download repro_10k_images and repro_10k_annotations from
#    https://doi.org/10.7302/e1f1-3d97 and extract them to data/raw
python data/prepare_sim10k.py --src data/raw --dst data/sim10k

# 4. Loader and training sweeps (GPU=<physical id>), then figures and poster
GPU=0 ./run_all.sh
python bench/plot.py && python poster/build_poster.py
```

Run the benchmarks on an otherwise idle GPU and CPU; other jobs on the same machine skew
the CPU-worker numbers.

## References

1. P. Chattopadhyay, K. Sarangmath, V. Vijaykumar, J. Hoffman. *PASTA: Proportional
   Amplitude Spectrum Training Augmentation for Syn-to-Real Domain Generalization.* ICCV 2023.
2. M. Johnson-Roberson et al. *Driving in the Matrix: Can Virtual Worlds Replace
   Human-Generated Annotations for Real World Tasks?* ICRA 2017. Sim10k dataset,
   doi:10.7302/e1f1-3d97 (CC BY-NC 4.0).
3. Ultralytics YOLO26.

`pasta_gpu/reference_official.py` is copied from the authors' repository under the MIT
License, Copyright (c) 2024 Prithvijit Chattopadhyay.
