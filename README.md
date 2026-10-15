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

### YOLO training on Sim10k

YOLO26s, batch 16 (8 at 1280 px), one RTX A5000, mixed precision, mosaic off so every mode
sees identical inputs. Steady-state epoch; "GPU waiting" is the share of the training loop
spent waiting for the next batch.

| Setup | Official, CPU workers | FlashPASTA, GPU | Speed-up |
|---|---|---|---|
| 640 px, 8 workers | 41.6 img/s, GPU waiting 77% | 173.8 img/s, GPU waiting 0.2% | 4.2× |
| 640 px, 16 workers | 78.5 img/s, GPU waiting 55% | 173.8 img/s, GPU waiting 0.3% | 2.2× |
| 1280 px, 8 workers | 7.1 img/s, GPU waiting 85% | 46.0 img/s, GPU waiting 0.1% | 6.4× |

Without PASTA the same model trains at about 176 img/s (640 px) and 48 img/s (1280 px), so
FlashPASTA costs about 1–5% of training throughput, while the official implementation never
reaches the GPU's rate even with all 32 cores.

### Case study: Red Sea coral reefs (Marjan dataset)

Same setup on 9,498 real 640×640 underwater images of healthy, bleached and dead coral [4],
8 CPU workers: the official implementation trains at 46.9 img/s with the GPU waiting for data
74% of the time; FlashPASTA trains at 169.1 img/s with 0.2% waiting (3.6×), cutting an epoch
from 3.4 to 0.9 minutes.

All raw numbers are in `results/` and on the poster (`poster/poster.pdf`).

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
| `run_all.sh` | Sim10k benchmark: loader sweep and training sweep |
| `run_marjan.sh` | Training-throughput sweep on the Marjan coral dataset |
| `poster/make_assets.py` | Example images and the QR code used on the poster |

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

# 4. Loader and training sweeps (GPU=<physical id>)
GPU=0 ./run_all.sh

# 5. Optional: Marjan coral dataset (YOLO export from Roboflow Universe,
#    maha-bil6y/marjan-detection v3) extracted to data/marjan, with data/marjan.yaml
GPU=0 ./run_marjan.sh

# 6. Figures and poster
python bench/plot.py && python poster/make_assets.py && python poster/build_poster.py
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
4. A. Aldhahri, E. Saif, H. Ali, M. Alsayed, F. Alshareef. *Harnessing Computer Vision and Deep
   Learning to Monitor Coral Reef Health.* ETASR 15(4), 2025, doi:10.48084/etasr.10324.
   Marjan dataset, Roboflow Universe (CC BY 4.0).

`pasta_gpu/reference_official.py` is copied from the authors' repository under the MIT
License, Copyright (c) 2024 Prithvijit Chattopadhyay.
