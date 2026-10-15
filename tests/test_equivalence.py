"""Check the batched PASTA against the unmodified official implementation.

1. Exact test: the reference draws its noise with np.random.normal after we seed
   numpy; we re-seed, draw the identical noise and feed it to our implementation.
   Outputs must match up to floating-point rounding.
2. Distribution test (rfft path, noise sampled internally): the symmetrised
   half-spectrum noise we sample must have the same per-frequency mean/std as the
   noise implied by the reference, and the augmented images must have the same
   per-pixel mean/std as the reference's over many draws.

Usage: python tests/test_equivalence.py [--image path] [--device cuda:0] [--out results/equivalence.json]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pasta_gpu.pasta import PASTA, official_sigma  # noqa: E402
from pasta_gpu.reference_official import PASTA as OfficialPASTA  # noqa: E402

ALPHA, BETA, K = 3.0, 0.25, 2


def exact_test(img, device, seed=0):
    """img: (3, H, W) float32 CPU in [0, 1]. Returns error stats per implementation variant."""
    _, H, W = img.shape
    np.random.seed(seed)
    ref = OfficialPASTA(ALPHA, BETA, K)(img).double()

    np.random.seed(seed)
    sigma = np.tile(official_sigma(H, W, ALPHA, BETA, K), (3, 1, 1))
    noise = np.random.normal(np.ones(sigma.shape), sigma)

    rows = {}
    for name, rfft, dtype in [
        ("fft2_fp64", False, torch.float64),
        ("rfft2_fp64", True, torch.float64),
        ("fft2_fp32", False, torch.float32),
        ("rfft2_fp32", True, torch.float32),
    ]:
        x = img[None].to(device, dtype)
        n = torch.as_tensor(noise[None], device=device, dtype=dtype)
        out = PASTA(ALPHA, BETA, K, use_rfft=rfft)(x, noise=n)[0].double().cpu()
        err = (out - ref).abs()
        ref8 = (ref * 255).round()
        out8 = (out * 255).round()
        rows[name] = {
            "max_abs_err": err.max().item(),
            "mean_abs_err": err.mean().item(),
            "uint8_pixels_differing_frac": (out8 != ref8).double().mean().item(),
            "uint8_max_diff": (out8 - ref8).abs().max().item(),
        }
    return {"H": H, "W": W, "seed": seed, "variants": rows}


def noise_distribution_test(H, W, device, n=20000, seed=0):
    """Compare our sampled half-spectrum noise with the symmetrised reference noise."""
    torch.manual_seed(seed)
    p = PASTA(ALPHA, BETA, K, use_rfft=True)
    t = p._get_tables(H, W, device, torch.float64)
    ours = PASTA.sample_half_noise((n, 3, H, W // 2 + 1), t, torch.float64)

    sigma = torch.as_tensor(np.tile(official_sigma(H, W, ALPHA, BETA, K), (3, 1, 1)), device=device)
    full = 1 + sigma * torch.randn((n, 3, H, W), device=device, dtype=torch.float64)
    theirs = PASTA.half_noise_from_full(full, t)

    # Hermitian constraint in columns that hold both f and -f
    sym_err = max(
        (ours[..., c] - ours[..., t["neg_u"], c]).abs().max().item() for c in t["edge_cols"]
    )
    std_ratio = ours.std(0) / theirs.std(0)
    mean_z = (ours.mean(0) - theirs.mean(0)) / torch.sqrt((ours.var(0) + theirs.var(0)) / n)
    return {
        "H": H, "W": W, "samples": n,
        "edge_column_symmetry_max_err": sym_err,
        "std_ratio_min": std_ratio.min().item(),
        "std_ratio_max": std_ratio.max().item(),
        "mean_abs_z_max": mean_z.abs().max().item(),
        "mean_abs_z_gt3_frac": (mean_z.abs() > 3).double().mean().item(),
    }


def output_distribution_test(img, device, n=2000, seed=0):
    """Per-pixel mean/std of augmented images: official vs ours (rfft, fp32, internal noise)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    official = OfficialPASTA(ALPHA, BETA, K)
    ref = torch.stack([official(img).float() for _ in range(n)])
    ours = PASTA(ALPHA, BETA, K, use_rfft=True)(img[None].to(device).expand(n, -1, -1, -1).contiguous()).cpu()
    z = (ours.mean(0) - ref.mean(0)) / torch.sqrt((ours.var(0) + ref.var(0)) / n + 1e-12)
    return {
        "H": img.shape[1], "W": img.shape[2], "draws": n,
        "pixel_mean_abs_z_max": z.abs().max().item(),
        "pixel_mean_abs_z_gt3_frac": (z.abs() > 3).double().mean().item(),
        "pixel_std_ratio_median": (ours.std(0) / ref.std(0).clamp_min(1e-6)).median().item(),
        "expected_gt3_frac_if_identical": 0.0027,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="test image; random if omitted")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/equivalence.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    if args.image:
        full = to_tensor(Image.open(args.image).convert("RGB"))
    else:
        full = torch.rand(3, 1052, 1914)
    crops = [full, full[:, :1051, :1913], full[:, :64, :96], full[:, :63, :95]]

    report = {"alpha": ALPHA, "beta": BETA, "k": K, "device": str(device),
              "image": Path(args.image).name if args.image else "random"}
    report["exact"] = [exact_test(c.contiguous(), device, seed=i) for i, c in enumerate(crops)]
    report["noise_distribution"] = [noise_distribution_test(h, w, device)
                                    for h, w in [(16, 24), (15, 23), (16, 23), (15, 24)]]
    report["output_distribution"] = output_distribution_test(full[:, :48, :64].contiguous(), device)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))

    ok = True
    for r in report["exact"]:
        for name, v in r["variants"].items():
            tol = 1e-5 if "fp64" in name else 1e-4
            passed = v["max_abs_err"] < tol
            ok &= passed
            print(f"exact {r['H']}x{r['W']} {name:11s} max_err={v['max_abs_err']:.2e} "
                  f"uint8_diff_frac={v['uint8_pixels_differing_frac']:.2e} {'PASS' if passed else 'FAIL'}")
    for r in report["noise_distribution"]:
        # each std ratio compares two independent n-sample estimates (log-ratio SE = 1/sqrt(n)); allow 5 SE per cell
        tol = 5 / math.sqrt(r["samples"])
        passed = r["edge_column_symmetry_max_err"] == 0 and 1 - tol < r["std_ratio_min"] \
            and r["std_ratio_max"] < 1 + tol and r["mean_abs_z_gt3_frac"] < 0.02
        ok &= passed
        print(f"noise {r['H']}x{r['W']} std_ratio=[{r['std_ratio_min']:.3f},{r['std_ratio_max']:.3f}] "
              f"|z|>3 frac={r['mean_abs_z_gt3_frac']:.4f} {'PASS' if passed else 'FAIL'}")
    r = report["output_distribution"]
    passed = r["pixel_mean_abs_z_gt3_frac"] < 0.02 and 0.95 < r["pixel_std_ratio_median"] < 1.05
    ok &= passed
    print(f"output distribution |z|>3 frac={r['pixel_mean_abs_z_gt3_frac']:.4f} "
          f"std_ratio_median={r['pixel_std_ratio_median']:.3f} {'PASS' if passed else 'FAIL'}")
    print("ALL PASS" if ok else "SOME CHECKS FAILED", "->", args.out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
