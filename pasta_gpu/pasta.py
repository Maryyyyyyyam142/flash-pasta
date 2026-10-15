"""Batched PASTA augmentation that runs on any torch device (GPU or CPU).

It computes the same thing as the official transform (pasta_gpu/reference_official.py)
but restructured for throughput:

* Batched: one FFT call for a whole (B, C, H, W) batch instead of one per image.
* Cached sigma mask: the frequency-dependent noise std is built once per (H, W).
* No amplitude/phase split: amp * eps * exp(i*phase) == spectrum * eps for real eps,
  so we skip abs/angle/exp entirely.
* Real FFT (use_rfft=True): the reference keeps only the real part of the inverse
  FFT, which equals the inverse FFT of the spectrum multiplied by the symmetrised
  noise (eps[f] + eps[-f]) / 2. That spectrum is Hermitian, so rfft2/irfft2 give
  the identical result with half the FFT work and half the random numbers.
  Noise for the stored half-spectrum is drawn with the matching std, so the output
  distribution equals the reference; with injected noise the output is identical
  up to floating-point rounding (see tests/test_equivalence.py).
"""
import math

import numpy as np
import torch


def official_sigma(H: int, W: int, alpha: float, beta: float, k: int) -> np.ndarray:
    """Noise std per frequency, in unshifted FFT layout, exactly as the official code builds it."""

    def centred(n):
        if n % 2 == 1:
            return np.arange(-1 * (n // 2), (n // 2) + 1)
        return np.concatenate([np.arange(-1 * (n // 2) + 1, 1), np.arange(0, n // 2)])

    XX, YY = np.meshgrid(centred(W), centred(H))
    inv = np.sqrt(np.square(XX) + np.square(YY))
    inv *= (1 / inv.max()) * alpha
    inv = np.power(inv, k)
    inv += beta
    return np.fft.fftshift(inv, axes=[-2, -1])


class PASTA(torch.nn.Module):
    """PASTA on a batch of images in [0, 1], shape (B, C, H, W), on the tensor's device."""

    def __init__(self, alpha: float = 3.0, beta: float = 0.25, k: int = 2, use_rfft: bool = True):
        super().__init__()
        self.alpha, self.beta, self.k = alpha, beta, k
        self.use_rfft = use_rfft
        self._tables = {}

    def _get_tables(self, H, W, device, dtype):
        key = (H, W, str(device), dtype)
        if key in self._tables:
            return self._tables[key]

        sigma = official_sigma(H, W, self.alpha, self.beta, self.k)
        neg_u = (-np.arange(H)) % H
        neg_v = (-np.arange(W)) % W
        t = {
            "sigma": torch.as_tensor(sigma, dtype=dtype, device=device),
            "neg_u": torch.as_tensor(neg_u, device=device),
            "neg_v": torch.as_tensor(neg_v, device=device),
        }
        if self.use_rfft:
            Wh = W // 2 + 1
            u = np.arange(H)[:, None]
            v = np.arange(Wh)[None, :]
            self_conj = (neg_u[:, None] == u) & (neg_v[None, :Wh] == v)
            sigma_neg = sigma[neg_u][:, neg_v][:, :Wh]
            std = np.where(self_conj, sigma[:, :Wh], np.sqrt(sigma[:, :Wh] ** 2 + sigma_neg ** 2) / 2)
            # Columns 0 and W/2 hold both f and -f, so their noise must be symmetric in u.
            t["std"] = torch.as_tensor(std, dtype=dtype, device=device)
            t["edge_cols"] = [0] + ([W // 2] if W % 2 == 0 else [])
            t["self_conj_rows"] = torch.as_tensor(neg_u == np.arange(H), device=device)
        self._tables[key] = t
        return t

    @torch.no_grad()
    def forward(self, x: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """x: (B, C, H, W) float in [0, 1].

        noise: optional full-spectrum multiplicative noise (B, C, H, W), i.e. the
        N(1, sigma) draw the reference makes; used only by the equivalence tests.
        """
        B, C, H, W = x.shape
        t = self._get_tables(H, W, x.device, x.dtype)

        if not self.use_rfft:
            spec = torch.fft.fft2(x)
            if noise is None:
                noise = torch.randn(x.shape, device=x.device, dtype=x.dtype).mul_(t["sigma"]).add_(1)
            return torch.fft.ifft2(spec.mul_(noise)).real.clamp_(0, 1)

        spec = torch.fft.rfft2(x)
        eps = self.sample_half_noise(spec.shape, t, x.dtype) if noise is None else self.half_noise_from_full(noise, t)
        return torch.fft.irfft2(spec.mul_(eps), s=(H, W)).clamp_(0, 1)

    @staticmethod
    def sample_half_noise(shape, t, dtype):
        """Symmetrised noise for the rfft half-spectrum, drawn directly with the matching std."""
        z = torch.randn(shape, device=t["std"].device, dtype=dtype)
        for c in t["edge_cols"]:
            col = z[..., c]
            z[..., c] = torch.where(t["self_conj_rows"], col, (col + col[..., t["neg_u"]]) * math.sqrt(0.5))
        return z.mul_(t["std"]).add_(1)

    @staticmethod
    def half_noise_from_full(noise, t):
        """(eps[f] + eps[-f]) / 2 restricted to the rfft half-spectrum."""
        W = noise.shape[-1]
        mirrored = noise[..., t["neg_u"], :][..., t["neg_v"]]
        return 0.5 * (noise + mirrored)[..., : W // 2 + 1]


def pasta_uint8(x: torch.Tensor, pasta: PASTA) -> torch.Tensor:
    """Apply PASTA to a uint8 (…, C, H, W) tensor and return uint8, as a data pipeline would store it."""
    batched = x.dim() == 4
    xf = (x if batched else x[None]).float().div_(255)
    out = pasta(xf).mul_(255).round_().to(torch.uint8)
    return out if batched else out[0]
