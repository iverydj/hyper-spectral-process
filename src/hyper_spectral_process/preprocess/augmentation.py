"""Dataset tools for machine learning: train/valid/test split and mixup augmentation.

These stages do not change spectra in place. ``split`` assigns every
spectrum to ``"train"``, ``"valid"`` or ``"test"`` (or to the single pool
``"all"`` when splitting is disabled); ``augmentation`` appends synthetic
spectra made inside each split from that split's original spectra only, so
no information crosses splits.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

SPLIT_NAMES = ("train", "valid", "test")


def split_counts(n: int, train: float, valid: float, test: float) -> tuple[int, int, int]:
    """Number of spectra per split (largest-remainder rounding).

    When ``n >= 3`` every split with a positive ratio receives at least one
    spectrum, taken from the largest split.
    """
    r = np.array([train, valid, test], dtype=np.float64)
    if np.any(r < 0) or not np.isclose(r.sum(), 1.0):
        raise ValueError(f"split ratios must be non-negative and sum to 1; got {tuple(r)}")
    if n <= 0:
        return 0, 0, 0
    raw = r * n
    cnt = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - cnt))[: n - int(cnt.sum())]:
        cnt[i] += 1
    if n >= 3:
        for i in range(3):
            if cnt[i] == 0 and r[i] > 0:
                j = int(np.argmax(cnt))
                if cnt[j] > 1:
                    cnt[j] -= 1
                    cnt[i] += 1
    return int(cnt[0]), int(cnt[1]), int(cnt[2])


def split_assignments(n: int, train: float, valid: float, test: float, seed: int) -> np.ndarray:
    """Random split label (``"train"``, ``"valid"``, ``"test"``) of each of ``n`` spectra."""
    n_train, n_valid, n_test = split_counts(n, train, valid, test)
    labels = np.empty(n, dtype=object)
    order = np.arange(n)
    np.random.default_rng(seed).shuffle(order)
    labels[order[:n_train]] = "train"
    labels[order[n_train:n_train + n_valid]] = "valid"
    labels[order[n_train + n_valid:n_train + n_valid + n_test]] = "test"
    return labels.astype(str)


def gaussian_kernel(sigma_points: float, radius: Optional[int] = None) -> np.ndarray:
    """Normalised Gaussian kernel of standard deviation ``sigma_points`` (default radius ``ceil(3 sigma)``)."""
    if not (math.isfinite(sigma_points) and sigma_points > 0):
        raise ValueError(f"sigma_points must be positive; got {sigma_points}")
    r = int(math.ceil(3.0 * sigma_points)) if radius is None else int(radius)
    if r < 1:
        raise ValueError("kernel radius must be >= 1")
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma_points) ** 2)
    return k / k.sum()


def lowpass_noise(m: int, rng: np.random.Generator, sigma_points: float, radius: Optional[int] = None,
                  pad_mode: str = "reflect") -> np.ndarray:
    """White noise smoothed by :func:`gaussian_kernel` and rescaled to unit standard deviation."""
    if pad_mode not in ("reflect", "edge", "symmetric", "wrap"):
        raise ValueError(f"pad_mode must be reflect, edge, symmetric or wrap; got {pad_mode!r}")
    kernel = gaussian_kernel(sigma_points, radius)
    r = kernel.size // 2
    smooth = np.convolve(np.pad(rng.normal(size=m), (r, r), mode=pad_mode), kernel, mode="valid")
    return smooth / smooth.std()


def mixup(pool: np.ndarray, n_new: int, rng: np.random.Generator, *, min_sources: int = 2, max_sources: int = 4,
          alpha_range: tuple[float, float] = (0.35, 0.65), noise_mode: str = "white", noise_std: float = 0.005,
          lowpass_sigma_points: float = 4.0, lowpass_kernel_radius: Optional[int] = None,
          lowpass_pad_mode: str = "reflect") -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Make ``n_new`` synthetic spectra as noisy convex combinations of rows of ``pool``.

    For each new spectrum: the number of sources ``k`` is uniform in
    ``[max(2, min_sources), min(max_sources, len(pool))]``; ``k`` distinct
    rows are drawn; their weights follow a symmetric Dirichlet distribution
    whose concentration is drawn uniformly from ``alpha_range``; Gaussian
    noise of standard deviation ``noise_std`` is added, either white or
    low-pass filtered (``noise_mode="lowpass"``, :func:`lowpass_noise`).

    Returns ``(spectra (n_new, M), first_source (n_new,), examples)``:
    ``first_source`` is the pool row drawn first for each new spectrum (its
    parent), ``examples`` holds the sources, weights and the spectrum before
    and after the noise for every new spectrum (for QA plots).
    """
    n_pool, m = pool.shape
    if n_pool < 2:
        raise ValueError("mixup needs at least two source spectra")
    lo, hi = float(alpha_range[0]), float(alpha_range[1])
    if not 0 < lo <= hi:
        raise ValueError(f"alpha_range must satisfy 0 < low <= high; got {alpha_range}")
    if noise_mode not in ("white", "lowpass"):
        raise ValueError(f"noise_mode must be 'white' or 'lowpass'; got {noise_mode!r}")
    k_lo = max(2, int(min_sources))
    k_hi = min(max(k_lo, int(max_sources)), n_pool)
    out = np.empty((int(n_new), m))
    first = np.empty(int(n_new), dtype=int)
    examples = []
    for j in range(int(n_new)):
        k = int(rng.integers(k_lo, k_hi + 1)) if k_hi > k_lo else k_lo
        src = rng.choice(n_pool, size=k, replace=False)
        weights = rng.dirichlet(np.full(k, rng.uniform(lo, hi)))
        mixed = weights @ pool[src]
        final = mixed.copy()
        if noise_std > 0:
            noise = (rng.normal(0.0, noise_std, size=m) if noise_mode == "white" else
                     noise_std * lowpass_noise(m, rng, lowpass_sigma_points, lowpass_kernel_radius, lowpass_pad_mode))
            final = final + noise
        out[j] = final
        first[j] = int(src[0])
        examples.append({"sources": src, "weights": weights, "mixed": mixed, "final": final})
    return out, first, examples
