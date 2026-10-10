"""Synthetic spectrum maps with known ground truth (for tests, documentation and trying settings).

Peaks of any supported type vary smoothly across a raster map; a quadratic
baseline, Gaussian noise and optional cosmic-ray spikes are added.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .fit.lineshapes import fano, pseudo_voigt, pseudo_voigt_tch
from .io import StandardSpectrumMap


@dataclass(frozen=True)
class SyntheticPeak:
    """A peak of a synthetic map (parameter meaning as in :class:`~hyper_spectral_process.fit.PeakSpec`).

    ``pos``, ``width``, ``height`` are scalars or ``(N,)`` arrays; on top of
    them ``pos_shift`` adds a smooth spatial variation of up to
    ``+/- pos_shift`` and ``width_change`` / ``height_change`` multiply by
    ``1 +/- change``. ``eta`` (fixed pseudo-Voigt), ``width_g`` (TCH) and ``q``
    (Fano) are needed for those types.
    """

    type: str
    pos: object
    width: object
    height: object
    eta: Optional[float] = None
    width_g: Optional[float] = None
    q: Optional[float] = None
    pos_shift: float = 0.0
    width_change: float = 0.0
    height_change: float = 0.0
    name: str = ""


@dataclass(frozen=True)
class SyntheticMap:
    """A generated map and its truth (arrays aligned with the rows of ``ssm``)."""

    ssm: StandardSpectrumMap
    shape: tuple                 # (ny, nx); x varies fastest
    truth: dict                  # "pos", "width", "height": (N, P)
    peaks_clean: np.ndarray      # (N, M) noise-free sum of peaks
    baseline: np.ndarray         # (N, M)
    spike_mask: np.ndarray       # (N, M)
    extra: dict = field(default_factory=dict)


def smooth_field(ny: int, nx: int, rng: np.random.Generator, n_blobs: int = 4) -> np.ndarray:
    """Random smooth field in [-1, 1] on a ``(ny, nx)`` grid: a gradient plus Gaussian blobs."""
    yy, xx = np.mgrid[0:ny, 0:nx]
    u, v = xx / max(nx - 1, 1), yy / max(ny - 1, 1)
    a = rng.normal(size=2)
    f = a[0] * (u - 0.5) + a[1] * (v - 0.5)
    for _ in range(n_blobs):
        cx, cy = rng.uniform(0, 1, size=2)
        r = rng.uniform(0.15, 0.4)
        f = f + rng.normal() * np.exp(-((u - cx) ** 2 + (v - cy) ** 2) / (2 * r * r))
    f = f - 0.5 * (f.max() + f.min())
    peak = np.abs(f).max()
    return f / peak if peak > 0 else f


def profile(peak_type: str, x, pos, width, *, eta=None, width_g=None, q=None):
    """Unit-height profile of ``peak_type`` (see :mod:`~hyper_spectral_process.fit.lineshapes`)."""
    if peak_type == "gaussian":
        return pseudo_voigt(x, pos, width, 0.0)
    if peak_type == "lorentzian":
        return pseudo_voigt(x, pos, width, 1.0)
    if peak_type == "pseudovoigt_fixed":
        return pseudo_voigt(x, pos, width, eta)
    if peak_type == "pseudovoigt":
        return pseudo_voigt_tch(x, pos, width, width_g)
    if peak_type == "fano":
        return fano(x, pos, width, q)
    raise ValueError(f"unknown peak type {peak_type!r}")


def synthetic_map(axis: np.ndarray, peaks: Sequence[SyntheticPeak], *, shape: tuple = (24, 24),
                  noise_std: float = 0.01, baseline: tuple = (0.0, 0.0, 0.0), spikes_per_spectrum: float = 0.0,
                  spike_height: tuple = (5.0, 20.0), spike_width: tuple = (1, 2), step: float = 1.0,
                  unit: str = "cm^-1", seed: int = 0) -> SyntheticMap:
    """Generate a ``shape = (ny, nx)`` raster map on ``axis``.

    The baseline is ``b0 + b1 t + b2 t^2`` with ``t`` the axis mapped onto
    [-1, 1]; spikes are runs of ``spike_width`` adjacent samples of height
    uniform in ``spike_height`` (Poisson count per spectrum); coordinates are
    a raster of spacing ``step`` with x varying fastest.
    """
    rng = np.random.default_rng(seed)
    axis = np.asarray(axis, dtype=np.float64)
    ny, nx = shape
    n, m = ny * nx, axis.size
    xy = np.column_stack([np.tile(np.arange(nx), ny) * step, np.repeat(np.arange(ny), nx) * step]).astype(float)

    def per_spectrum(v):
        v = np.asarray(v, dtype=np.float64)
        return np.full(n, float(v)) if v.ndim == 0 else v.copy()

    truth = {k: np.zeros((n, len(peaks))) for k in ("pos", "width", "height")}
    clean = np.zeros((n, m))
    for j, pk in enumerate(peaks):
        c = per_spectrum(pk.pos) + pk.pos_shift * smooth_field(ny, nx, rng).ravel()
        w = per_spectrum(pk.width) * (1.0 + pk.width_change * smooth_field(ny, nx, rng).ravel())
        h = per_spectrum(pk.height) * (1.0 + pk.height_change * smooth_field(ny, nx, rng).ravel())
        truth["pos"][:, j], truth["width"][:, j], truth["height"][:, j] = c, w, h
        clean += h[:, None] * profile(pk.type, axis[None, :], c[:, None], w[:, None], eta=pk.eta,
                                      width_g=pk.width_g, q=pk.q)
    t = (axis - 0.5 * (axis[0] + axis[-1])) / (0.5 * (axis[-1] - axis[0]))
    base = np.tile(sum(b * t**k for k, b in enumerate(baseline)), (n, 1))
    spikes = np.zeros((n, m))
    mask = np.zeros((n, m), dtype=bool)
    if spikes_per_spectrum > 0:
        for i, count in enumerate(rng.poisson(spikes_per_spectrum, size=n)):
            for _ in range(count):
                k = int(rng.integers(2, m - 2 - spike_width[1]))
                wdt = int(rng.integers(spike_width[0], spike_width[1] + 1))
                spikes[i, k:k + wdt] += rng.uniform(*spike_height)
                mask[i, k:k + wdt] = True
    noisy = clean + base + spikes + rng.normal(0.0, noise_std, size=(n, m))
    return SyntheticMap(ssm=StandardSpectrumMap(noisy, xy, axis, unit), shape=(ny, nx), truth=truth,
                        peaks_clean=clean, baseline=base, spike_mask=mask)
