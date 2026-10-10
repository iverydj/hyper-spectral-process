"""Summary figures of a :class:`~hyper_spectral_process.fit.FitSpectrumMap`.

:func:`save_summary_plots` writes all of them; each ``plot_*`` function
returns one figure.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..fit.result import FitSpectrumMap, fit_metrics
from .maps import plot_map
from .spectra import plot_fit

PARAMS = (("pos", "coolwarm"), ("width", "plasma"), ("height", "inferno"), ("area", "magma"))


def _fitted(fsm: FitSpectrumMap) -> np.ndarray:
    return np.flatnonzero(fsm.valid_mask)


def _param(fsm: FitSpectrumMap, name: str) -> np.ndarray:
    return fsm.peak_areas() if name == "area" else getattr(fsm, name)


def plot_parameter_maps(fsm: FitSpectrumMap) -> plt.Figure:
    """Rows: peaks; columns: position, width, height and analytic area maps (2-98 % colour range)."""
    p = fsm.n_peaks
    fig, axes = plt.subplots(p, 4, figsize=(18, 3.9 * p), squeeze=False)
    for j, name in enumerate(fsm.peak_names):
        for c, (par, cmap) in enumerate(PARAMS):
            v = _param(fsm, par)[:, j].copy()
            v[~fsm.valid_mask] = np.nan
            plot_map(axes[j, c], fsm.xy, v, cmap=cmap, percentiles=(2, 98))
            axes[j, c].set_title(f"{name} ({fsm.peak_types[j]}): {par}", fontsize=9)
    fig.tight_layout()
    return fig


def plot_quality_maps(fsm: FitSpectrumMap) -> plt.Figure:
    """Maps of the normalised RMSE, R^2 and the baseline of every fitted spectrum."""
    m = fit_metrics(fsm)
    base = np.where(fsm.valid_mask, fsm.base, np.nan)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for ax, (v, t, c) in zip(axes, ((m["wnrmse"], "NRMSE", "magma"), (m["wr2"], "R$^2$", "viridis"),
                                    (base, "baseline", "coolwarm"))):
        plot_map(ax, fsm.xy, v, cmap=c, percentiles=(1, 99))
        ax.set_title(t)
    fig.tight_layout()
    return fig


def plot_parameter_histograms(fsm: FitSpectrumMap) -> plt.Figure:
    """Histograms of position, width, height and area of every peak with the median."""
    rows = _fitted(fsm)
    p = fsm.n_peaks
    fig, axes = plt.subplots(p, 4, figsize=(16, max(2.6 * p, 3.5)), squeeze=False)
    for j, name in enumerate(fsm.peak_names):
        for c, (par, _) in enumerate(PARAMS):
            v = _param(fsm, par)[rows, j]
            v = v[np.isfinite(v)]
            ax = axes[j, c]
            if v.size:
                ax.hist(v, bins=min(50, max(10, int(np.sqrt(v.size)))), color="C0", alpha=0.75)
                ax.axvline(np.median(v), color="C3", ls="--", lw=1, label=f"median {np.median(v):.4g}")
                ax.legend(fontsize=7, frameon=False)
            ax.set_title(f"{name}: {par}", fontsize=9)
    fig.tight_layout()
    return fig


def plot_average_fit(fsm: FitSpectrumMap) -> plt.Figure:
    """Mean data and mean fit over the fitted spectra, with the mean residual."""
    rows = _fitted(fsm)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    y, r = fsm.spectra_original[rows].mean(axis=0), fsm.recon[rows].mean(axis=0)
    a1.plot(fsm.axis, y, color="0.45", lw=1.0, label="mean data")
    a1.plot(fsm.axis, r, color="C0", lw=1.2, label="mean fit")
    a1.set_title(f"average over {rows.size} fitted spectra")
    a1.legend(fontsize=8, frameon=False)
    a2.plot(fsm.axis, y - r, color="C3", lw=0.8)
    a2.axhline(0, color="0.5", lw=0.5)
    a2.set_xlabel(fsm.unit)
    a2.set_ylabel("residual")
    fig.tight_layout()
    return fig


def _grid_of_fits(fsm: FitSpectrumMap, rows: np.ndarray, title: str, ncols: int = 4) -> plt.Figure:
    nrows = max(1, math.ceil(rows.size / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.3 * nrows), squeeze=False)
    for ax, i in zip(axes.flat, rows):
        plot_fit(ax, fsm, int(i))
    for ax in list(axes.flat)[rows.size:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_sample_fits(fsm: FitSpectrumMap, n: int = 16, seed: int = 42) -> plt.Figure:
    """``n`` random fitted spectra."""
    rows = _fitted(fsm)
    pick = np.sort(np.random.default_rng(seed).choice(rows, size=min(n, rows.size), replace=False))
    return _grid_of_fits(fsm, pick, f"random fitted spectra (n = {pick.size})")


def plot_worst_fits(fsm: FitSpectrumMap, n: int = 12) -> plt.Figure:
    """The ``n`` fitted spectra with the largest normalised RMSE."""
    rows = _fitted(fsm)
    nrmse = fit_metrics(fsm)["wnrmse"][rows]
    worst = np.sort(rows[np.argsort(nrmse)[::-1][:min(n, rows.size)]])
    return _grid_of_fits(fsm, worst, f"largest NRMSE (n = {worst.size})", ncols=3)


def plot_ratio_maps(fsm: FitSpectrumMap, max_pairs: int = 6) -> plt.Figure:
    """Area ratio maps of the first ``max_pairs`` peak pairs."""
    area = fsm.peak_areas()
    pairs = [(i, j) for i in range(fsm.n_peaks) for j in range(i + 1, fsm.n_peaks)][:max_pairs]
    if not pairs:
        raise ValueError("ratio maps need at least two peaks")
    ncols = min(3, len(pairs))
    nrows = math.ceil(len(pairs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.8 * nrows), squeeze=False)
    names = fsm.peak_names
    for ax, (i, j) in zip(axes.flat, pairs):
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(fsm.valid_mask & (area[:, j] > 0), area[:, i] / area[:, j], np.nan)
        plot_map(ax, fsm.xy, ratio, cmap="RdBu_r", percentiles=(2, 98))
        ax.set_title(f"area ratio {names[i]} / {names[j]}", fontsize=9)
    for ax in list(axes.flat)[len(pairs):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_position_height_scatter(fsm: FitSpectrumMap) -> plt.Figure:
    """Height against position of every peak, coloured by width."""
    rows = _fitted(fsm)
    fig, axes = plt.subplots(1, fsm.n_peaks, figsize=(5 * fsm.n_peaks, 4.2), squeeze=False)
    for j, (ax, name) in enumerate(zip(axes[0], fsm.peak_names)):
        sc = ax.scatter(fsm.pos[rows, j], fsm.height[rows, j], c=fsm.width[rows, j], s=3, alpha=0.5)
        plt.colorbar(sc, ax=ax, label="width")
        ax.set_xlabel(f"pos ({fsm.unit})")
        ax.set_ylabel("height")
        ax.set_title(name)
    fig.tight_layout()
    return fig


def save_summary_plots(fsm: FitSpectrumMap, out_dir: str | Path, *, n_samples: int = 16, n_worst: int = 12,
                       dpi: int = 150) -> list[Path]:
    """Write every summary figure of this module to ``out_dir``; returns the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = {"parameter_maps": plot_parameter_maps(fsm), "quality_maps": plot_quality_maps(fsm),
               "parameter_histograms": plot_parameter_histograms(fsm), "average_fit": plot_average_fit(fsm),
               "sample_fits": plot_sample_fits(fsm, n_samples), "worst_fits": plot_worst_fits(fsm, n_worst),
               "position_height_scatter": plot_position_height_scatter(fsm)}
    if fsm.n_peaks >= 2:
        figures["ratio_maps"] = plot_ratio_maps(fsm)
    paths = []
    for name, fig in figures.items():
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
