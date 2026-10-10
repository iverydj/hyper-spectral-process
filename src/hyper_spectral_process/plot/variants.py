"""Figures of a per-pixel variant selection (:class:`~hyper_spectral_process.fit.VariantSelection`)."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from ..fit.result import FitSpectrumMap
from ..fit.variants import VariantSelection
from ..io import to_image
from .maps import _edges, plot_map


def plot_variant_map(ax: plt.Axes, sel: VariantSelection, xy: np.ndarray) -> None:
    """Discrete map of the winning variant (grey: not valid in every variant)."""
    k = len(sel.names)
    cmap = ListedColormap([plt.get_cmap("tab10")(i) for i in range(k)])
    cmap.set_bad("lightgray")
    values = np.where(sel.variant_id >= 0, sel.variant_id, np.nan).astype(float)
    img, xs, ys = to_image(values, xy)
    art = ax.pcolormesh(_edges(xs), _edges(ys), np.ma.masked_invalid(img), cmap=cmap,
                        norm=BoundaryNorm(np.arange(-0.5, k), k), shading="flat")
    cb = plt.colorbar(art, ax=ax, ticks=range(k), fraction=0.046, pad=0.04)
    cb.ax.set_yticklabels(sel.names)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("best variant per pixel (BIC)")


def plot_selection_summary(sel: VariantSelection, xy: np.ndarray) -> plt.Figure:
    """Winner map, BIC-margin map, BIC histograms and wRMSE histograms."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    plot_variant_map(axes[0, 0], sel, xy)
    plot_map(axes[0, 1], xy, sel.delta_bic(), cmap="magma", percentiles=(0, 95), label="BIC margin")
    axes[0, 1].set_title("BIC(runner-up) - BIC(winner): small = ambiguous")
    for v, name in enumerate(sel.names):
        b = sel.bic[v, sel.valid_mask]
        axes[1, 0].hist(b[np.isfinite(b)], bins=80, histtype="step", lw=1.3, label=f"{name} (k = {sel.k_params[v]})")
        w = sel.wrmse[v, sel.valid_mask]
        axes[1, 1].hist(w[np.isfinite(w)], bins=80, histtype="step", lw=1.3, label=name)
    axes[1, 0].set_xlabel("BIC per pixel")
    axes[1, 1].set_xlabel("wRMSE per pixel")
    for ax in axes[1]:
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def plot_winner_means(sel: VariantSelection, fits: Mapping[str, FitSpectrumMap]) -> plt.Figure:
    """For every variant: mean data and mean fit over the pixels it wins."""
    k = len(sel.names)
    fig, axes = plt.subplots(k, 1, figsize=(10, 2.8 * k), sharex=True, squeeze=False)
    for v, (ax, name) in enumerate(zip(axes[:, 0], sel.names)):
        f = fits[name]
        win = (sel.variant_id == v) & sel.valid_mask
        if not win.any():
            ax.set_title(f"{name}: no pixel")
            continue
        y, r = f.spectra_original[win].mean(axis=0), f.recon[win].mean(axis=0)
        ax.plot(f.axis, y, color="0.45", lw=1.0, label="mean data")
        ax.plot(f.axis, r, "--", color="C0", lw=1.2, label="mean fit")
        ax.fill_between(f.axis, y - r, 0, color="C3", alpha=0.25, lw=0, label="mean residual")
        ax.set_title(f"{name}: {int(win.sum())} pixels")
        ax.legend(fontsize=7, frameon=False)
    axes[-1, 0].set_xlabel(fits[sel.names[0]].unit)
    fig.tight_layout()
    return fig


def plot_drift_maps(sel: VariantSelection, xy: np.ndarray) -> plt.Figure:
    """Largest position drift in units of ``pos_limit`` of every variant (values > 1 are penalised)."""
    k = len(sel.names)
    fig, axes = plt.subplots(1, k, figsize=(5.4 * k, 4.4), squeeze=False)
    for v, (ax, name) in enumerate(zip(axes[0], sel.names)):
        plot_map(ax, xy, np.where(sel.valid_mask, sel.drift_max[v], np.nan), cmap="inferno", vmin=0,
                 label="max |drift| / pos_limit")
        ax.set_title(name)
    fig.tight_layout()
    return fig


def save_variant_plots(sel: VariantSelection, fits: Mapping[str, FitSpectrumMap], xy: np.ndarray,
                       out_dir: str | Path, dpi: int = 150) -> list[Path]:
    """Write the selection figures to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = {"variant_selection": plot_selection_summary(sel, xy), "winner_mean_spectra": plot_winner_means(sel, fits)}
    if sel.drift_penalty_alpha > 0:
        figs["variant_drift"] = plot_drift_maps(sel, xy)
    paths = []
    for name, fig in figs.items():
        paths.append(out_dir / f"{name}.png")
        fig.savefig(paths[-1], dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return paths
