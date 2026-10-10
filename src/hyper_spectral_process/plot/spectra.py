"""Single-spectrum plots: a fit with its components, the initial guess, the loss history."""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..fit.initial_guess import InitialGuess
from ..fit.result import FitSpectrumMap


def plot_fit(ax: plt.Axes, fsm: FitSpectrumMap, idx: int, *, residual: bool = True, title: Optional[str] = None,
             names: Optional[Sequence[str]] = None) -> None:
    """Data, reconstruction and the individual peaks (filled above the baseline) of row ``idx``.

    Inactive composite slots are skipped; ``residual`` adds the residual
    (data - fit) below the data, offset by the data minimum.
    """
    if not fsm.valid_mask[idx]:
        raise ValueError(f"row {idx} was not fitted")
    x, y = fsm.axis, fsm.spectra_original[idx]
    recon = fsm.recon[idx] if fsm.recon is not None else fsm.reconstruct(idx)[0]
    comps = fsm.components(idx)[0]
    base = float(fsm.base[idx])
    names = list(names or fsm.peak_names)
    ax.plot(x, y, color="0.45", lw=0.9, label="data")
    ax.plot(x, recon, color="C0", lw=1.3, label="fit")
    for j in range(fsm.n_peaks):
        if fsm.active_mask is not None and not fsm.active_mask[idx, j]:
            continue
        ax.fill_between(x, base, comps[j] + base, alpha=0.22, color=f"C{j + 1}", lw=0)
        ax.plot(x, comps[j] + base, color=f"C{j + 1}", lw=0.9, label=f"{names[j]} ({fsm.peak_types[j]})")
    if residual:
        off = float(np.nanmin(y)) - 0.05 * float(np.ptp(y))
        ax.plot(x, y - recon + off, color="C3", lw=0.7, label="residual (offset)")
        ax.axhline(off, color="C3", lw=0.4, ls=":")
    ax.set_xlabel(fsm.unit)
    ax.set_title(title if title is not None else f"row {idx}  ({fsm.xy[idx, 0]:.3g}, {fsm.xy[idx, 1]:.3g})",
                 fontsize=9)
    ax.legend(fontsize=7, frameon=False)


def plot_initial_guess(guess: InitialGuess, *, x_label: str = "x", y_label: str = "intensity") -> plt.Figure:
    """Averaged spectrum with the model before and after the initial-guess fit, its peaks and loss history."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    ax = axes[0]
    ax.plot(guess.x, guess.y, color="0.4", lw=1.0, label=f"mean of {guess.rows.size} spectra")
    ax.plot(guess.x, guess.recon_before, "--", color="C1", lw=1.1, label="input peaks")
    ax.plot(guess.x, guess.recon, color="C0", lw=1.2, label="after fit")
    ax.set_title("initial guess")
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    ax.plot(guess.x, guess.y, color="0.6", lw=0.8)
    for j, p in enumerate(guess.peaks):
        ax.plot(guess.x, guess.components[j], lw=1.0, label=f"{p.name or f'P{j + 1}'} {p.type}: pos {p.pos:.4g}")
    ax.set_title(f"refined peaks (BIC {guess.bic:.1f})")
    ax.legend(fontsize=7, frameon=False)
    for a in axes[:2]:
        a.set_xlabel(x_label)
        a.set_ylabel(y_label)
    plot_loss_history(axes[2], guess.loss_history[None, :], guess.stage_ends)
    fig.tight_layout()
    return fig


def plot_loss_history(ax: plt.Axes, histories: np.ndarray, stage_ends: Sequence[int] = ()) -> None:
    """Mean loss per iteration over batches (with a +/- 1 SD band when there are several) and stage boundaries."""
    histories = np.atleast_2d(histories)
    it = np.arange(histories.shape[1])
    mean = histories.mean(axis=0)
    ax.plot(it, mean, color="C0", lw=1.2, label=f"mean of {histories.shape[0]} batch(es)")
    if histories.shape[0] > 1:
        sd = histories.std(axis=0)
        ax.fill_between(it, mean - sd, mean + sd, color="C0", alpha=0.25, lw=0, label="+/- 1 SD")
    for e in list(stage_ends)[:-1]:
        ax.axvline(e - 0.5, color="C1", ls="--", lw=0.8)
    if np.all(mean > 0):
        ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8, frameon=False)
