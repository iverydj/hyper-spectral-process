"""Write the standard outputs of a map fit to a directory."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .fit_map import MapFit
from .initial_guess import format_peaks_block
from .result import write_metrics_csv, write_params_csv


def save_fit_outputs(fit: MapFit, out_dir: str | Path, stem: str, *, summary_plots: bool = True,
                     n_samples: int = 16, n_worst: int = 12, dpi: int = 150) -> Path:
    """Write, under ``out_dir``::

        <stem>__FitSpectrumMap_v4.npz       the result (FitSpectrumMap.load)
        params_<stem>.csv, metrics_<stem>.csv
        <stem>_loss_batch_mean.png          mean loss of the batches with +/- 1 SD
        initial/                            initial-guess figure and the refined peaks as Python source
        summary/                            summary figures (:func:`~hyper_spectral_process.plot.save_summary_plots`)

    Returns the path of the NPZ file.
    """
    from ..plot import plot_initial_guess, plot_loss_history, save_summary_plots

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fsm = fit.result
    npz = out_dir / f"{stem}__FitSpectrumMap_v4.npz"
    fsm.save(npz)
    write_params_csv(fsm, out_dir / f"params_{stem}.csv")
    write_metrics_csv(fsm, out_dir / f"metrics_{stem}.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_loss_history(ax, fit.loss_history, fit.stage_ends)
    ax.set_title("fitting loss vs. iteration")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_loss_batch_mean.png", dpi=dpi)
    plt.close(fig)
    if fit.initial_guess is not None:
        (out_dir / "initial").mkdir(exist_ok=True)
        fig = plot_initial_guess(fit.initial_guess, x_label=fsm.unit)
        fig.savefig(out_dir / "initial" / f"{stem}_initial_guess.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        (out_dir / "initial" / f"PEAKS_{stem}_initial.txt").write_text(
            format_peaks_block(fit.initial_guess.peaks, conversion=fit.prepared.conversion) + "\n", encoding="utf-8")
    if summary_plots:
        save_summary_plots(fsm, out_dir / "summary", n_samples=n_samples, n_worst=n_worst, dpi=dpi)
    return npz
