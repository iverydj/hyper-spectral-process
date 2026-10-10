"""Quality-assurance plots for preprocessing.

Every function returns a :class:`matplotlib.figure.Figure` (the caller saves
or shows it). :mod:`.runner` writes them for every stage of a run.
"""
from __future__ import annotations

import math
import re
from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..io import StandardSpectrumMap

#: y-axis scale of each stage panel in :func:`plot_trace` (others: linear).
LOG_STAGES = ("spike", "spike1", "spike2", "norm")


def _xy(x: np.ndarray, y: np.ndarray, yscale: str) -> tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(y) & ((y > 0) if yscale == "log" else True)
    return x[m], y[m]


def sample_rows(n_total: int, n: int, seed: Optional[int]) -> np.ndarray:
    """``min(n, n_total)`` distinct random rows, sorted."""
    return np.sort(np.random.default_rng(seed).choice(n_total, size=min(n, n_total), replace=False))


def plot_spectrum_grid(before: StandardSpectrumMap, after: StandardSpectrumMap,
                       extra: Optional[StandardSpectrumMap] = None, *, n: int = 9, seed: Optional[int] = 42,
                       yscale: str = "linear", labels: Sequence[str] = ("before", "after", "extra"),
                       suptitle: str = "") -> plt.Figure:
    """3 x 3 grid of random spectra before and after a stage (and an optional third curve, e.g. a baseline)."""
    if yscale not in ("linear", "log"):
        raise ValueError("yscale must be 'linear' or 'log'")
    rows = sample_rows(before.n_spectra, n, seed)
    ncols = 3
    nrows = max(1, math.ceil(rows.size / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False)
    for ax, i in zip(axes.flat, rows):
        ax.set_yscale(yscale)
        ax.plot(*_xy(before.axis, before.spectra[i], yscale), lw=1.1, label=labels[0])
        if extra is not None:
            ax.plot(*_xy(extra.axis, extra.spectra[i], yscale), lw=1.0, ls=":", label=labels[2])
        ax.plot(*_xy(after.axis, after.spectra[i], yscale), lw=1.1, ls="--", label=labels[1])
        ax.set_title(f"row {int(i)}  (x={before.xy[i, 0]:.3g}, y={before.xy[i, 1]:.3g})", fontsize=9)
        ax.set_xlabel(f"axis ({before.unit})" if before.unit else "axis", fontsize=8)
        ax.grid(True, alpha=0.3, lw=0.6)
        ax.legend(fontsize=7, frameon=False)
    for ax in list(axes.flat)[rows.size:]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def plot_trace(stages: Mapping[str, StandardSpectrumMap], order: Sequence[str], idx: int, *,
               baseline_curve: Optional[np.ndarray] = None, despike_reference: Optional[np.ndarray] = None,
               ncols: int = 2) -> plt.Figure:
    """Every stage of one spectrum: each panel overlays the previous (solid) and current (dashed) stage.

    The baseline panel adds the baseline curve, the ``spike2`` panel the
    reference of the residual pass; derivative panels use a second y axis.
    """
    nrows = math.ceil(len(order) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 3.2 * nrows), squeeze=False)
    xy = stages[order[0]].xy[idx]
    for p, (ax, key) in enumerate(zip(axes.flat, order)):
        cur = stages[key]
        prev = stages[order[p - 1]] if p else cur
        yscale = "log" if key in LOG_STAGES else "linear"
        ax.set_yscale(yscale)
        ax.plot(*_xy(prev.axis, prev.spectra[idx], yscale), lw=1.1, color="tab:blue",
                label=f"before ({order[p - 1]})" if p else "source")
        if key == "baseline" and baseline_curve is not None:
            ax.plot(*_xy(prev.axis, baseline_curve[idx], yscale), lw=1.0, ls=":", color="tab:red", label="baseline")
        if key == "spike2" and despike_reference is not None:
            ax.plot(*_xy(prev.axis, despike_reference[idx], yscale), lw=1.0, ls=":", color="tab:green",
                    label="reference")
        target = ax
        if p:
            target = ax.twinx() if key.startswith("deriv") else ax
            target.plot(*_xy(cur.axis, cur.spectra[idx], yscale), lw=1.1, ls="--", color="tab:orange",
                        label=f"after ({key})")
            if target is not ax:
                target.tick_params(axis="y", colors="tab:orange", labelsize=7)
        ax.set_title(f"{p + 1}) {key}", fontsize=10)
        ax.grid(True, alpha=0.3, lw=0.6)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = target.get_legend_handles_labels() if target is not ax else ([], [])
        ax.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False)
    for ax in list(axes.flat)[len(order):]:
        ax.axis("off")
    fig.suptitle(f"row {idx}  (x={xy[0]:.3g}, y={xy[1]:.3g})")
    fig.tight_layout()
    return fig


def population_snr(spectra: np.ndarray) -> dict:
    """Signal-to-noise estimate that treats the mean spectrum as the signal.

    ``snr_db = 10 log10(var_axis(mean) / mean_axis(var))`` and
    ``peak_snr_db = 20 log10(max |mean| / mean_axis(std))``; meaningful for a
    population of nominally identical spectra (needs at least two).
    """
    out = {"n": int(spectra.shape[0]), "snr_db": math.nan, "peak_snr_db": math.nan,
           "signal_std": math.nan, "noise_std": math.nan}
    if spectra.shape[0] < 2:
        return out
    mu = spectra.mean(axis=0)
    var = spectra.var(axis=0)
    p_signal, p_noise = float(mu.var()), float(var.mean())
    if p_noise > 0:
        if p_signal > 0:
            out["snr_db"] = 10.0 * math.log10(p_signal / p_noise)
        out["peak_snr_db"] = 20.0 * math.log10(float(np.abs(mu).max()) / float(np.sqrt(var).mean()))
        out["signal_std"], out["noise_std"] = math.sqrt(p_signal), math.sqrt(p_noise)
    return out


def plot_overview(ssm: StandardSpectrumMap, title: str = "", *, n: int = 20, seed: int = 42,
                  reference: Optional[tuple[np.ndarray, np.ndarray]] = None,
                  reference_label: str = "reference") -> plt.Figure:
    """``n`` random spectra, the mean spectrum and the :func:`population_snr` numbers.

    ``reference = (axis, curve)`` adds a dashed reference curve.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    for i in sample_rows(ssm.n_spectra, n, seed):
        ax.plot(ssm.axis, ssm.spectra[i], lw=0.6, alpha=0.45)
    ax.plot(ssm.axis, ssm.spectra.mean(axis=0), color="black", lw=1.2, label="mean")
    if reference is not None:
        ax.plot(reference[0], reference[1], color="tab:red", ls="--", lw=1.2, label=reference_label)
    snr = population_snr(ssm.spectra)
    lines = [f"N = {snr['n']}"]
    if math.isfinite(snr["snr_db"]):
        lines.append(f"SNR(power) = {snr['snr_db']:+.1f} dB")
    if math.isfinite(snr["peak_snr_db"]):
        lines.append(f"SNR(peak) = {snr['peak_snr_db']:+.1f} dB")
    ax.text(0.985, 0.985, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=9,
            family="monospace", bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.6"))
    ax.set_xlabel(f"axis ({ssm.unit})" if ssm.unit else "axis")
    ax.set_ylabel("intensity")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, ls=":", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_total_intensity(ssm: StandardSpectrumMap, title: str = "") -> plt.Figure:
    """Map of the summed intensity of every spectrum."""
    from ..plot.maps import plot_map

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    plot_map(ax, ssm.xy, ssm.spectra.sum(axis=1), label="total intensity")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_mixup_examples(before: StandardSpectrumMap, examples: Sequence[dict], *, n: int = 9) -> plt.Figure:
    """Sources (faint, with their weights), their mixture (black) and the noisy result (red)."""
    shown = list(examples)[:n]
    ncols = 3
    nrows = max(1, math.ceil(len(shown) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False)
    for k, (ax, ex) in enumerate(zip(axes.flat, shown)):
        for s, w in zip(ex["sources"], ex["weights"]):
            ax.plot(before.axis, before.spectra[int(s)], lw=1.0, alpha=0.35, label=f"row {int(s)} (w={w:.2f})")
        ax.plot(before.axis, ex["mixed"], "k--", lw=1.4, label="mixture")
        ax.plot(before.axis, ex["final"], color="tab:red", lw=1.2, label="with noise")
        ax.set_title(f"mixup {k + 1} ({ex.get('split', '')}, k={len(ex['sources'])})", fontsize=9)
        ax.legend(fontsize=6, frameon=False)
    for ax in list(axes.flat)[len(shown):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_cross_file_comparison(results: Mapping[str, StandardSpectrumMap]) -> plt.Figure:
    """Mean processed spectrum of every input file, on linear and log scales."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5))
    for name, obj in results.items():
        mean = obj.spectra.mean(axis=0)
        a1.plot(obj.axis, mean, lw=1.2, label=name)
        a2.plot(*_xy(obj.axis, mean, "log"), lw=1.2, label=name)
    a2.set_yscale("log")
    for ax, t in ((a1, "mean processed spectrum"), (a2, "mean processed spectrum (log)")):
        ax.set_title(t)
        ax.legend(fontsize=8)
        ax.grid(True, ls=":", alpha=0.3)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------- t-SNE
def stage_display_name(key: str) -> str:
    names = {"source": "Source", "roi": "ROI", "spike": "Despike", "spike1": "Despike pass 1",
             "spike2": "Despike pass 2", "augment": "Augmentation", "norm": "Normalize", "smooth": "Smooth",
             "smooth2nd": "Smooth (2nd)", "baseline": "Baseline", "interp": "Interpolate"}
    m = re.fullmatch(r"deriv(\d+)", key)
    return f"Derivative d{m.group(1)}" if m else names.get(key, key)


def default_tsne_keys(stage_order: Sequence[str], stages: Mapping[str, StandardSpectrumMap],
                      deriv_orders: Sequence[int]) -> list[str]:
    """``source``, the output of every listed stage and every derivative order."""
    keys = ["source"]
    for name in stage_order:
        if name == "split":
            continue
        if name == "despike":
            keys.append("spike2" if "spike2" in stages else "spike")
        elif name == "deriv":
            keys += [f"deriv{int(k)}" for k in deriv_orders]
        else:
            keys.append({"augmentation": "augment", "normalize": "norm"}.get(name, name))
    return list(dict.fromkeys(keys))


def tsne_embeddings(sources: Sequence[tuple[str, Mapping[str, StandardSpectrumMap]]], stage_keys: Sequence[str], *,
                    perplexity: float, max_iter: int = 1000, random_state: int = 123, n_jobs: int = 1,
                    thread_limit: int = 1) -> dict:
    """Class-wise t-SNE of every stage, all input files pooled.

    ``sources`` is a list of ``(class_name, stages)`` per input file. Each
    stage is embedded independently (its own axis). Returns
    ``{stage_key: (coords (N, 2), labels (N,), kl_divergence)}``. Requires
    scikit-learn.
    """
    from sklearn.manifold import TSNE
    from threadpoolctl import threadpool_limits

    out = {}
    for key in stage_keys:
        blocks, labels, axis0 = [], [], None
        for class_name, stages in sources:
            if key not in stages:
                raise ValueError(f"t-SNE stage {key!r} is missing for class {class_name!r}")
            obj = stages[key]
            if axis0 is None:
                axis0 = obj.axis
            elif obj.axis.shape != axis0.shape or not np.allclose(obj.axis, axis0, rtol=0, atol=1e-5):
                raise ValueError(f"t-SNE stage {key!r}: input files have different axes")
            blocks.append(obj.spectra)
            labels += [class_name] * obj.n_spectra
        x = np.vstack(blocks)
        if x.shape[0] <= perplexity:
            raise ValueError(f"t-SNE stage {key!r} has {x.shape[0]} spectra, not more than perplexity {perplexity}")
        tsne = TSNE(n_components=2, perplexity=float(perplexity), learning_rate="auto", max_iter=int(max_iter),
                    init="pca", random_state=int(random_state), n_jobs=int(n_jobs))
        with threadpool_limits(limits=int(thread_limit)):
            coords = tsne.fit_transform(x)
        out[key] = (coords, np.asarray(labels), float(tsne.kl_divergence_))
    return out


def plot_tsne_stages(embeddings: Mapping[str, tuple], *, max_cols: int = 5, point_size: float = 7.0,
                     alpha: float = 0.62, title: str = "Class-wise t-SNE across preprocessing stages") -> plt.Figure:
    """One scatter panel per stage, coloured by class."""
    keys = list(embeddings)
    classes = list(dict.fromkeys(np.concatenate([embeddings[k][1] for k in keys]).tolist()))
    cmap = plt.get_cmap("tab10" if len(classes) <= 10 else "tab20")
    ncols = min(max_cols, len(keys))
    nrows = math.ceil(len(keys) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.4 * nrows), squeeze=False)
    for ax, key in zip(axes.flat, keys):
        coords, labels, kl = embeddings[key]
        for c_i, c in enumerate(classes):
            m = labels == c
            ax.scatter(coords[m, 0], coords[m, 1], s=point_size, alpha=alpha, lw=0, color=cmap(c_i % cmap.N),
                       label=c, rasterized=True)
        ax.set_title(f"{stage_display_name(key)}\nKL = {kl:.2f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(6, len(classes)), frameon=False, markerscale=2)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig
