"""Figure baselines.png: minline, peak stripping (with its stopping rule), protected ranges and anchors."""
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import savgol_filter

from _common import BLUE, GREEN, GREY, ORANGE, PURPLE, RED, gauss, lorentz, save
from hyper_spectral_process.preprocess import (anchor_baseline, minline_baseline, peakstrip_baseline, protect_mask,
                                               strip_windows)


@dataclass(frozen=True)
class AppConfig:
    name: str = "baselines.png"
    dpi: int = 150
    seed: int = 0


def strip_history(y, wmin, wmax):
    """Area removed per stripping iteration (same loop as peakstrip_baseline)."""
    pad = int(np.ceil(0.5 * y.size))
    s = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    areas, ws = [], strip_windows(wmin, wmax)
    for w in ws:
        b = np.minimum(s, savgol_filter(s, w, 0, mode="interp"))
        areas.append(trapezoid(s - b))
        s = b
    return np.array(ws), np.array(areas)


def main(cfg: AppConfig) -> None:
    rng = np.random.default_rng(cfg.seed)
    x = np.linspace(200, 800, 601)
    bg = 40 + 30 * np.exp(-((x - 300) / 250) ** 2) + 0.02 * (x - 200)
    bands = lorentz(x, 420, 10, 120) + lorentz(x, 450, 12, 80) + gauss(x, 640, 40, 25)
    y = bg + bands + rng.normal(0, 1.5, x.size)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    ax = axes[0, 0]
    win, step = 120, 60
    b = minline_baseline(x, y, win, step)
    nodes = np.unique([s + int(np.argmin(y[s:min(s + win, y.size)])) for s in range(0, y.size, step)])
    ax.plot(x, y, color=GREY, lw=0.8, label="data")
    ax.plot(x, bg, color=ORANGE, ls=":", lw=1.2, label="true background")
    ax.plot(x, b, color=BLUE, lw=1.2, label=f"minline (window {win}, step {step} samples)")
    ax.plot(x[nodes], y[nodes], "o", color=RED, ms=4, label="window minima")
    ax.set_title("(a) minline: minima of sliding windows joined by lines")
    ax.legend()

    ax = axes[0, 1]
    b_ps = peakstrip_baseline(y, 21, 301)
    ax.plot(x, y, color=GREY, lw=0.8, label="data")
    ax.plot(x, bg, color=ORANGE, ls=":", lw=1.2, label="true background")
    ax.plot(x, b_ps, color=BLUE, lw=1.2, label="peakstrip (windows 21 ... 301 samples)")
    ws, areas = strip_history(y, 21, 301)
    stop = next(k for k in range(3, len(areas)) if areas[k - 1] < areas[k - 2] and areas[k - 1] < areas[k])
    ins = ax.inset_axes([0.55, 0.5, 0.42, 0.42])
    ins.plot(ws, areas, color=PURPLE, lw=1.0)
    ins.plot(ws[stop - 1], areas[stop - 1], "o", color=RED, ms=5)
    ins.set_xlim(ws[0], ws[min(stop + 8, len(ws) - 1)])
    ins.set_title("area removed per iteration;\nstop at the first local minimum", fontsize=7)
    ins.set_xlabel("window (samples)", fontsize=7)
    ins.tick_params(labelsize=6)
    ax.set_title("(b) peak stripping: y <- min(y, SavGol(y, w)), w growing")
    ax.legend(loc="upper left")

    ax = axes[1, 0]
    mask = protect_mask(x, [420.0, 450.0], 25.0)
    b_free = peakstrip_baseline(y, 5, 41)
    b_prot = peakstrip_baseline(y, 5, 41, protect=mask)
    ax.plot(x, y, color=GREY, lw=0.8, label="data")
    ax.fill_between(x, 0, 220, where=mask, color=GREEN, alpha=0.15, lw=0, label="protected (union of 420+/-25, 450+/-25)")
    ax.plot(x, b_free, color=RED, lw=1.0, label="peakstrip, windows 5 ... 41: follows the band bases")
    ax.plot(x, b_prot, color=BLUE, lw=1.2, label="same, protect_peaks=[420, 450], half width 25")
    ax.set_xlim(300, 560)
    ax.set_ylim(30, 220)
    ax.set_title("(c) protected peaks are bridged from samples outside every range")
    ax.legend(loc="upper right", fontsize=7)

    ax = axes[1, 1]
    anchors = [(220, 330), (520, 590), (720, 790)]
    reference = 40 + 30 * np.exp(-((x - 300) / 250) ** 2)                       # e.g. an off-sample spectrum
    b_ref = anchor_baseline(y[None, :], x, anchors, degree=1, reference=reference)[0]
    b_lin = anchor_baseline(y[None, :], x, anchors[:2], degree=1)[0]
    ax.plot(x, y, color=GREY, lw=0.8, label="data")
    for lo, hi in anchors:
        ax.axvspan(lo, hi, color=GREEN, alpha=0.15, lw=0)
    ax.plot(x, bg, color=ORANGE, ls=":", lw=1.2, label="true background")
    ax.plot(x, b_lin, color=RED, lw=1.0, label="straight line through the first two anchors")
    ax.plot(x, b_ref, color=BLUE, lw=1.2, label="scaled reference + line, all anchors")
    ax.set_title("(d) anchors (option): least squares on band-free ranges (green)")
    ax.legend(loc="upper right", fontsize=7)
    for a in axes.flat:
        a.set_xlabel("axis")
    fig.tight_layout()
    save(fig, cfg.name, cfg.dpi)


if __name__ == "__main__":
    main(AppConfig())
