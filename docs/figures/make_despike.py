"""Figure despike.png: Whitaker-Hayes detection with spike pairing, its replacement, and the residual pass."""
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from _common import BLUE, GREY, ORANGE, RED, gauss, lorentz, save
from hyper_spectral_process.io import StandardSpectrumMap
from hyper_spectral_process.preprocess import (despike_residual, despike_whitaker_hayes, residual_reference,
                                               whitaker_hayes_mask)
from hyper_spectral_process.preprocess._common import modified_zscore


@dataclass(frozen=True)
class AppConfig:
    name: str = "despike.png"
    dpi: int = 150
    threshold: float = 7.0
    m: int = 3
    published_threshold: float = 3.5      # original pipeline default
    published_m: int = 8
    seed: int = 3


def main(cfg: AppConfig) -> None:
    rng = np.random.default_rng(cfg.seed)
    x = np.arange(300.0)
    clean = 20 + lorentz(x, 150, 6, 300) + gauss(x, 230, 20, 60)
    y = clean + rng.normal(0, 2.0, x.size)
    y[60] += 150                     # single-sample spike
    y[200:202] += 110                # two-sample spike
    y[262:266] += [40, 90, 85, 35]   # wide spike (4 samples): left for the residual pass
    m = StandardSpectrumMap(y[None, :], [[0.0, 0.0]], x)
    z = modified_zscore(np.diff(y)[None, :])[0]
    mask = whitaker_hayes_mask(y[None, :], cfg.threshold, 2)[0]
    wh = despike_whitaker_hayes(m, cfg.threshold, cfg.m).spectra[0]
    # the published rule (max_width=None) with the original pipeline defaults (threshold 3.5, m 8)
    mask_pub = whitaker_hayes_mask(y[None, :], cfg.published_threshold, None)[0]
    wh_pub = despike_whitaker_hayes(m, cfg.published_threshold, cfg.published_m, max_width=None).spectra[0]
    ref = residual_reference(m.replace(spectra=wh[None, :]), 9, 3)[0]
    res = despike_residual(m.replace(spectra=wh[None, :]), 9, 3, 6.0, m=3, replace="mean",
                           protect=[(140.0, 160.0)]).spectra[0]
    zr = modified_zscore((wh - ref)[None, :])[0]

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    ax = axes[0, 0]
    ax.plot(x, y, color=GREY, lw=0.9, label="spectrum with spikes")
    ax.plot(x[mask_pub], y[mask_pub], "o", mfc="none", color=ORANGE, ms=6,
            label=f"published rule (max_width=None), threshold {cfg.published_threshold:g}")
    ax.plot(x[mask], y[mask], "x", color=RED, ms=7, label=f"pair rule (default), threshold {cfg.threshold:g}")
    ax.set_title("(a) data: a sharp band, a broad band and three spikes; flagged samples")
    ax.legend(loc="upper right")
    ax = axes[0, 1]
    ax.plot(x[:-1] + 0.5, z, color=BLUE, lw=0.8)
    for s in (1, -1):
        ax.axhline(s * cfg.threshold, color=RED, ls="--", lw=0.8)
    ax.set_ylim(-60, 60)
    ax.set_title(f"(b) modified z-score of the first difference, threshold +/-{cfg.threshold:g}")
    ax.annotate("jump up then down within\nmax_width samples: a spike", xy=(60, 45), xytext=(75, 48), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("band flanks: up and down jumps\nfar apart, not paired", xy=(147, 15), xytext=(160, -45), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("sample")
    ax = axes[1, 0]
    ax.plot(x, y, color="0.75", lw=0.8, label="input")
    ax.plot(x, wh_pub, color=ORANGE, lw=1.0, ls="--", label="published rule: band flanks and top replaced")
    ax.plot(x, wh, color=BLUE, lw=1.0, label=f"pair rule: flagged -> mean of unflagged within +/-{cfg.m}")
    ax.plot(x, clean, color=GREY, lw=0.8, ls=":", label="truth")
    ax.set_ylim(0, 360)
    ax.set_title("(c) result: the published rule clips the sharp band top,\nthe pair rule leaves it")
    ax.legend(loc="upper left")
    ax = axes[1, 1]
    ax.plot(x, wh, color="0.7", lw=0.8, label="pass 1 (wide spike remains)")
    ax.plot(x, ref, color=ORANGE, lw=0.9, label="reference: Savitzky-Golay (9, 3)")
    ax.plot(x, res, color=BLUE, lw=1.0, label="pass 2: |z(residual)| > 6 replaced (sharp band protected)")
    ax.axvspan(140, 160, color="0.85", lw=0)
    ax.set_xlim(120, 300)
    ax.set_ylim(0, 160)
    ax2 = ax.inset_axes([0.62, 0.55, 0.36, 0.4])
    ax2.plot(x, zr, color=RED, lw=0.7)
    ax2.axhline(6, color=RED, ls="--", lw=0.6)
    ax2.axhline(-6, color=RED, ls="--", lw=0.6)
    ax2.set_xlim(120, 300)
    ax2.set_title("z of residual", fontsize=7)
    ax2.tick_params(labelsize=6)
    ax.set_title("(d) residual pass ('wh2'): reduces the wide spike; it would\nalso flag the sharp band top, so that range is protected (grey)")
    ax.legend(loc="upper left", fontsize=7)
    for a in axes[1]:
        a.set_xlabel("sample")
    fig.tight_layout()
    save(fig, cfg.name, cfg.dpi)


if __name__ == "__main__":
    main(AppConfig())
