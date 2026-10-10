"""Figure normalization.png: instance, global and peak-window normalization."""
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from _common import BLUE, GREEN, ORANGE, RED, lorentz, save
from hyper_spectral_process.io import StandardSpectrumMap
from hyper_spectral_process.preprocess import normalize


@dataclass(frozen=True)
class AppConfig:
    name: str = "normalization.png"
    dpi: int = 150
    seed: int = 4


def main(cfg: AppConfig) -> None:
    rng = np.random.default_rng(cfg.seed)
    x = np.linspace(370, 420, 201)
    y = np.vstack([10 + lorentz(x, 384, 3, 60) + lorentz(x, 404, 5, 50),      # E2g-like band tallest
                   25 + lorentz(x, 384, 3, 40) + lorentz(x, 404, 5, 90)])     # A1g-like band tallest
    y = y + rng.normal(0, 1.0, y.shape)
    m = StandardSpectrumMap(y, np.zeros((2, 2)), x, "cm^-1")
    win = (378.0, 392.0)
    out = {"instance": normalize(m, "instance", round_decimals=None),
           "global": normalize(m, "global", round_decimals=None),
           "peak": normalize(m, "peak", round_decimals=None, peak_window=win)}
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.9))
    for row, c in zip(m.spectra, (BLUE, ORANGE)):
        axes[0].plot(x, row, color=c, lw=1.0)
    axes[0].set_title("input: two spectra, different offsets\nand a different tallest band")
    titles = {"instance": "instance: (y - min) / max(y - min)\nevery spectrum peaks at 1; the anchor band flips",
              "global": "global: (y - min) / largest value of the map\nrelative intensities kept",
              "peak": f"peak: (y - min) / height in window {win}\nthe reference band is 1 in every spectrum"}
    for ax, (mode, r) in zip(axes[1:], out.items()):
        for row, c in zip(r.spectra, (BLUE, ORANGE)):
            ax.plot(x, row, color=c, lw=1.0)
        ax.axhline(1.0, color=RED, ls=":", lw=0.8)
        ax.set_title(titles[mode], fontsize=9)
    axes[3].axvspan(*win, color=GREEN, alpha=0.15, lw=0)
    axes[3].text(win[0] + 0.3, 1.5, "parabola through the\nmaximum +/- 2 samples\ngives the height", fontsize=7)
    for ax in axes:
        ax.set_xlabel("Raman shift (cm$^{-1}$)")
    fig.tight_layout()
    save(fig, cfg.name, cfg.dpi)


if __name__ == "__main__":
    main(AppConfig())
