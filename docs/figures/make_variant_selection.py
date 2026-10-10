"""Figure variant_selection.png: per-pixel BIC between peak sets and the composite slot mapping."""
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from _common import BLUE, GREEN, ORANGE, RED, box, gauss, lorentz, save
from hyper_spectral_process.fit import (FitConfig, InitialGuessOptions, MapFitOptions, PeakSpec, StageSpec,
                                        composite_fit, fit_variants, input_peaks_of, select_variants)
from hyper_spectral_process.io import StandardSpectrumMap


@dataclass(frozen=True)
class AppConfig:
    name: str = "variant_selection.png"
    dpi: int = 150
    n: int = 60
    seed: int = 9


def spec(t, pos, w, h, name):
    return PeakSpec(t, pos, w, h, 3.0, 0.0, 3.0, 0.0, 0.5, 0.0, name=name)


def main(cfg: AppConfig) -> None:
    rng = np.random.default_rng(cfg.seed)
    x = np.linspace(500, 560, 241)
    shoulder = np.clip(np.linspace(-0.1, 0.35, cfg.n), 0, None)      # shoulder height grows along a line scan
    y = lorentz(x, 520, 5, 1.0)[None, :] + shoulder[:, None] * gauss(x, 532, 7)[None, :]
    y = y + 0.05 + rng.normal(0, 0.012, (cfg.n, x.size))
    m = StandardSpectrumMap(y, np.column_stack([np.arange(cfg.n), np.zeros(cfg.n)]), x, "cm^-1")
    main_p, sh_p = spec("lorentzian", 520, 5, 1, "main"), spec("gaussian", 532, 7, 0.2, "shoulder")
    cfg_fit = FitConfig((main_p,), (StageSpec("c", 2000, 1e-2), StageSpec("f", 1000, 2e-3)), fit_baseline=True)
    fits = fit_variants(m, [("1 peak", (main_p,)), ("2 peaks", (main_p, sh_p))], cfg_fit,
                        MapFitOptions(skip_low_signal=False), initial_guess=InitialGuessOptions("all"))
    res = {k: f.result for k, f in fits.items()}
    sel = select_variants(res)
    comp = composite_fit(res, sel, input_peaks_of(fits))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    ax = axes[0]
    for i, c in ((5, BLUE), (cfg.n - 5, ORANGE)):
        ax.plot(x, y[i], color=c, lw=0.9, label=f"pixel {i}: shoulder height {shoulder[i]:.2f}")
    ax.set_title("data: a shoulder appears along the scan")
    ax.legend()
    ax.set_xlabel("axis")
    ax = axes[1]
    ax.plot(shoulder, sel.bic[0] - sel.bic[1], "o", ms=3, color=BLUE)
    ax.axhline(0, color=RED, ls=":")
    ax.set_xlabel("true shoulder height")
    ax.set_ylabel("BIC(1 peak) - BIC(2 peaks)")
    ax.set_title("BIC = L ln(SSR/L) + k ln L; k = 4 vs 7\n> 0: the 2-peak variant wins the pixel")
    ax.text(0.02, 0.9, f"2 peaks chosen for {int(np.sum(sel.variant_id == 1))} of {cfg.n} pixels",
            transform=ax.transAxes, fontsize=8)
    ax = axes[2]
    ax.set_axis_off()
    ax.set_title("composite: slots of the largest variant")
    box(ax, 0.02, 0.62, 0.42, 0.22, "variant '2 peaks'\nmain (520) | shoulder (532)", GREEN, 8.5)
    box(ax, 0.02, 0.28, 0.42, 0.22, "variant '1 peak'\nmain (520)", BLUE, 8.5)
    box(ax, 0.56, 0.45, 0.42, 0.22, "slot 0: main\nslot 1: shoulder", ORANGE, 8.5)
    ax.annotate("", xy=(0.56, 0.58), xytext=(0.44, 0.72), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.56, 0.52), xytext=(0.44, 0.39), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->"))
    ax.text(0.02, 0.05, "each peak -> the unused slot with the closest input position (types must match);\n"
            "a slot the winner does not use is NaN and active_mask = False\n"
            f"(here: shoulder slot active in {int(comp.active_mask[:, 1].sum())} pixels)",
            transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    save(fig, cfg.name, cfg.dpi)


if __name__ == "__main__":
    main(AppConfig())
