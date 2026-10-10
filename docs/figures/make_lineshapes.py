"""Figure lineshapes.png: the five peak types and their parameterisation."""
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from _common import BLUE, GREY, ORANGE, RED, save
from hyper_spectral_process.fit.lineshapes import (fano, gaussian, lorentzian, peak_fwhm, pseudo_voigt,
                                                   pseudo_voigt_tch, tch_fwhm_eta)


@dataclass(frozen=True)
class AppConfig:
    name: str = "lineshapes.png"
    dpi: int = 150


def main(cfg: AppConfig) -> None:
    x = np.linspace(-15, 15, 3001)
    w = 4.0
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.0))

    ax = axes[0]
    ax.plot(x, gaussian(x, 0, w), color=BLUE, label="gaussian: exp(-4 ln2 d$^2$/w$^2$)")
    ax.plot(x, lorentzian(x, 0, w), color=ORANGE, label="lorentzian: 1 / (1 + 4 d$^2$/w$^2$)")
    ax.annotate("", xy=(w / 2, 0.5), xytext=(-w / 2, 0.5), arrowprops=dict(arrowstyle="<->", color=RED))
    ax.text(0, 0.53, "width = FWHM", ha="center", color=RED, fontsize=8)
    ax.set_title("Gaussian and Lorentzian (height 1 at pos)")
    ax.legend(fontsize=7, loc="upper right")

    ax = axes[1]
    for eta, c in zip((0.0, 0.3, 0.7, 1.0), plt.get_cmap("viridis")(np.linspace(0, 0.9, 4))):
        ax.plot(x, pseudo_voigt(x, 0, w, eta), color=c, label=f"eta = {eta}")
    ax.axhline(0.5, color=RED, ls=":", lw=0.7)
    ax.set_title("pseudovoigt_fixed: eta L + (1 - eta) G\none FWHM, eta fixed by the user")
    ax.legend(fontsize=7)

    ax = axes[2]
    for wg, c in zip((0.5, 2.0, 4.0, 8.0), plt.get_cmap("plasma")(np.linspace(0, 0.85, 4))):
        fw, eta = tch_fwhm_eta(np.array(wg), np.array(w))
        ax.plot(x, pseudo_voigt_tch(x, 0, w, wg), color=c,
                label=f"$\\Gamma_G$={wg:g}: FWHM {float(fw):.2f}, area eta {float(eta):.2f}")
    ax.set_title(f"pseudovoigt (TCH): width = $\\Gamma_L$ = {w:g},\nextra width_g = $\\Gamma_G$; FWHM and eta follow")
    ax.legend(fontsize=7)

    ax = axes[3]
    for q, c in zip((-2.0, -5.0, 10.0, 1e4), (RED, BLUE, ORANGE, GREY)):
        lab = "q -> inf (Lorentzian)" if abs(q) > 100 else f"q = {q:g}: max at pos + w/(2q)"
        ax.plot(x, fano(x, 0, w, q), color=c, label=lab)
    ax.axvline(0, color="0.6", ls=":", lw=0.8)
    ax.axhline(1 / (1 + 4), color=RED, ls=":", lw=0.6)
    ax.text(-14.5, 0.23, "asymptote 1/(q$^2$+1)\n(q = -2): area undefined", fontsize=7, color=RED)
    ax.set_title(f"fano (Breit-Wigner-Fano): width = resonance FWHM,\nextra q; FWHM(q=-5) = "
                 f"{float(peak_fwhm('fano', w, q=-5.0)):.2f}")
    ax.legend(fontsize=7, loc="upper right")
    for a in axes:
        a.set_xlabel("x - pos")
    fig.tight_layout()
    save(fig, cfg.name, cfg.dpi)


if __name__ == "__main__":
    main(AppConfig())
