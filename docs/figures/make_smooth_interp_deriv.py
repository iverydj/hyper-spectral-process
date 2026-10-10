"""Figure smooth_interp_deriv.png: Savitzky-Golay strength, the interpolation axis rule and derivatives."""
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from _common import BLUE, GREEN, GREY, ORANGE, PURPLE, RED, gauss, lorentz, save
from hyper_spectral_process.io import StandardSpectrumMap
from hyper_spectral_process.preprocess import interpolate, interpolation_axis, savgol_derivative, smooth_savgol
from hyper_spectral_process.preprocess.derivative import SmoothBlock


@dataclass(frozen=True)
class AppConfig:
    name: str = "smooth_interp_deriv.png"
    dpi: int = 150
    seed: int = 1


def main(cfg: AppConfig) -> None:
    rng = np.random.default_rng(cfg.seed)
    x = np.linspace(360.0, 440.0, 161)
    clean = lorentz(x, 384, 3.0, 1.0) + lorentz(x, 404, 5.0, 1.6)
    m = StandardSpectrumMap((clean + rng.normal(0, 0.05, x.size))[None, :], [[0.0, 0.0]], x, "cm^-1")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    ax = axes[0]
    ax.plot(x, m.spectra[0], color="0.75", lw=0.8, label="data")
    for s, c in ((1, BLUE), (2, ORANGE), (4, RED)):
        ax.plot(x, smooth_savgol(m, 7, 3, strength=s).spectra[0], color=c, lw=1.0, label=f"window 7, strength {s}")
    ax.plot(x, clean, color=GREEN, ls=":", lw=1.0, label="truth")
    ax.set_xlim(370, 400)
    ax.set_title("(a) smoothing: repeated passes narrow the pass band\nand flatten narrow bands")
    ax.legend()

    ax = axes[1]
    src = np.array([139.94, 140.46, 140.98, 141.50, 142.02, 142.54, 143.06, 143.58, 144.10])
    tgt = interpolation_axis(src, 1.0)
    ax.plot(src, np.zeros_like(src), "o", color=GREY, label="source axis (non-integer start)")
    ax.plot(tgt, np.full(tgt.size, 1.0), "s", color=BLUE, label="target axis, step 1")
    ax.axvline(np.floor(src.min()) + 1, color=RED, ls="--", lw=0.8)
    ax.text(np.floor(src.min()) + 1.05, 0.5, "start = floor(min) + 1\n(smallest integer > min)", color=RED, fontsize=8)
    ax.set_ylim(-0.5, 1.6)
    ax.set_yticks([])
    ax.set_title("(b) interpolation axis: integer start, fixed step,\nlast point <= axis maximum")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_xlabel("axis")

    ax = axes[2]
    u = interpolate(m, 0.5)
    d = (u.axis[1] - u.axis[0])
    d1 = savgol_derivative(u, 1, 11, 3, delta=d, post=SmoothBlock(True, 2, 13, 3))
    d2 = savgol_derivative(u, 2, 11, 3, delta=d, post=SmoothBlock(True, 2, 13, 3))
    ax.plot(u.axis, d1.plain.spectra[0], color="0.7", lw=0.8, label="d1, plain Savitzky-Golay (11, 3)")
    ax.plot(u.axis, d1.final.spectra[0], color=BLUE, lw=1.1, label="d1 after post-smoothing (13, 3) x 2")
    ax.plot(u.axis, 5 * d2.final.spectra[0], color=PURPLE, lw=1.1, label="5 x d2 after post-smoothing")
    ax.set_title("(c) derivatives: pre-smooth -> SavGol derivative -> post-smooth")
    ax.legend(fontsize=7)
    for a in (axes[0], axes[2]):
        a.set_xlabel("Raman shift (cm$^{-1}$)")
    fig.tight_layout()
    save(fig, cfg.name, cfg.dpi)


if __name__ == "__main__":
    main(AppConfig())
