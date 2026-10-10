"""Fit every pixel of a map with the staged, spring-regularised peak fit.

The map is synthetic (known truth): a Lorentzian, a Thompson-Cox-Hastings
pseudo-Voigt and a weak broad Gaussian whose positions, widths and heights
vary smoothly across a 30 x 30 raster, on a constant background with noise
(the fit models a constant baseline; remove sloped backgrounds in preprocessing).
The fit uses the original defaults: per-pixel 0-1 normalisation, a fitted
constant baseline, an initial guess on the average of 20 random pixels and a
coarse-to-fine stage plan. Outputs (FitSpectrumMap NPZ, CSV tables, loss
curve, initial-guess figure, summary figures) go to ``AppConfig.out_dir``.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hyper_spectral_process.fit import (FitConfig, InitialGuessOptions, MapFitOptions, PeakSpec, StageSpec, fit_map,
                                        save_fit_outputs)
from hyper_spectral_process.synthetic import SyntheticPeak, synthetic_map


@dataclass(frozen=True)
class AppConfig:
    out_dir: Path = Path("example_output") / "03_fit_map"
    device: str = "cpu"                    # e.g. "cuda:0"
    batch_size: int = 256
    shape: tuple = (30, 30)
    noise_std: float = 0.02
    seed: int = 1


CONFIG = AppConfig()

# The truth of the synthetic map (positions/widths in cm^-1).
TRUTH = (SyntheticPeak("lorentzian", 380.0, 6.0, 1.0, pos_shift=1.5, width_change=0.15, height_change=0.3),
         SyntheticPeak("pseudovoigt", 405.0, 4.0, 0.8, width_g=5.0, pos_shift=1.0, height_change=0.3),
         SyntheticPeak("gaussian", 440.0, 14.0, 0.2, pos_shift=2.0))

# Initial guesses with spring priors: *_limit is the natural scale of a deviation, *_limit_reg the penalty
# there (in units of the normalised mean squared residual, which is ~1e-4 at convergence here).
PEAKS = (
    PeakSpec("lorentzian", pos=381.0, width=5.0, height=0.9, pos_limit=4.0, pos_limit_reg=1e-5,
             width_limit=3.0, width_limit_reg=1e-5, height_limit=0.5, height_limit_reg=0.0, name="L380"),
    PeakSpec("pseudovoigt", pos=404.0, width=3.0, height=0.7, pos_limit=4.0, pos_limit_reg=1e-5,
             width_limit=3.0, width_limit_reg=1e-5, height_limit=0.5, height_limit_reg=0.0,
             extra={"width_g": 4.0}, name="PV405"),
    PeakSpec("gaussian", pos=438.0, width=12.0, height=0.2, pos_limit=6.0, pos_limit_reg=1e-4,
             width_limit=6.0, width_limit_reg=1e-4, height_limit=0.2, height_limit_reg=0.0, name="G440"),
)

FIT = FitConfig(
    peaks=PEAKS,
    stages=(StageSpec("coarse", 4000, 5e-3), StageSpec("fine", 3000, 1e-3)),
    fit_baseline=True,
    x_fit_range=(340.0, 480.0),
    x_label="Raman shift (cm$^{-1}$)",
)


def main(cfg: AppConfig) -> None:
    axis = np.linspace(320.0, 500.0, 451)
    sm = synthetic_map(axis, TRUTH, shape=cfg.shape, noise_std=cfg.noise_std, baseline=(0.15, 0.0, 0.0),
                       seed=cfg.seed)
    opts = MapFitOptions(normalize_01=True, unnormalize_results=True, devices=(cfg.device,),
                         batch_size=cfg.batch_size)
    fit = fit_map(sm.ssm, FIT, opts, initial_guess=InitialGuessOptions(select_mode="random", k_random=20))
    save_fit_outputs(fit, cfg.out_dir, "synthetic")
    r = fit.result
    print(f"fitted {int(r.valid_mask.sum())} spectra in {r.metadata['fit_seconds']:.1f} s")
    for j, name in enumerate(r.peak_names):
        dpos = np.median(np.abs(r.pos[:, j] - sm.truth["pos"][:, j]))
        dh = np.median(np.abs(r.height[:, j] / sm.truth["height"][:, j] - 1))
        print(f"{name}: median |pos error| {dpos:.3f} cm^-1, median |height error| {100 * dh:.2f} %")


if __name__ == "__main__":
    main(CONFIG)
