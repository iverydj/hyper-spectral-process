"""Figure plotting.png: band-intensity map, parameter map on the raster, scatter layout and a fit plot."""
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from _common import save
from hyper_spectral_process.fit import FitConfig, MapFitOptions, PeakSpec, StageSpec, fit_map
from hyper_spectral_process.plot import plot_band_intensity, plot_fit, plot_map
from hyper_spectral_process.synthetic import SyntheticPeak, synthetic_map


@dataclass(frozen=True)
class AppConfig:
    name: str = "plotting.png"
    dpi: int = 150
    seed: int = 10


def main(cfg: AppConfig) -> None:
    sm = synthetic_map(np.linspace(360, 420, 241), [SyntheticPeak("lorentzian", 385, 4, 1.0, pos_shift=1.5,
                                                                  height_change=0.5)],
                       shape=(16, 20), noise_std=0.02, seed=cfg.seed)
    ssm = sm.ssm.select(np.arange(sm.ssm.n_spectra) != 37)             # one missing raster node
    fit = fit_map(ssm, FitConfig((PeakSpec("lorentzian", 385, 4, 1, 3, 0, 3, 0, 0.5, 0, name="band"),),
                                 (StageSpec("s", 800, 1e-2),), fit_baseline=True), MapFitOptions())
    r = fit.result
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.0))
    plot_band_intensity(ssm, 380, 390, aggregate="sum", ax=axes[0])
    plot_map(axes[1], r.xy, r.pos[:, 0], cmap="coolwarm", percentiles=(2, 98), label="position")
    axes[1].set_title("plot_map(layout='grid'): fitted position\n(missing node shown grey)")
    plot_map(axes[2], r.xy, r.height[:, 0], cmap="magma", layout="scatter", label="height")
    axes[2].set_title("plot_map(layout='scatter'): fitted height")
    plot_fit(axes[3], r, 0)
    axes[3].set_title("plot_fit: data, fit, peaks, residual", fontsize=9)
    fig.tight_layout()
    save(fig, cfg.name, cfg.dpi)


if __name__ == "__main__":
    main(AppConfig())
