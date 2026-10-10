"""Choose between peak models pixel by pixel with the BIC and assemble a composite result.

The synthetic map has three regions: one band (left), a shoulder band added
(middle), and a third band added (right). Three model variants with one, two
and three peaks are fitted to every pixel; for each pixel the variant with
the lowest BIC wins (optionally penalising variants whose peaks drifted
beyond their ``pos_limit``), and the composite result holds every pixel's
winner in the slots of the largest variant.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hyper_spectral_process.fit import (FitConfig, InitialGuessOptions, MapFitOptions, PeakSpec, StageSpec,
                                        composite_fit, fit_variants, input_peaks_of, select_variants)
from hyper_spectral_process.io import StandardSpectrumMap
from hyper_spectral_process.plot import save_summary_plots, save_variant_plots
from hyper_spectral_process.synthetic import SyntheticPeak, synthetic_map


@dataclass(frozen=True)
class AppConfig:
    out_dir: Path = Path("example_output") / "04_variants"
    device: str = "cpu"
    shape: tuple = (24, 36)
    drift_penalty_alpha: float = 100.0     # 0 = plain BIC
    seed: int = 2


CONFIG = AppConfig()


def spec(ptype, pos, width, height, name):
    return PeakSpec(ptype, pos=pos, width=width, height=height, pos_limit=3.0, pos_limit_reg=1e-5,
                    width_limit=3.0, width_limit_reg=1e-5, height_limit=0.5, height_limit_reg=0.0, name=name)


MAIN = spec("lorentzian", 520.0, 5.0, 1.0, "main")
SHOULDER = spec("gaussian", 532.0, 8.0, 0.3, "shoulder")
SIDE = spec("lorentzian", 555.0, 6.0, 0.4, "side")
VARIANTS = (("1 peak", (MAIN,)), ("2 peaks", (MAIN, SHOULDER)), ("3 peaks", (MAIN, SHOULDER, SIDE)))
FIT = FitConfig(peaks=(MAIN,), stages=(StageSpec("coarse", 3000, 5e-3), StageSpec("fine", 2000, 1e-3)),
                fit_baseline=True, x_fit_range=(490.0, 580.0))


def three_region_map(cfg: AppConfig) -> StandardSpectrumMap:
    axis = np.linspace(480.0, 590.0, 441)
    bands = [SyntheticPeak("lorentzian", 520.0, 5.0, 1.0, pos_shift=0.8),
             SyntheticPeak("gaussian", 533.0, 8.0, 0.25, pos_shift=1.0),
             SyntheticPeak("lorentzian", 555.0, 6.0, 0.35, pos_shift=1.0)]
    parts = [synthetic_map(axis, bands[:k], shape=cfg.shape, noise_std=0.015, baseline=(0.1, 0.0, 0.0),
                           seed=cfg.seed + k) for k in (1, 2, 3)]
    ix = parts[0].ssm.xy[:, 0]
    region = np.digitize(ix, [cfg.shape[1] / 3, 2 * cfg.shape[1] / 3])      # 0, 1, 2 from left to right
    spectra = np.choose(region[:, None], [p.ssm.spectra for p in parts])
    return parts[0].ssm.replace(spectra=spectra)


def main(cfg: AppConfig) -> None:
    ssm = three_region_map(cfg)
    opts = MapFitOptions(devices=(cfg.device,), skip_low_signal=False)
    fits = fit_variants(ssm, VARIANTS, FIT, opts, initial_guess=InitialGuessOptions(select_mode="all"))
    results = {name: f.result for name, f in fits.items()}
    sel = select_variants(results, input_peaks=input_peaks_of(fits), drift_penalty_alpha=cfg.drift_penalty_alpha)
    comp = composite_fit(results, sel, input_peaks_of(fits))
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    sel.save(cfg.out_dir / "variant_id_map.npz", ssm.xy)
    comp.save(cfg.out_dir / "composite__FitSpectrumMap_v4.npz")
    save_variant_plots(sel, results, ssm.xy, cfg.out_dir)
    save_summary_plots(comp, cfg.out_dir / "composite_summary")
    for v, name in enumerate(sel.names):
        print(f"{name}: wins {int(np.sum(sel.variant_id == v))} of {ssm.n_spectra} pixels")


if __name__ == "__main__":
    main(CONFIG)
