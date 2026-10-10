"""Run the stage-ordered preprocessing pipeline with QA images on two groups of spectra.

The bundled Raman map is split into on-flake and off-flake (substrate)
spectra, saved as two input files with two class names, and preprocessed
together. The run directory holds the processed NPZ families (spectra and
their first and second derivatives), the metadata table, a QA image for every
stage, per-spectrum traces through all stages and a class-wise t-SNE of every
stage (requires scikit-learn).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hyper_spectral_process.io import StandardSpectrumMap
from hyper_spectral_process.preprocess import (BaselineParams, DerivativeParams, DespikeParams, InputFile,
                                               InterpParams, NormalizeParams, PreprocessConfig, QAParams, RoiParams,
                                               SmoothParams, anchor_baseline, run_preprocess)

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AppConfig:
    raman_map: Path = HERE / "data" / "mos2_wrinkled" / "raman_map.npz"
    out_dir: Path = Path("example_output") / "02_preprocess"
    run_id: str = "demo"
    n_per_class: int = 400                 # spectra per class (random subset, to keep the demo quick)
    seed: int = 0


CONFIG = AppConfig()

PIPELINE = PreprocessConfig(
    stage_order=("roi", "despike", "smooth", "baseline", "normalize", "interp", "deriv"),
    roi=RoiParams(xmin=250.0, xmax=600.0),
    despike=DespikeParams(mode="wh", wh_threshold=7.0, wh_m=3),
    smooth=SmoothParams(window=5, polyorder=3, strength=1),
    baseline=BaselineParams(mode="peakstrip", ps_window_min=11, ps_window_max=201),
    normalize=NormalizeParams(mode="instance", round_decimals=None),
    interp=InterpParams(step=0.5),
    derivative=DerivativeParams(orders=(1, 2), window=11, polyorder=3),
    qa=QAParams(trace_n=3, tsne_perplexity=(30,)),
)


def main(cfg: AppConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    ssm = StandardSpectrumMap.load(cfg.raman_map)
    # on-flake: A1g band above a straight background through band-free anchors
    sub = ssm.spectra - anchor_baseline(ssm.spectra, ssm.axis, [(300, 345), (478, 505)], degree=1)
    a1g = sub[:, (ssm.axis >= 398) & (ssm.axis <= 410)].max(axis=1)
    on_flake = a1g > np.median(a1g) + 5 * 1.4826 * np.median(np.abs(a1g - np.median(a1g)))
    rng = np.random.default_rng(cfg.seed)
    inputs = []
    for name, rows in (("flake", np.flatnonzero(on_flake)), ("substrate", np.flatnonzero(~on_flake))):
        pick = np.sort(rng.choice(rows, size=min(cfg.n_per_class, rows.size), replace=False))
        path = cfg.out_dir / f"input_{name}.npz"
        ssm.select(pick).save(path)
        inputs.append(InputFile(path, source_id=name, class_name=name))
    run = run_preprocess(inputs, PIPELINE, cfg.out_dir / "runs", run_id=cfg.run_id)
    print(f"outputs in {run}")


if __name__ == "__main__":
    main(CONFIG)
