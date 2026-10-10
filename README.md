# hyper-spectral-process

`hyper_spectral_process` is a Python toolkit for hyperspectral maps such as
Raman and photoluminescence (PL) maps: it reads instrument map exports into
one simple data format, runs a configurable, stage-ordered preprocessing
pipeline with quality-control images, and fits a multi-peak model to every
pixel of a map at once on CPU or GPU, including a per-pixel choice between
competing peak models. It is meant for researchers who want to apply these
algorithms to their own maps and build their own analyses on top of them;
every stage is a small, documented function that can also be used on its
own.

![pipeline](docs/figures/pipeline.png)

## Features

- **Data**: `StandardSpectrumMap` (spectra, stage coordinates, axis) with an
  NPZ format; readers for HORIBA LabSpec, Nanobase, NT-MDT and Andor SIF
  exports; wavelength / energy / Raman-shift conversions with an explicit
  intensity Jacobian.
- **Preprocessing**: ROI, despiking (Whitaker-Hayes with spike pairing,
  residual z-score pass, simple threshold), Savitzky-Golay smoothing,
  normalisation (per spectrum, global, reference-peak window), baselines
  (sliding-window minima, iterative peak stripping with protected peaks,
  least squares through anchor ranges), interpolation, derivatives,
  train/valid/test split and mixup augmentation; any order, validated before
  it runs; QA images of every stage, per-spectrum traces, class-wise t-SNE.
- **Peak fitting**: Gaussian, Lorentzian, fixed-eta pseudo-Voigt,
  Thompson-Cox-Hastings pseudo-Voigt and Breit-Wigner-Fano peaks; a staged
  Adam optimisation with a soft spring prior on every parameter; all pixels
  of a batch fitted as one tensor problem, batches spread over devices; an
  initial guess on an averaged spectrum; per-pixel model selection by BIC
  with a composite result; parameter maps and summary figures.

## Install

```bash
pip install git+https://github.com/iverydj/hyper-spectral-process.git
# optional: scikit-learn for the t-SNE QA, sif_parser for Andor SIF files
pip install scikit-learn sif_parser
```

Python >= 3.10 with NumPy, SciPy, Matplotlib and PyTorch (install the CUDA
build of PyTorch to fit on a GPU).

## Minimal example

```python
from hyper_spectral_process.io import read_horiba_txt
from hyper_spectral_process.preprocess import (PreprocessConfig, DespikeParams, BaselineParams, RoiParams,
                                               DerivativeParams, run_pipeline)
from hyper_spectral_process.fit import FitConfig, PeakSpec, StageSpec, MapFitOptions, InitialGuessOptions, fit_map

ssm = read_horiba_txt("raman_map.txt", unit="cm^-1")
pre = PreprocessConfig(stage_order=("roi", "despike", "baseline", "normalize"),
                       roi=RoiParams(300, 500), despike=DespikeParams(mode="wh", wh_threshold=7.0, wh_m=3),
                       baseline=BaselineParams(mode="peakstrip", ps_window_min=11, ps_window_max=201),
                       derivative=DerivativeParams(orders=()))
ssm = run_pipeline(ssm, pre).final

peaks = (PeakSpec("lorentzian", pos=384.0, width=4.0, height=0.5, pos_limit=3.0, pos_limit_reg=0.0,
                  width_limit=3.0, width_limit_reg=0.0, height_limit=0.5, height_limit_reg=0.0, name="E2g"),
         PeakSpec("lorentzian", pos=404.0, width=5.0, height=1.0, pos_limit=3.0, pos_limit_reg=0.0,
                  width_limit=3.0, width_limit_reg=0.0, height_limit=0.5, height_limit_reg=0.0, name="A1g"))
cfg = FitConfig(peaks, stages=(StageSpec("coarse", 6000, 5e-3), StageSpec("fine", 3000, 1e-3)),
                fit_baseline=True, x_fit_range=(360.0, 420.0))
fit = fit_map(ssm, cfg, MapFitOptions(devices=("cuda:0",)), initial_guess=InitialGuessOptions("all"))
fit.result.save("fit.npz")                       # FitSpectrumMap: pos, width, height, area, ... per pixel
```

## Documentation

- [Documentation index](docs/index.md)
- [Data format](docs/data_format.md) and [reading instrument files](docs/reading_instrument_files.md)
- [Preprocessing](docs/preprocessing.md): every stage with its algorithm, parameters and pitfalls
- [Peak fitting](docs/peak_fitting.md): lineshapes, peak specification, spring prior, staged optimisation, GPU batching, BIC model selection
- [Plotting](docs/plotting.md)
- [Examples](docs/examples.md): the scripts in `examples/`, including a worked analysis of measured MoS2 Raman and PL maps
- [Changes from the original code](docs/changes_from_original.md), [tests](docs/testing.md), [API reference](docs/api.md)

## Citation

If you use this software, please cite it (see `CITATION.cff`):

> Dong-Joon Yi, *hyper-spectral-process: preprocessing and GPU-batched
> multi-peak fitting of hyperspectral maps*, https://github.com/iverydj/hyper-spectral-process

## License

The code is released under the MIT License (`LICENSE`). The example data in
`examples/data/mos2_wrinkled/` are licensed under CC BY 4.0 (attribution:
Dong-Joon Yi); see the README in that directory.
