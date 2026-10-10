# Changelog

## 0.1.0 (unreleased)

First public version: a modular re-implementation of the original
StandardSpectrumMap converters, `preprocess_v2_4` (with the peak-window
normalisation) and `fit_v4_1`, with the bug fixes and design changes listed
in `docs/changes_from_original.md`.

- `io`: `StandardSpectrumMap` and its NPZ format; HORIBA LabSpec, Nanobase,
  NT-MDT and Andor SIF readers; HORIBA writer; wavelength / energy /
  Raman-shift conversions with an explicit intensity Jacobian.
- `preprocess`: ROI, despike (`simple`, `wh`, `wh2`), smoothing,
  normalisation (`instance`, `global`, `peak`), baselines (`minline`,
  `peakstrip` with protected peaks, `anchors`), interpolation, derivative
  variants, split and mixup augmentation; configuration dataclasses with
  validation; `run_pipeline` and the batch runner `run_preprocess` with QA
  images and class-wise t-SNE.
- `fit`: Gaussian, Lorentzian, fixed-eta and TCH pseudo-Voigt and
  Breit-Wigner-Fano peaks; `PeakSpec`/`StageSpec`/`FitConfig`; staged Adam
  optimisation with the soft spring prior; batched fitting over devices;
  initial guess; nm-to-energy fitting with the Jacobian; per-pixel BIC
  variant selection with drift penalty and composite results;
  `FitSpectrumMap` NPZ, CSV tables.
- `plot`: maps, fit plots, summary and variant figures; `synthetic`: test maps
  with known truth.
- Examples (`examples/01`-`05`), including measured MoS2 Raman and PL maps
  (CC BY 4.0); documentation with algorithm figures; CPU test suite and CI.
