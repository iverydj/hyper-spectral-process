# Documentation

| page | what it covers |
|---|---|
| [Data format](data_format.md) | `StandardSpectrumMap` (spectra, coordinates, axis) and `FitSpectrumMap` (fit results): fields, NPZ keys, raster helpers |
| [Reading instrument files](reading_instrument_files.md) | readers for HORIBA LabSpec, Nanobase, NT-MDT and Andor SIF exports; wavelength, energy and Raman-shift axes and the intensity Jacobian |
| [Preprocessing](preprocessing.md) | the stage-ordered pipeline and its validation; one section per stage (ROI, despike, smoothing, normalisation, baselines, interpolation, derivatives, split and augmentation) with the algorithm, parameters and pitfalls; QA images and t-SNE; batch runs |
| [Peak fitting](peak_fitting.md) | lineshapes and their parameters; peak specification; the objective with the soft spring prior; staged optimisation; batching over GPUs; map fitting with initial guess and axis conversion; per-pixel model selection by BIC and the composite result; outputs |
| [Plotting](plotting.md) | parameter and intensity maps, fit plots, summary figures |
| [Examples](examples.md) | walk-throughs of the scripts in `examples/`, including the analysis of measured MoS2 Raman and photoluminescence maps |
| [Changes from the original code](changes_from_original.md) | every behavioural difference from the original implementation, classified as bug fix or design change |
| [Tests and validation](testing.md) | what the test suite checks and how to run it |
| [API reference](api.md) | every public function and class (generated from the docstrings) |

The figures of these pages are made by the scripts `docs/figures/make_*.py`
(run any of them to regenerate its figure); `docs/make_api.py` regenerates
the API reference.

Where to start: read [data format](data_format.md), convert your files
([reading](reading_instrument_files.md)), run `examples/02_preprocess_map.py`
and `examples/03_fit_map.py` with your own data and peaks, and use the
stage sections of [preprocessing](preprocessing.md) and
[peak fitting](peak_fitting.md) to choose the parameters.
