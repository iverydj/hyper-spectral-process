# Tests and validation

```bash
pip install -e ".[test,qa]"
python -m pytest -q            # CPU; the GPU tests are skipped without CUDA
```

The continuous integration (`.github/workflows/ci.yml`) runs the suite on
CPU with Python 3.10 and 3.12. What the tests check:

| file | checks |
|---|---|
| `test_io.py` | container validation; sorting and merging of duplicate axis values; NPZ round trip (with and without `unit`); raster images with missing nodes; HORIBA write/read round trip and rejection of ragged rows; Nanobase and NT-MDT layouts and orientation options; SIF frame order; the wavelength-to-energy Jacobian reproduces a Lorentzian defined per unit energy and conserves the integral; Raman shift <-> wavelength round trip |
| `test_preprocess.py` | ROI rules; linear interpolation of the simple despike; Whitaker-Hayes flags 1- and 2-sample spikes but not band flanks or the top of a sharp, strong band; protect ranges; MAD = 0 raises; residual and two-pass despike; window validation; the three normalisation modes and the parabolic peak height; `minline` accuracy and its degenerate cases; peak-strip accuracy and window schedule; protected ranges bridged from outside every range; anchor baselines (line, scaled reference); interpolation axis rule; Savitzky-Golay derivatives of a quadratic; split counts and reproducibility; mixup weights and parents; configuration validation; stage snapshots of a pipeline run; splits never mixed by augmentation; a full `run_preprocess` with QA images and its refusal to overwrite |
| `test_lineshapes.py` | unit height and FWHM of the Gaussian, Lorentzian and fixed-eta pseudo-Voigt; TCH limits and agreement with the Voigt profile (`scipy.special.voigt_profile`; FWHM within 1 %, profile within 0.02); Fano maximum, zero, asymptote, FWHM and its Lorentzian limit; analytic areas against numerical integration; torch and NumPy agree |
| `test_fit.py` | spring-penalty values; a stiff spring holds a peak near its anchor; a pixel's result does not depend on the batch it is fitted in; two worker threads give the same result as one; recovery of a synthetic two-peak map with save/load and an executable printed peak block; fixed-eta pseudo-Voigt and Fano fits; inactive peaks and a fixed baseline stay unchanged; zero-weight ranges are ignored; an energy-space fit of a map recorded in nm recovers a Lorentzian defined in energy (Jacobian applied); invalid options and specifications raise |
| `test_variants.py` | per-pixel BIC picks the right model on a map with one- and two-peak regions; composite slots, NaN and `active_mask`; drift penalty; slot type check |
| `test_plot.py` | output files and summary figures; band intensity |
| `test_example_data.py` | the bundled MoS2 maps load with the documented layout; the PL pixels are uniform in wavelength |
| `test_gpu.py` | GPU and CPU agree; a GPU and a CPU worker together give the same result as one GPU (needs CUDA) |

Two numbers from these checks: the TCH pseudo-Voigt stays within 0.02 (of
the peak height) of the Voigt profile with the same Gaussian and Lorentzian
widths, and a pixel fitted alone and inside a batch of 256 gives identical
parameters on the CPU (figure in [peak fitting](peak_fitting.md#batching-and-devices)).
`examples/03_fit_map.py` prints its errors against the known truth of its
synthetic map.
