# Examples

The scripts in `examples/` show what the functions do on synthetic and
measured data and how analyses are put together from them. Each script is
configured by the `AppConfig` dataclass at its top (and module-level peak or
pipeline definitions); run it from any directory, outputs go to
`example_output/` there. Set `device="cuda:0"` in the fitting examples to
use a GPU. The numbers below are from CPU runs; GPU runs agree to the
float32 tolerance.

| script | shows | data |
|---|---|---|
| `01_convert_instrument_files.py` | readers and the NPZ format | a LabSpec-format export written from the bundled map |
| `02_preprocess_map.py` | a full preprocessing run with QA images and t-SNE | bundled Raman map, on-flake and off-flake spectra as two classes |
| `03_fit_map.py` | fitting every pixel; outputs and summary figures | synthetic map with known truth |
| `04_variant_selection.py` | per-pixel model choice by BIC and the composite result | synthetic map with three regions |
| `05_mos2_raman_pl_analysis.py` | a worked analysis: measured Raman and PL maps -> position, width and separation maps | bundled MoS2 maps |

## 01: converting instrument files

Each entry of `AppConfig.inputs` is `(path, reader, unit)`; the script reads
the file and writes `<name>.npz`. To be self-contained it first writes a
10 x 8 corner of the bundled Raman map in the LabSpec layout
(`write_horiba_txt`) and converts it back (80 spectra x 1024 points).

## 02: preprocessing with QA

The bundled Raman map is split into on-flake spectra (MoS2 A1g band clearly
above a straight background) and substrate spectra, 400 of each, saved as
two inputs with the class names `flake` and `substrate`, and run through

    roi (250-600 cm^-1) -> despike (wh, threshold 7, m 3) -> smooth (5, 3) -> baseline (peakstrip 11-201)
    -> normalize (instance) -> interp (0.5 cm^-1) -> deriv (orders 1, 2)

The run directory holds `npz_proc_d0/1/2`, the metadata table and the QA
images. Baseline QA of nine random flake spectra (before, baseline, after):

![baseline QA](figures/examples/ex02_baseline_grid.png)

Class-wise t-SNE of every stage (perplexity 30): the two classes stay
separated up to the interpolation; the derivative panels mix them, since the
normalised substrate spectra are mostly noise and noise dominates their
derivatives.

![t-SNE QA](figures/examples/ex02_tsne.png)

## 03: fitting a map

A 30 x 30 synthetic map with a Lorentzian (380 cm^-1), a TCH pseudo-Voigt
(405 cm^-1) and a weak broad Gaussian (440 cm^-1) whose positions, widths
and heights vary smoothly, plus a constant background and noise (standard
deviation 2 % of the main band), is fitted with the defaults of the original
template: per-pixel 0-1 normalisation, a fitted constant baseline, an initial
guess on the average of 20 random pixels, `coarse` (4000 iterations,
lr 5e-3) and `fine` (3000, lr 1e-3) stages, weak springs (`*_limit_reg`
1e-5 on the main peaks, 1e-4 on the weak band). The initial-guess figure
(averaged spectrum, refined peaks, loss):

![initial guess](figures/examples/ex03_initial_guess.png)

Parameter maps written by `save_fit_outputs` (summary/parameter_maps.png):

![parameter maps](figures/examples/ex03_parameter_maps.png)

The script prints the errors against the truth: median absolute position
errors 0.018, 0.022 and 0.118 cm^-1 and median height errors 0.6 %, 0.7 % and
1.7 % for the three bands (900 spectra in 169 s on 4 CPU threads). The
weak, broad band is the least constrained, as expected at its signal level.

## 04: choosing between peak models per pixel

A 24 x 36 synthetic map has one band on the left third, a shoulder band
added in the middle third and a third band added on the right. Three
variants (1, 2 and 3 peaks) are fitted to every pixel and
`select_variants` picks the lowest BIC (with the drift penalty, alpha = 100).
The winner map, the BIC margin between the best two variants, and the BIC and
wRMSE histograms of every variant (`save_variant_plots`):

![variant selection](figures/examples/ex04_variant_selection.png)

Each variant wins exactly its own region (288 of 864 pixels each). The
margin is small (about `3 ln L`, the price of three extra parameters) where
the extra peak is absent and large where it is present. The composite result
(`composite__FitSpectrumMap_v4.npz`) holds three peak slots with NaN where a
pixel's winner has no such peak.

## 05: Raman and PL maps of monolayer MoS2

MOS2_SECTION
