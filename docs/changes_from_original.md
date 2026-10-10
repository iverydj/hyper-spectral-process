# Changes from the original code

This package re-implements the original StandardSpectrumMap tools
(`convert/`, the generic NT-MDT converter, `preprocess_v2_4` with the
peak-window normalisation of its later copy, `fit_v4_1` and the heatmap
script). The algorithms are kept; this table lists every behavioural
difference, classified as

- **bug fix**: the original did not do what it was meant to do;
- **design change**: the behaviour differs on purpose (an interface
  simplification, an added option whose default keeps the original
  behaviour, a dropped feature, or an input that used to be adjusted
  silently and now raises an error).

"Raises" means `ValueError` (or `RuntimeError` for devices) with a message
that names the offending value.

## Data container and readers

| # | area | original | now | class | justification |
|---|---|---|---|---|---|
| 1 | container | spectra could contain NaN (ragged rows were NaN-padded) | spectra must be finite | design change | a NaN in the data is an export problem to fix, not something every stage should work around |
| 2 | container | every stage sorted the axis and merged duplicate values | the axis must be strictly increasing; `from_unsorted` sorts and merges duplicates (mean) once, the readers use it | design change | same result, done once at the boundary |
| 3 | HORIBA reader | header accepted with >= 80 % numeric cells; rows with < 3 numbers skipped; ragged rows NaN-padded | header must be numeric; every row must have exactly `M + 2` numbers | design change | silent skipping and padding hid malformed files |
| 4 | HORIBA reader | fallback to a "2-line pair" layout with an index axis 0..M-1 when the header was not numeric; the stand-alone converter also replaced a mismatching axis by an index axis | fallbacks removed (intentionally dropped) | design change | a file in an unexpected layout must fail, not be guessed |
| 5 | Nanobase reader | coordinates swapped and x negated by default (`SWAP_COORD`, `NEGATE_X`), spectra arranged through `XINDEX`/`YINDEX` and flattened in a frame order; missing pixels and empty cells became NaN or were skipped; index axis when no numeric header | coordinates as stored, `swap_xy`/`negate_x`/`negate_y` options; row order irrelevant (coordinates place the rows); empty cells and missing axis raise | design change | instrument-specific orientation as an explicit option; no silent NaN |
| 6 | Andor SIF reader | frames were always reshaped as x-fastest; `FRAME_ORDER="XY"` and `FLIP_X_ORDER` only reordered the output rows, so the frame-to-coordinate assignment never changed | `frame_order="x_fast"`/`"y_fast"` sets the assignment; reversed coordinate arrays replace `FLIP_X_ORDER` | **bug fix** | the options had no effect on the map |
| 7 | Andor SIF reader | `sifreader` backend; 2-D frames flattened into one spectrum | `sif_parser` only; frames must be `(1, M)` | design change | one backend; flattening an image into a spectrum is not meaningful |
| 8 | HORIBA writer | `FLIP_X_ORDER`, `NEGATE_X/Y`, nearest-value grid matching | raster order `x_fast`/`y_fast`; change coordinates with `ssm.replace(xy=...)` | design change | fewer options, exact grid |
| 9 | axis conversions | inside the fit template only | public functions with a required `jacobian` argument | design change | the Jacobian is a decision the user states |
| 10 | FitSpectrumMap NPZ | strings stored as object arrays (pickle needed to load) | unicode arrays, loads with `allow_pickle=False`; `load(..., allow_pickle=True)` reads original files | design change | pickle-free files |
| 11 | FitSpectrumMap NPZ | no area; `peaks_3d` key; `params_width_g` of non-TCH peaks held the 1e-4 clamp value | `params_area` (analytic) and `loss_final` added; `peaks_3d` empty; `width_g`/`q` are 0 for peaks of other types | design change | derived quantities stored once; no clamp artefacts |

## Preprocessing

| # | area | original | now | class | justification |
|---|---|---|---|---|---|
| 12 | configuration | one nested dict (`CONFIG["PIPELINE"]["SG_WINDOW"]`, ...) | typed dataclasses with the same parameters (`PreprocessConfig(smooth=SmoothParams(window=...))`); `stage_order` required | design change | typos become errors; defaults equal the original run configuration |
| 13 | inputs | substrate/analyte/concentration fields; optional parsing of `substrate_analyte_conc.npz` file names | free-form `metadata` per input (columns of the metadata CSV); file-name parsing dropped | design change | project-specific convention |
| 14 | run directory | date-only `RUN_ID` got a time appended | `run_id` used as given (timestamp when None) | design change | no hidden renaming |
| 15 | outputs | optional `npz_basic` copies | dropped | design change | duplicate of `npz_proc_d0` |
| 16 | outputs | the no-augmentation pipeline always ran | runs only when its output is requested | design change | same files, less work |
| 17 | Savitzky-Golay windows | silently changed: even -> +1, < 5 -> 5, >= M -> M - 1, polyorder clipped (smoothing, derivative, residual reference) | odd windows > polyorder and <= M required, otherwise raise | design change | a requested filter is either applied as requested or rejected |
| 18 | smoothing | `strength <= 0` still smoothed once | `strength >= 1` required | design change | as above |
| 19 | ROI | `xmin > xmax` swapped | raises | design change | probable typo |
| 20 | despike `simple` | `linspace(y[s-1], y[e+1], num=n)` wrote the neighbour values into the first and last replaced samples (a step, not a line); exceptions printed and ignored; threshold <= 0 was a silent no-op | interior points of the straight line from `y[s-1]` to `y[e+1]`; errors propagate; threshold must be positive | **bug fix** | the documented linear interpolation |
| 21 | despike `wh` detection | every difference with `|z| > threshold` was a spike: the flanks and the tops of sharp bands were replaced (clipped) | a spike is a jump followed within `max_width` (new, default 2) samples by an opposite jump, with ordinary differences just outside the pair; band flanks and sharp band tops are kept | **bug fix** | the clipping of sharp band tops |
| 22 | despike `wh` replacement | the replaced sample was the difference index `i` (the sample *before* an upward jump, and the last sample of a spike); only `i`, `i+1` were excluded from the neighbour mean, so multi-sample spikes were averaged with spike samples | the flagged spike samples are replaced by the mean of the unflagged samples within `+/- m` | **bug fix** | the spike itself is replaced, from clean neighbours only |
| 23 | despike `wh` | a spike on the first sample was replaced | first- and last-sample spikes are not detected (pair condition) | design change | consequence of #21 |
| 24 | despike | (none) | `protect` ranges that are never modified (option, default empty) | design change | known strong narrow bands (e.g. Si 520 cm^-1) |
| 25 | despike | MAD = 0 gave all-zero z-scores (nothing detected); residual pass silently skipped rows with M < 5 or a failing filter | raises | design change | silent no-ops |
| 26 | despike `wh2` mean replacement | neighbour mean, reference value when no unflagged neighbour | unchanged | - | kept as designed |
| 27 | normalize | rows with a non-positive scale left unscaled | raises | design change | a flat spectrum or a missing reference band is an error |
| 28 | normalize `peak` | only in the later copy | included (same algorithm: parabola through the maximum +/- `peak_refine`, sample maximum when not concave or at the edge) | design change | merged as requested |
| 29 | baseline `minline` | a window longer than the data could leave one distinct node and `interp1d` returned NaN; one window gave an end-point line | window <= M required; fewer than two distinct minima raise | **bug fix** | NaN baseline |
| 30 | baseline protect | every range bridged separately from the samples at its own edges in the unmasked spectrum, so adjacent or overlapping ranges took their end values inside each other's peaks; `linspace(..., n)` endpoint duplication | the union of all ranges is bridged by linear interpolation from samples outside every range | **bug fix** | the protected baseline rose into neighbouring peaks |
| 31 | baseline protect | `protect_peaks` silently ignored with `minline` | raises | design change | ambiguous configuration |
| 32 | baseline `peakstrip` | windows longer than the padded spectrum silently capped | raises | design change | as #17 (the window schedule itself, even -> +1, is unchanged) |
| 33 | baseline | mode alias `"original"` for `"minline"` | `"minline"` only | design change | one name |
| 34 | baseline | (none) | mode `"anchors"`: least-squares polynomial and/or scaled reference through band-free ranges (option) | design change | needed for two-stage background extraction |
| 35 | derivative | `polyorder < order` returned zeros (scipy) | raises; `deriv` listed with empty `orders` raises | design change | silent zeros |
| 36 | split | for n >= 3 every split got at least one spectrum, even one with ratio 0 | only splits with a positive ratio | **bug fix** | a 0 ratio means no such split |
| 37 | augmentation | `alpha_range` low > high swapped | raises | design change | probable typo |
| 38 | QA | figure sizes 24 x 20 in at 240 dpi; stage grids and traces as separate functions with many call forms | one function per figure type, `qa.dpi` setting (150); same content (grids, 3-curve baseline/reference grids, traces, derivative overviews with the plain derivative of the mean source spectrum, SNR overview, mixup examples, cross-file comparison, class-wise t-SNE per perplexity) | design change | readability |

## Fitting

| # | area | original | now | class | justification |
|---|---|---|---|---|---|
| 39 | objective | mean over all pixels of a batch (data and penalties) | sum over pixels of the identical per-pixel objective; the logged loss is still the mean | **bug fix** | gradients were scaled by 1/B, so through Adam's epsilon the effective step depended on the batch size |
| 40 | loss history | `.item()` every iteration (host-device synchronisation) | accumulated on the device, copied once per batch | **bug fix** | the synchronisation serialised the multi-GPU path |
| 41 | result extraction | a full forward pass of the batch for every pixel (`O(B^2)`) | one forward pass per batch | **bug fix** | quadratic cost |
| 42 | devices | all visible GPUs used automatically (streams); CPU used silently when CUDA was unavailable | explicit `devices` list, one worker thread per device; missing devices raise; default `("cpu",)` | design change | shared machines; no silent fallback |
| 43 | nm -> energy fit | the Jacobian `lambda^2 / K` was computed but not applied (spectra only reordered) | applied as `(lambda / lambda_ref)^2` (`lambda_ref` = window centre, so the intensity scale is kept); peak heights and height limits converted with it | **bug fix** | lineshapes per unit energy need intensities per unit energy |
| 44 | nm -> energy fit | `downweight_x_ranges` given in nm were applied to the energy axis (no effect) | converted with the window | **bug fix** | down-weighting was lost |
| 45 | nm -> energy fit | per-pixel normalisation before the conversion; results back-converted to nm (widths and areas by the linear factor at the peak) | conversion first, then normalisation; results stored on the energy axis (`unit` = `"eV"`/`"cm^-1"`, conversion recorded in metadata); refined peaks printed back in nm | design change | with the Jacobian applied, a constant baseline and energy lineshapes have no exact nm representation |
| 46 | axis space | `fit_axis_space` with non-nm data silently fitted natively; unit from `AppConfig.data_axis_unit` | raises; unit taken from the map | design change | silent no-op |
| 47 | peak area | CSV: trapezoid integral over the fit window (window-dependent, divergent for Fano); summary plots: `height * width * factor(eta)` with the Lorentzian width of TCH peaks and the Gaussian factor for Fano | analytic integral over the whole axis for every type (TCH with its profile FWHM and eta); NaN for Fano | **bug fix** (Fano, TCH) / design change (window integral -> analytic) | the Fano integral diverges; TCH used the wrong width |
| 48 | pixel selection | `select_mode="k_random"` (documented in the template) matched no branch and fitted nothing; out-of-range `indices` dropped silently | modes `all`/`indices`/`random`; unknown modes and out-of-range indices raise | **bug fix** | a run that silently fitted nothing |
| 49 | fit window | no overlap with the data: full axis used | raises | design change | silent fallback |
| 50 | normalisation | flat selected spectra got range 1 | raises | design change | as above |
| 51 | specifications | legacy two-stage path (`FitHyperParams.num_iter_stage1/2`, `lr_stage1/2`, `FitConfig.stage1/2`) and unused fields (`basic_three_only`, `pos/width/height_var_reg`); `height_sign_reg` inside `FitHyperParams` | `stages` required; `height_sign_reg` is a `FitConfig` field (same default 5.0) | design change | one path |
| 52 | specifications | unknown `extra` keys ignored; `eta` outside [0, 1] and initial `q` outside `q_range` clamped; limits <= 0 clamped to 1e-6; out-of-range `active` indices dropped; `downweight_factor` clamped to [0, 1] | all raise; types stored in canonical form | design change | silent adjustments |
| 53 | composite | a variant peak could be matched to a slot of another type (the slot kept the reference type) | raises | **bug fix** | parameters would be reconstructed with the wrong lineshape |
| 54 | variants | `variant_select_metric` option (never used) | dropped | design change | dead option |
| 55 | variants | files: per-variant directories, then BIC from the saved files | in memory (`select_variants`, `composite_fit`); `VariantSelection.save` writes the same `variant_id_map.npz` keys | design change | composable functions |
| 56 | initial guess | refined peaks printed with integer positions and widths | 6 significant digits (positions), 4 (widths) | **bug fix** | integer rounding was wrong for eV axes and lost precision |
| 57 | driver | interactive Enter/Esc preview; a diagnostic PNG for every pixel by default; Origin CSV and text exports | dropped (use `plot_fit` for any pixel) | design change | batch use; unused helpers |
| 58 | analysis plots | spatial-coherence map, position-pair scatter, per-peak active maps, overview grid, spectrum-fit examples of the multi-variant analysis | not ported; winner map, BIC margin, BIC/wRMSE histograms, winner mean spectra, drift maps and the summary plots of the composite remain | design change | scope |
| 59 | heatmap tool | irregular coordinates interpolated with `griddata` | raster with NaN for missing nodes, or `layout="scatter"` | design change | no invented values |
| 60 | BIC, spring prior, height-sign penalty, stage plan, recentering, clamps (`width >= 1e-4`, `height >= -1e3`, `q` in `q_range`), Adam, float32, initial guess, low-signal rule, 0-1 normalisation, drift penalty, composite slots | | unchanged | - | the original design |
