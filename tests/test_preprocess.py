import numpy as np
import pytest

from hyper_spectral_process.io import StandardSpectrumMap
from hyper_spectral_process.preprocess import (BaselineParams, DerivativeParams, DespikeParams, InputFile,
                                               NormalizeParams,
                                               PreprocessConfig, QAParams, RoiParams, SplitParams, anchor_baseline,
                                               crop_roi, despike_residual, despike_simple, despike_two_pass,
                                               despike_whitaker_hayes, fill_masked, interpolate, interpolation_axis,
                                               minline_baseline, mixup, normalize, peak_height, peakstrip_baseline,
                                               protect_mask, run_pipeline, run_preprocess, savgol_derivative,
                                               smooth_savgol, split_assignments, split_counts, strip_windows,
                                               validate_config, whitaker_hayes_mask)
from hyper_spectral_process.preprocess.derivative import SmoothBlock


def gauss(x, c, w, h=1.0):
    return h * np.exp(-4 * np.log(2) * ((x - c) / w) ** 2)


def one(y, x=None, unit="cm^-1"):
    y = np.atleast_2d(y)
    x = np.arange(y.shape[1], dtype=float) if x is None else x
    return StandardSpectrumMap(y, np.zeros((y.shape[0], 2)) + np.arange(y.shape[0])[:, None], x, unit)


def noisy_band(m=400, seed=0, width=30.0):
    rng = np.random.default_rng(seed)
    x = np.arange(m, dtype=float)
    return x, gauss(x, m / 2, width, 100.0) + 10 + rng.normal(0, 1.0, m)


# ------------------------------------------------------------------- roi
def test_crop_roi():
    m = one(np.arange(10.0))
    r = crop_roi(m, 2.5, 6.0, round_decimals=0)
    np.testing.assert_array_equal(r.axis, [3, 4, 5, 6])
    assert crop_roi(m, -5, 100).n_points == 10                     # clipped to the data
    with pytest.raises(ValueError, match="xmin <= xmax"):
        crop_roi(m, 6, 2)
    with pytest.raises(ValueError, match="fewer than two"):
        crop_roi(m, 20, 30)


# --------------------------------------------------------------- despike
def test_simple_despike_interpolates_linearly():
    y = np.arange(20, dtype=float)
    y[10] += 100
    out = despike_simple(one(y), 50).spectra[0]
    np.testing.assert_allclose(out, np.arange(20))              # samples 9, 10 bridged by the straight line 8..11


def test_whitaker_hayes_flags_spike_not_band_flank():
    x, y = noisy_band()
    y[100] += 80.0            # one-sample spike
    y[250:252] += 60.0        # two-sample spike on the band flank
    mask = whitaker_hayes_mask(y[None, :], threshold=7.0, max_width=2)[0]
    assert mask[100] and mask[250] and mask[251]
    assert mask.sum() == 3                                       # the steep but real band is untouched
    out = despike_whitaker_hayes(one(y, x), 7.0, 3).spectra[0]
    assert abs(out[100] - np.mean(y[[97, 98, 99, 101, 102, 103]])) < 1e-9
    assert np.all(out[~mask] == y[~mask])


def test_whitaker_hayes_keeps_sharp_strong_band_top():
    x = np.arange(300.0)
    y = 20 + 300 / (1 + 4 * ((x - 150) / 6.0) ** 2) + np.random.default_rng(3).normal(0, 2.0, x.size)
    y[60] += 150
    mask = whitaker_hayes_mask(y[None, :], threshold=7.0, max_width=2)[0]
    assert np.flatnonzero(mask).tolist() == [60]


@pytest.mark.parametrize("bump", [(80.0, 20.0), (20.0, 80.0), (100.0, 30.0)])
def test_whitaker_hayes_two_sample_spike_shapes(bump):
    y = 20 + np.random.default_rng(5).normal(0, 1.0, 300)
    y[100:102] += bump
    mask = whitaker_hayes_mask(y[None, :], threshold=7.0, max_width=2)[0]
    assert np.flatnonzero(mask).tolist() == [100, 101]


def test_simple_despike_keeps_clean_neighbours():
    y = np.full(12, 20.0)
    y[1] = 21.0
    y[2] = 90.0
    out = despike_simple(one(y), 30.0).spectra[0]
    assert out[1] == 21.0 and out[2] == pytest.approx(20.5)
    z = np.full(12, 20.0)
    z[1] = 90.0                                   # a spike on the second sample is removed too
    assert despike_simple(one(z), 30.0).spectra[0, 1] == pytest.approx(20.0)


def test_whitaker_hayes_protect_and_zero_mad():
    x, y = noisy_band()
    y[100] += 80.0
    out = despike_whitaker_hayes(one(y, x), protect=[(95.0, 105.0)]).spectra[0]
    assert out[100] == y[100]
    with pytest.raises(ValueError, match="median absolute deviation"):
        whitaker_hayes_mask(np.zeros((1, 50)))


def test_residual_pass_and_two_pass():
    x, y = noisy_band(seed=1)
    y[300] += 40.0
    out = despike_residual(one(y, x), 9, 3, 6.0, replace="mean").spectra[0]
    assert out[300] < y[300] - 20
    p1, p2, ref = despike_two_pass(one(y, x), wh_threshold=7.0)
    assert ref.shape == (1, 400) and p2.spectra[0, 300] < y[300] - 20


# ------------------------------------------------------- smooth/normalize
def test_smooth_validates_window():
    m = one(np.random.default_rng(0).normal(size=50))
    assert smooth_savgol(m, 7, 3, strength=2).n_points == 50
    with pytest.raises(ValueError, match="odd"):
        smooth_savgol(m, 6, 3)
    with pytest.raises(ValueError, match="exceeds"):
        smooth_savgol(m, 51, 3)


def test_normalize_modes():
    x = np.linspace(0, 100, 201)
    y = np.vstack([gauss(x, 30, 8, 2.0) + gauss(x, 70, 8, 5.0) + 1.0, gauss(x, 30, 8, 4.0) + gauss(x, 70, 8, 1.0)])
    m = StandardSpectrumMap(y, np.zeros((2, 2)), x)
    inst = normalize(m, "instance", round_decimals=None).spectra
    np.testing.assert_allclose(inst.max(axis=1), 1.0)
    assert inst.min() == 0.0
    glob = normalize(m, "global", round_decimals=None).spectra
    assert glob.max() == pytest.approx(1.0) and glob[1].max() < 1.0
    peak = normalize(m, "peak", round_decimals=None, peak_window=(20, 40)).spectra
    np.testing.assert_allclose([peak_height(x, r, (20, 40)) for r in peak], 1.0, rtol=1e-6)
    assert peak[0].max() > 2.0                                   # the other band may exceed 1
    with pytest.raises(ValueError, match="peak_window"):
        normalize(m, "peak")
    with pytest.raises(ValueError, match="not positive"):
        normalize(StandardSpectrumMap(np.ones((1, 5)), [[0, 0]], np.arange(5.0)), "instance")


def test_peak_height_refine():
    x = np.linspace(0, 10, 101)
    y = gauss(x, 5.03, 2.0)
    assert peak_height(x, y, (4, 6), refine=2) == pytest.approx(1.0, abs=2e-4)
    assert peak_height(x, y, (4, 6), refine=0) < 1.0
    xs = np.arange(101.0)
    slope = 1 - (xs - 200) ** 2 / 1e5                           # no peak top inside the window
    assert peak_height(xs, slope, (40, 60), refine=2) == pytest.approx(slope[60])


# --------------------------------------------------------------- baseline
def test_minline_baseline_and_degenerate_case():
    x = np.arange(100.0)
    y = 0.05 * x + gauss(x, 50, 6, 10)
    b = minline_baseline(x, y, 20, 10)
    assert np.max(np.abs(b - 0.05 * x)) < 0.06
    with pytest.raises(ValueError, match="window"):
        minline_baseline(x, y, 500, 150)
    rising = np.arange(100.0)
    with pytest.raises(ValueError, match="distinct"):
        minline_baseline(x, -rising, 100, 30)                    # every window has its minimum at the last sample


def test_peakstrip_baseline_and_windows():
    assert strip_windows(10, 100)[:3] == [11, 13, 15] and strip_windows(10, 100)[-1] == 99
    x = np.arange(400.0)
    bg = 5 + 0.01 * x
    y = bg + gauss(x, 200, 10, 50)
    b = peakstrip_baseline(y, 11, 151)
    assert np.max(np.abs(b - bg)[50:350]) < 2.0


def test_protect_union_reads_anchors_outside_all_ranges():
    x = np.arange(100.0)
    y = np.zeros(100)
    y[40:50] = 10.0
    y[50:60] = 20.0                                               # two adjacent protected peaks
    mask = protect_mask(x, [45.0, 55.0], 5.0)
    filled = fill_masked(y, mask)
    assert np.all(filled[mask] == 0.0)                           # bridged from 39 and 61, not from the other peak


def test_anchor_baseline_recovers_line():
    x = np.linspace(300, 500, 401)
    line = 100 + 0.2 * (x - 300)
    y = (line + gauss(x, 400, 10, 500))[None, :]
    b = anchor_baseline(y, x, [(300, 340), (460, 500)], degree=1)
    np.testing.assert_allclose(b[0], line, atol=1e-6)
    ref = np.exp(-(x - 300) / 300)
    b2 = anchor_baseline((3 * ref)[None, :], x, [(300, 330), (480, 500)], degree=None, reference=ref)
    np.testing.assert_allclose(b2[0], 3 * ref, rtol=1e-10)


# ---------------------------------------------------------- interp/deriv
def test_interpolation_axis_rule():
    ax = interpolation_axis(np.array([139.9, 200.3]), 2.0)
    assert ax[0] == 140.0 and ax[-1] == 200.0
    assert interpolation_axis(np.array([140.0, 150.0]), 1.0)[0] == 141.0   # strictly greater than the minimum
    assert interpolation_axis(np.array([0.5, 1000.3]), 0.1)[-1] == pytest.approx(1000.3)   # no rounding loss
    m = one(np.linspace(0, 1, 50)[None, :] * 2, np.linspace(10.5, 59.5, 50))
    out = interpolate(m, 0.5)
    np.testing.assert_allclose(out.spectra[0], (out.axis - 10.5) / 49.0 * 2, atol=1e-12)


def test_derivative_of_quadratic():
    x = np.linspace(0, 10, 201)
    m = one((x**2)[None, :], x)
    d1 = savgol_derivative(m, 1, 11, 3, delta=x[1] - x[0])
    np.testing.assert_allclose(d1.final.spectra[0, 10:-10], 2 * x[10:-10], atol=1e-8)
    d2 = savgol_derivative(m, 2, 11, 3, delta=x[1] - x[0], post=SmoothBlock(True, 1, 7, 2))
    assert d2.post is not None and np.allclose(d2.plain.spectra[0, 10:-10], 2.0)


# ----------------------------------------------------- split/augmentation
def test_split_counts_and_assignments():
    assert split_counts(10, 0.6, 0.2, 0.2) == (6, 2, 2)
    assert split_counts(3, 0.98, 0.01, 0.01) == (1, 1, 1)
    labels = split_assignments(10, 0.6, 0.2, 0.2, seed=3)
    assert sorted(labels.tolist()).count("train") == 6
    np.testing.assert_array_equal(labels, split_assignments(10, 0.6, 0.2, 0.2, seed=3))


def test_mixup_is_convex_combination():
    pool = np.random.default_rng(0).uniform(size=(5, 30))
    out, first, ex = mixup(pool, 20, np.random.default_rng(1), noise_std=0.0)
    for y, e in zip(out, ex):
        assert e["weights"].sum() == pytest.approx(1.0)
        np.testing.assert_allclose(y, e["weights"] @ pool[e["sources"]])
        assert 2 <= len(e["sources"]) <= 4
    assert np.all(first == [e["sources"][0] for e in ex])


# ------------------------------------------------------- config/pipeline
def test_config_validation():
    with pytest.raises(ValueError, match="unknown"):
        validate_config(PreprocessConfig(stage_order=("roi", "magic")))
    with pytest.raises(ValueError, match="despike.mode"):
        validate_config(PreprocessConfig(stage_order=("despike",)))
    with pytest.raises(ValueError, match="'baseline' is not in stage_order"):
        validate_config(PreprocessConfig(stage_order=("roi",), baseline=BaselineParams(mode="peakstrip"),
                                         derivative=DerivativeParams(orders=())))
    with pytest.raises(ValueError, match="split"):
        validate_config(PreprocessConfig(stage_order=("roi", "split"), split=SplitParams(enable=True)))
    with pytest.raises(ValueError, match="derivative.orders"):
        validate_config(PreprocessConfig(stage_order=("roi",)))


def _demo_map(n=30, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(200, 600, 401)
    y = gauss(x, 400 + rng.normal(0, 2, (n, 1)), 15, 50) + 5 + 0.01 * x + rng.normal(0, 0.5, (n, x.size))
    y[3, 100] += 200.0
    xy = np.column_stack([np.arange(n) % 6, np.arange(n) // 6]).astype(float)
    return StandardSpectrumMap(y, xy, x, "cm^-1")


def _cfg(**kw):
    base = dict(stage_order=("roi", "despike", "smooth", "baseline", "normalize", "interp", "deriv"),
                roi=RoiParams(250, 550), despike=DespikeParams(mode="wh", wh_threshold=7.0, wh_m=3),
                baseline=BaselineParams(mode="peakstrip", ps_window_min=11, ps_window_max=121),
                normalize=NormalizeParams(mode="instance", round_decimals=None),
                derivative=DerivativeParams(orders=(2,)), qa=QAParams(tsne_enable=False, trace_n=2))
    base.update(kw)
    return PreprocessConfig(**base)


def test_pipeline_stage_snapshots():
    res = run_pipeline(_demo_map(), _cfg())
    assert res.trace_order == ["source", "roi", "spike", "smooth", "baseline", "norm", "interp",
                               "deriv1_plain", "deriv1", "deriv2_plain", "deriv2"]
    assert set(res.derivatives) == {"procd2"}                       # deriv1 is computed for QA only
    assert res.baseline_curve.shape == res.stages["smooth"].spectra.shape
    np.testing.assert_allclose(res.stages["norm"].spectra.max(axis=1), 1.0)
    assert res.stages["spike"].spectra[3].max() < 100               # spike removed


def test_pipeline_split_and_augmentation():
    from hyper_spectral_process.preprocess import AugmentationParams
    cfg = PreprocessConfig(stage_order=("split", "augmentation", "normalize"), split=SplitParams(enable=True),
                           augmentation=AugmentationParams(max_new_per_split={"train": 7, "valid": 3, "test": 2}),
                           normalize=NormalizeParams(round_decimals=None), derivative=DerivativeParams(orders=()))
    m = _demo_map()
    res = run_pipeline(m, cfg, splits=split_assignments(m.n_spectra, 0.6, 0.2, 0.2, 0))
    assert res.final.n_spectra == m.n_spectra + 12 and res.is_augmented.sum() == 12
    for label in ("train", "valid", "test"):                        # mixup never crosses splits
        sel = res.is_augmented & (res.splits == label)
        assert np.all(res.splits[res.parent[sel]] == label)


def test_run_preprocess_outputs(tmp_path):
    m = _demo_map()
    m.save(tmp_path / "a.npz")
    _demo_map(seed=1).save(tmp_path / "b.npz")
    inputs = [InputFile(tmp_path / "a.npz", "a", "class_a"), InputFile(tmp_path / "b.npz", "b", "class_b")]
    run = run_preprocess(inputs, _cfg(), tmp_path / "out", run_id="t1")
    assert (run / "npz_proc_d0" / "a__proc.npz").exists() and (run / "npz_proc_d2" / "b__procd2.npz").exists()
    assert (run / "metadata" / "samples_metadata.csv").exists() and (run / "qa_images" / "trace").exists()
    with pytest.raises(FileExistsError):
        run_preprocess(inputs, _cfg(), tmp_path / "out", run_id="t1")
