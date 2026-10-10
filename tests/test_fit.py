import numpy as np
import pytest
import torch

from hyper_spectral_process.fit import (BatchFitter, FitConfig, FitSpectrumMap, InitialGuessOptions, MapFitOptions,
                                        PeakSpec, PeakTable, StageSpec, fit_map, fit_metrics, format_peaks_block,
                                        peak_to_energy, peak_to_wavelength, spring)
from hyper_spectral_process.fit.lineshapes import lorentzian
from hyper_spectral_process.io import HC_EV_NM, StandardSpectrumMap
from hyper_spectral_process.synthetic import SyntheticPeak, synthetic_map

FAST = (StageSpec("coarse", 1500, 1e-2), StageSpec("fine", 1000, 2e-3))


def peak(ptype, pos, width, height, *, reg=0.0, extra=None, name=""):
    return PeakSpec(ptype, pos, width, height, 3.0, reg, 3.0, reg, 0.5, reg, extra=extra, name=name)


def two_peak_map(shape=(5, 5), seed=0):
    ax = np.linspace(340, 440, 300)
    return synthetic_map(ax, [SyntheticPeak("lorentzian", 385, 4.0, 1.0, pos_shift=1.0),
                              SyntheticPeak("gaussian", 405, 6.0, 0.6, pos_shift=0.5)],
                         shape=shape, noise_std=0.005, seed=seed)


# --------------------------------------------------------------- objective
def test_spring_prior_values():
    v = torch.tensor([0.0, 1.0, 2.0, -3.0])
    s = spring(v, torch.zeros(4), torch.ones(4), torch.full((4,), 0.5))
    np.testing.assert_allclose(s.numpy(), 0.5 * np.array([0.0, 1.0, 4.0 + 1.0, 9.0 + 2.0]))   # reg at |d| = 1


def test_spring_prior_pulls_towards_anchor():
    sm = two_peak_map(shape=(1, 1))
    peaks = (peak("lorentzian", 383.0, 4.0, 1.0, reg=10.0), peak("gaussian", 405.0, 6.0, 0.6))
    free = (peak("lorentzian", 383.0, 4.0, 1.0), peak("gaussian", 405.0, 6.0, 0.6))
    opts = MapFitOptions(normalize_01=False, skip_low_signal=False)
    stiff = fit_map(sm.ssm, FitConfig(peaks, FAST), opts).result.pos[0, 0]
    loose = fit_map(sm.ssm, FitConfig(free, FAST), opts).result.pos[0, 0]
    truth = sm.truth["pos"][0, 0]
    assert abs(loose - truth) < 0.05
    assert abs(stiff - 383.0) < abs(loose - 383.0)                 # the spring holds the position back


def test_batch_size_does_not_change_the_fit():
    sm = two_peak_map(shape=(4, 4))
    cfg = FitConfig((peak("lorentzian", 384, 5, 0.9), peak("gaussian", 404, 5, 0.5)), FAST)
    y, x = sm.ssm.spectra, sm.ssm.axis
    table = PeakTable.from_peaks(cfg.fit_peaks())
    batch = BatchFitter(x, y, np.ones(x.size), table, cfg, torch.device("cpu"))
    batch.fit()
    single = BatchFitter(x, y[:1], np.ones(x.size), table, cfg, torch.device("cpu"))
    single.fit()
    np.testing.assert_allclose(batch.results()["pos"][0], single.results()["pos"][0], atol=2e-4)


# ------------------------------------------------------------------ map fit
def test_fit_map_recovers_truth_and_roundtrips(tmp_path):
    sm = two_peak_map()
    cfg = FitConfig((peak("lorentzian", 384, 5, 0.9, name="L"), peak("gaussian", 404, 5, 0.5, name="G")), FAST,
                    fit_baseline=True)
    fit = fit_map(sm.ssm, cfg, MapFitOptions(unnormalize_results=True), initial_guess=InitialGuessOptions("all"))
    r = fit.result
    assert np.max(np.abs(r.pos - sm.truth["pos"])) < 0.05
    assert np.max(np.abs(r.width - sm.truth["width"]) / sm.truth["width"]) < 0.02
    assert np.max(np.abs(r.height - sm.truth["height"]) / sm.truth["height"]) < 0.02
    assert np.nanmin(fit_metrics(r)["wr2"]) > 0.99
    np.testing.assert_allclose(r.reconstruct(), r.recon, atol=1e-5)
    r.save(tmp_path / "f.npz")
    back = FitSpectrumMap.load(tmp_path / "f.npz")
    assert back.peak_names == ("L", "G") and back.area.shape == (25, 2)
    ns = {}
    exec("from hyper_spectral_process.fit import PeakSpec\n" + format_peaks_block(fit.initial_guess.peaks), ns)
    assert ns["PEAKS"][0].pos == pytest.approx(fit.initial_guess.peaks[0].pos, abs=1e-3)


def test_fixed_eta_and_fano():
    x = np.linspace(-30, 30, 400)
    from hyper_spectral_process.fit.lineshapes import fano, pseudo_voigt
    y = np.vstack([pseudo_voigt(x, 0.5, 5.0, 0.3), 0.8 * fano(x, -1.0, 6.0, -4.0)])
    m = StandardSpectrumMap(y, [[0, 0], [1, 0]], x)
    opts = MapFitOptions(normalize_01=False, skip_low_signal=False, select_mode="indices", indices=(0,))
    f1 = fit_map(m, FitConfig((peak("pseudovoigt_fixed", 0.0, 4.0, 0.9, extra={"eta": 0.3}),), FAST), opts).result
    assert f1.pos[0, 0] == pytest.approx(0.5, abs=1e-3) and f1.eta[0, 0] == pytest.approx(0.3)
    assert not f1.valid_mask[1] and np.isnan(f1.pos[1, 0])
    opts = MapFitOptions(normalize_01=False, skip_low_signal=False, select_mode="indices", indices=(1,))
    fano_peak = peak("fano", -0.5, 5.0, 0.7, extra={"q": -3.0, "q_range": (-20.0, -1.5)})
    f2 = fit_map(m, FitConfig((fano_peak,), FAST), opts).result
    assert f2.q[1, 0] == pytest.approx(-4.0, rel=0.02) and np.isnan(f2.area[1, 0])


def test_inactive_peak_is_not_updated_and_baseline_fixed():
    sm = two_peak_map(shape=(1, 2))
    cfg = FitConfig((peak("lorentzian", 384, 5, 0.9), peak("gaussian", 404, 5, 0.5)),
                    (StageSpec("first", 800, 1e-2, active=(0,)),), base_init=0.0, fit_baseline=False)
    r = fit_map(sm.ssm, cfg, MapFitOptions(normalize_01=False, skip_low_signal=False)).result
    np.testing.assert_allclose(r.pos[:, 1], 404.0, atol=1e-5)
    np.testing.assert_allclose(r.base, 0.0)


def test_zero_weight_range_is_ignored():
    sm = two_peak_map(shape=(1, 1))
    y = sm.ssm.spectra.copy()
    ax = sm.ssm.axis
    y[0, (ax > 420) & (ax < 430)] += 5.0                          # an artefact
    m = sm.ssm.replace(spectra=y)
    peaks = (peak("lorentzian", 384, 5, 0.9), peak("gaussian", 404, 5, 0.5))
    opts = MapFitOptions(normalize_01=False, skip_low_signal=False)
    clean = fit_map(m, FitConfig(peaks, FAST, downweight_x_ranges=((419.0, 431.0),), downweight_factor=0.0), opts)
    assert abs(clean.result.pos[0, 1] - sm.truth["pos"][0, 1]) < 0.05


def test_energy_space_fit_applies_jacobian():
    lam = np.linspace(560, 760, 1200)
    e = HC_EV_NM / lam
    band_e = 2.0 * lorentzian(e, 1.85, 0.06)                     # a Lorentzian band per unit energy
    m = StandardSpectrumMap((band_e * HC_EV_NM / lam**2)[None, :], [[0.0, 0.0]], lam, "nm")   # recorded per nm
    p_nm = PeakSpec("lorentzian", HC_EV_NM / 1.84, 20.0, 1.0, 10.0, 0.0, 10.0, 0.0, 0.5, 0.0)
    cfg = FitConfig((p_nm,), (StageSpec("a", 3000, 2e-3), StageSpec("b", 2000, 2e-4)), x_fit_range=(580, 740),
                    fit_baseline=True)          # normalize_01 shifts the minimum of the Lorentzian tails to 0
    r = fit_map(m, cfg, MapFitOptions(fit_axis_space="eV", normalize_01=True, skip_low_signal=False)).result
    assert r.unit == "eV"
    assert r.pos[0, 0] == pytest.approx(1.85, abs=2e-4)
    assert r.width[0, 0] == pytest.approx(0.06, rel=5e-3)
    assert np.nanmax(np.abs(r.spectra_original - r.recon)) < 2e-3
    back = peak_to_wavelength(peak_to_energy(p_nm, "eV", 650.0), "eV", 650.0)
    assert back.pos == pytest.approx(p_nm.pos) and back.width == pytest.approx(p_nm.width)


def test_energy_conversion_mirrors_fano_and_keeps_peak_order():
    from hyper_spectral_process.fit import config_to_energy
    from hyper_spectral_process.fit.lineshapes import fano
    f = PeakSpec("fano", 650.0, 10.0, 1.0, 5.0, 0.0, 5.0, 0.0, 0.5, 0.0, extra={"q": 3.0, "q_range": (0.5, 50.0)})
    e = peak_to_energy(f, "eV", 650.0)
    assert e.get("q") == -3.0 and e.get("q_range") == (-50.0, -0.5)
    g = peak("gaussian", 600.0, 10.0, 1.0)
    cfg = config_to_energy(FitConfig((f, g), FAST), "eV", 650.0)
    assert [p.type for p in cfg.fit_peaks()] == ["gaussian", "fano"]          # nm order kept
    lam = np.linspace(560, 760, 1500)
    band_e = fano(HC_EV_NM / lam, 1.90, 0.08, -3.0)                         # q = -3 per unit energy
    m = StandardSpectrumMap((band_e * HC_EV_NM / lam**2)[None, :], [[0.0, 0.0]], lam, "nm")
    p_nm = PeakSpec("fano", HC_EV_NM / 1.89, 25.0, 1.0, 10.0, 0.0, 10.0, 0.0, 0.5, 0.0,
                    extra={"q": 2.5, "q_range": (0.5, 50.0)})                # the asymmetry as seen in nm
    r = fit_map(m, FitConfig((p_nm,), (StageSpec("a", 3000, 2e-3), StageSpec("b", 2000, 2e-4)), fit_baseline=True,
                             x_fit_range=(580, 740)),
                MapFitOptions(fit_axis_space="eV", skip_low_signal=False)).result
    assert r.q[0, 0] == pytest.approx(-3.0, rel=0.03) and r.pos[0, 0] == pytest.approx(1.90, abs=1e-3)


def test_options_fail_loudly():
    sm = two_peak_map(shape=(2, 2))
    cfg = FitConfig((peak("lorentzian", 384, 5, 0.9),), FAST)
    with pytest.raises(ValueError, match="needs a map in nm"):
        fit_map(sm.ssm, cfg, MapFitOptions(fit_axis_space="eV"))
    with pytest.raises(ValueError, match="indices"):
        fit_map(sm.ssm, cfg, MapFitOptions(select_mode="indices", indices=(99,)))
    with pytest.raises(ValueError, match="fewer than two"):
        fit_map(sm.ssm, FitConfig(cfg.peaks, FAST, x_fit_range=(1000, 2000)))
    with pytest.raises(RuntimeError, match="CUDA"):
        fit_map(sm.ssm, cfg, MapFitOptions(devices=("cuda:999",)))   # missing device: no silent CPU fallback


def test_spec_validation():
    with pytest.raises(ValueError, match="eta"):
        PeakSpec("pv_fixed", 1, 1, 1, 1, 0, 1, 0, 1, 0)
    with pytest.raises(ValueError, match="do not apply"):
        PeakSpec("gaussian", 1, 1, 1, 1, 0, 1, 0, 1, 0, extra={"q": 2})
    with pytest.raises(ValueError, match="q_range"):
        PeakSpec("bwf", 1, 1, 1, 1, 0, 1, 0, 1, 0, extra={"q": 5, "q_range": (-10, -1)})
    assert PeakSpec("PV", 1, 1, 1, 1, 0, 1, 0, 1, 0).type == "pseudovoigt"
    with pytest.raises(ValueError, match="out of range"):
        FitConfig((peak("gaussian", 1, 1, 1),), (StageSpec("s", 10, 0.1, active=(1,)),))
