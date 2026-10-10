import numpy as np
import pytest

from hyper_spectral_process.fit import (FitConfig, InitialGuessOptions, MapFitOptions, PeakSpec, StageSpec,
                                        composite_fit, fit_variants, input_peaks_of, select_variants, slot_map)
from hyper_spectral_process.io import StandardSpectrumMap
from hyper_spectral_process.fit.lineshapes import gaussian, lorentzian

FAST = (StageSpec("coarse", 1500, 1e-2), StageSpec("fine", 800, 2e-3))


def pk(ptype, pos, width, height, name):
    return PeakSpec(ptype, pos, width, height, 3.0, 0.0, 3.0, 0.0, 0.5, 0.0, name=name)


A = pk("lorentzian", 385, 4.0, 1.0, "A")
B = pk("gaussian", 400, 5.0, 0.5, "B")


def mixed_map(n_side=6, seed=0):
    """Left half: peak A only; right half: A and B."""
    rng = np.random.default_rng(seed)
    x = np.linspace(360, 425, 260)
    n = n_side * n_side
    xy = np.column_stack([np.arange(n) % n_side, np.arange(n) // n_side]).astype(float)
    two = xy[:, 0] >= n_side / 2
    y = lorentzian(x, 385.0, 4.0)[None, :] + 0.5 * two[:, None] * gaussian(x, 400.0, 5.0)[None, :]
    y = y + 0.02 + rng.normal(0, 0.01, (n, x.size))
    return StandardSpectrumMap(y, xy, x, "cm^-1"), two


@pytest.fixture(scope="module")
def fitted():
    m, two = mixed_map()
    cfg = FitConfig((A,), FAST, fit_baseline=True)
    fits = fit_variants(m, [("A", (A,)), ("A+B", (A, B))], cfg, MapFitOptions(skip_low_signal=False),
                        initial_guess=InitialGuessOptions("all"))
    return m, two, fits


def test_bic_picks_the_right_variant(fitted):
    m, two, fits = fitted
    sel = select_variants({n: f.result for n, f in fits.items()})
    assert sel.names == ("A", "A+B")
    np.testing.assert_array_equal(sel.variant_id, two.astype(int))
    assert np.all(sel.k_params == [4, 7])
    assert np.all(sel.delta_bic()[two] > 10)


def test_composite_slots(fitted):
    m, two, fits = fitted
    sel = select_variants({n: f.result for n, f in fits.items()})
    comp = composite_fit({n: f.result for n, f in fits.items()}, sel, input_peaks_of(fits))
    assert comp.peak_names == ("A", "B") and comp.metadata["reference_variant"] == "A+B"
    np.testing.assert_array_equal(comp.active_mask[:, 1], two)
    assert np.all(np.isnan(comp.pos[~two, 1])) and np.all(np.isfinite(comp.pos[two, 1]))
    np.testing.assert_allclose(comp.reconstruct(), comp.recon, atol=1e-5)


def test_drift_penalty_and_slot_type_check(fitted):
    m, two, fits = fitted
    res = {n: f.result for n, f in fits.items()}
    sel = select_variants(res, input_peaks=input_peaks_of(fits), drift_penalty_alpha=100.0)
    assert sel.drift_penalty.shape == (2, m.n_spectra) and np.all(sel.drift_max >= 0)
    with pytest.raises(ValueError, match="input_peaks"):
        select_variants(res, drift_penalty_alpha=1.0)
    with pytest.raises(ValueError, match="matching types"):
        slot_map({"one": (pk("gaussian", 385, 4, 1, "x"),), "two": (A, B)})
