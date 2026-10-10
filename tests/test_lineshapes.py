import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.optimize import least_squares
from scipy.special import voigt_profile

from hyper_spectral_process.fit.lineshapes import (fano, gaussian, lorentzian, peak_area, peak_fwhm, pseudo_voigt,
                                                   pseudo_voigt_tch, tch_fwhm_eta)

X = np.linspace(-60, 60, 240001)


def numeric_fwhm(x, y):
    half = np.flatnonzero(y >= 0.5 * y.max())
    return x[half[-1]] - x[half[0]]


@pytest.mark.parametrize("f", [lambda x: gaussian(x, 0.0, 3.0), lambda x: lorentzian(x, 0.0, 3.0),
                               lambda x: pseudo_voigt(x, 0.0, 3.0, 0.3)])
def test_unit_height_and_fwhm(f):
    y = f(X)
    assert y.max() == pytest.approx(1.0)
    assert numeric_fwhm(X, y) == pytest.approx(3.0, abs=1e-3)


def test_tch_limits_and_voigt_accuracy():
    fw, eta = tch_fwhm_eta(np.array(2.0), np.array(1e-9))
    assert fw == pytest.approx(2.0) and eta == pytest.approx(0.0, abs=1e-6)
    fw, eta = tch_fwhm_eta(np.array(1e-9), np.array(2.0))
    assert fw == pytest.approx(2.0) and eta == pytest.approx(1.0, abs=1e-6)
    gg, gl = 3.0, 2.0                                     # Gaussian and Lorentzian FWHM
    sigma, gamma = gg / (2 * np.sqrt(2 * np.log(2))), gl / 2
    voigt = voigt_profile(X, sigma, gamma)
    voigt /= voigt.max()
    tch = pseudo_voigt_tch(X, 0.0, gl, gg)
    assert tch.max() == pytest.approx(1.0)
    assert numeric_fwhm(X, tch) == pytest.approx(numeric_fwhm(X, voigt), rel=0.01)
    assert np.max(np.abs(tch - voigt)) < 0.02             # TCH approximates the Voigt profile within ~1 %


@pytest.mark.parametrize("gl,gg", [(3.0, 3.0), (1.0, 3.0), (3.0, 1.5)])
def test_tch_fit_of_a_voigt_recovers_both_widths(gl, gg):
    x = np.linspace(-40, 40, 8001)
    sigma, gamma = gg / (2 * np.sqrt(2 * np.log(2))), gl / 2
    v = voigt_profile(x, sigma, gamma)
    area = 1.0 / v.max()                          # area of the unit-height Voigt profile
    v = v / v.max()
    fit = least_squares(lambda p: p[2] * pseudo_voigt_tch(x, 0.0, p[0], p[1]) - v, x0=[2.0, 2.0, 1.0],
                        bounds=(0.01, 20))                 # widths and height free, as in a peak fit
    np.testing.assert_allclose(fit.x[:2], [gl, gg], rtol=0.01)
    assert float(peak_area("pseudovoigt", 1.0, gl, width_g=gg)) == pytest.approx(area, rel=0.005)


def test_fano_shape():
    q, w = -3.0, 4.0
    gamma = w / 2
    assert fano(np.array(gamma / q), 0.0, w, q) == pytest.approx(1.0)        # maximum at gamma / q
    assert fano(np.array(-q * gamma), 0.0, w, q) == pytest.approx(0.0, abs=1e-15)
    assert fano(np.array(1e7), 0.0, w, q) == pytest.approx(1.0 / (q * q + 1), rel=1e-5)
    y = fano(X, 0.0, w, q)
    assert numeric_fwhm(X, y) == pytest.approx(float(peak_fwhm("fano", w, q=q)), rel=1e-3)
    assert np.isnan(peak_area("fano", 1.0, w))
    assert np.isinf(peak_fwhm("fano", w, q=0.5))


def test_lorentz_limit_of_fano():
    x = np.linspace(-20, 20, 2001)
    np.testing.assert_allclose(fano(x, 0.0, 3.0, 1e5), lorentzian(x, 0.0, 3.0), atol=1e-4)


@pytest.mark.parametrize("ptype,eta,wg", [("gaussian", None, None), ("lorentzian", None, None),
                                          ("pseudovoigt_fixed", 0.4, None), ("pseudovoigt", None, 3.0)])
def test_analytic_area(ptype, eta, wg):
    h, w = 2.5, 2.0
    prof = {"gaussian": lambda x: gaussian(x, 0, w), "lorentzian": lambda x: lorentzian(x, 0, w),
            "pseudovoigt_fixed": lambda x: pseudo_voigt(x, 0, w, eta),
            "pseudovoigt": lambda x: pseudo_voigt_tch(x, 0, w, wg)}[ptype]
    numeric = h * quad(prof, -np.inf, np.inf, limit=400)[0]
    assert float(peak_area(ptype, h, w, eta, wg)) == pytest.approx(numeric, rel=1e-6)


def test_torch_and_numpy_agree():
    x = np.linspace(-10, 10, 101)
    for fn, args in ((gaussian, (0.5, 2.0)), (pseudo_voigt_tch, (0.5, 2.0, 1.5)), (fano, (0.5, 2.0, -4.0))):
        a = fn(x, *args)
        b = fn(torch.as_tensor(x), *(torch.as_tensor(v, dtype=torch.float64) for v in args)).numpy()
        np.testing.assert_allclose(a, b, rtol=1e-12)
