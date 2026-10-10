"""Unit-height peak profiles and their analytic area and FWHM.

Every profile equals 1 at its maximum, so a peak is ``height * profile`` and
``height`` is the peak height in data units. The functions accept NumPy
arrays or torch tensors (``d = x - pos``):

* Gaussian      ``G(x) = exp(-4 ln2 d^2 / w^2)``, FWHM ``w``.
* Lorentzian    ``L(x) = 1 / (1 + 4 d^2 / w^2)``, FWHM ``w``.
* Fixed-eta pseudo-Voigt ``eta L(x; w) + (1 - eta) G(x; w)``: both parts
  share the FWHM ``w`` and equal 0.5 at ``d = +/- w/2``, so the mixture has
  FWHM ``w`` and height 1 for every ``eta``.
* Thompson-Cox-Hastings pseudo-Voigt (P. Thompson, D. E. Cox and J. B.
  Hastings, J. Appl. Cryst. 20, 79 (1987)): independent Lorentzian FWHM
  ``Gamma_L`` and Gaussian FWHM ``Gamma_G`` give the profile FWHM ``Gamma`` and
  mixing ``eta`` through

      Gamma^5 = G^5 + 2.69269 G^4 L + 2.42843 G^3 L^2 + 4.47163 G^2 L^3 + 0.07842 G L^4 + L^5
      eta     = 1.36603 r - 0.47719 r^2 + 0.11116 r^3,   r = L / Gamma

  and the profile is ``eta L(x; Gamma) + (1 - eta) G(x; Gamma)``, an
  approximation of the Voigt profile within about 1 %.
* Breit-Wigner-Fano with resonance FWHM ``w`` (``gamma = w / 2``) and
  asymmetry ``q``:

      F(x) = (q gamma + d)^2 / ((q^2 + 1) (gamma^2 + d^2)),

  equal to 1 at ``d = gamma / q`` (the visible maximum lies there, not at
  ``pos``) and 0 at ``d = -q gamma``. It tends to ``1 / (q^2 + 1)`` far from
  the resonance, so its integral diverges and no area is defined;
  ``|q| -> infinity`` gives the Lorentzian.
"""
from __future__ import annotations

import math

import numpy as np

LN2 = math.log(2.0)
AREA_C_GAUSS = math.sqrt(math.pi / (4.0 * LN2))   # area of a unit-height Gaussian per unit FWHM (1.0645)
AREA_C_LORENTZ = math.pi / 2.0                    # area of a unit-height Lorentzian per unit FWHM (1.5708)
TCH_FWHM_COEFFS = (1.0, 2.69269, 2.42843, 4.47163, 0.07842, 1.0)
TCH_ETA_COEFFS = (1.36603, -0.47719, 0.11116)


def _is_torch(a) -> bool:
    return type(a).__module__.startswith("torch")


def _exp(a):
    return a.exp() if _is_torch(a) else np.exp(a)


def _clip01(a):
    return a.clamp(0.0, 1.0) if _is_torch(a) else np.clip(a, 0.0, 1.0)


def gaussian(x, pos, fwhm):
    """Unit-height Gaussian of FWHM ``fwhm``."""
    z = (x - pos) / fwhm
    return _exp(-4.0 * LN2 * z * z)


def lorentzian(x, pos, fwhm):
    """Unit-height Lorentzian of FWHM ``fwhm``."""
    z = (x - pos) / fwhm
    return 1.0 / (1.0 + 4.0 * z * z)


def pseudo_voigt(x, pos, fwhm, eta):
    """Fixed-eta pseudo-Voigt ``eta L + (1 - eta) G`` with one shared FWHM."""
    return eta * lorentzian(x, pos, fwhm) + (1.0 - eta) * gaussian(x, pos, fwhm)


def tch_fwhm_eta(width_g, width_l):
    """Profile FWHM and mixing ``eta`` (clipped to [0, 1]) of the TCH pseudo-Voigt."""
    g, lw = width_g, width_l
    a = TCH_FWHM_COEFFS
    f5 = a[0] * g**5 + a[1] * g**4 * lw + a[2] * g**3 * lw**2 + a[3] * g**2 * lw**3 + a[4] * g * lw**4 + a[5] * lw**5
    fwhm = f5 ** 0.2
    r = lw / fwhm
    b = TCH_ETA_COEFFS
    return fwhm, _clip01(b[0] * r + b[1] * r**2 + b[2] * r**3)


def pseudo_voigt_tch(x, pos, width_l, width_g):
    """TCH pseudo-Voigt from the Lorentzian FWHM ``width_l`` and the Gaussian FWHM ``width_g``."""
    fwhm, eta = tch_fwhm_eta(width_g, width_l)
    return pseudo_voigt(x, pos, fwhm, eta)


def fano(x, pos, width, q):
    """Unit-height Breit-Wigner-Fano profile with resonance FWHM ``width`` and asymmetry ``q``."""
    gamma = 0.5 * width
    d = x - pos
    return (q * gamma + d) ** 2 / ((q * q + 1.0) * (gamma * gamma + d * d))


# --------------------------------------------------------- derived quantities
def peak_area(peak_type: str, height, width, eta=None, width_g=None):
    """Analytic integral of ``height * profile`` over the whole axis (NumPy).

    Gaussian, Lorentzian and fixed-eta pseudo-Voigt: ``height * width *
    ((1 - eta) C_G + eta C_L)``; TCH pseudo-Voigt: the same with the profile
    FWHM and eta from :func:`tch_fwhm_eta`; Fano: NaN (divergent integral).
    """
    height = np.asarray(height, dtype=np.float64)
    width = np.asarray(width, dtype=np.float64)
    if peak_type == "fano":
        return np.full(np.broadcast(height, width).shape, np.nan)
    if peak_type == "pseudovoigt":
        fwhm, e = tch_fwhm_eta(np.asarray(width_g, dtype=np.float64), width)
    else:
        fwhm = width
        e = {"gaussian": 0.0, "lorentzian": 1.0}.get(peak_type, eta)
        if e is None:
            raise ValueError("peak_area of a fixed-eta pseudo-Voigt needs eta")
        e = np.asarray(e, dtype=np.float64)
    return height * fwhm * ((1.0 - e) * AREA_C_GAUSS + e * AREA_C_LORENTZ)


def peak_fwhm(peak_type: str, width, width_g=None, q=None):
    """FWHM of the profile (NumPy).

    ``width`` for Gaussian, Lorentzian and fixed-eta pseudo-Voigt; the TCH
    profile FWHM for ``pseudovoigt``; for Fano ``width (1 + p^2) / |1 - p^2|``
    with ``p = 1/q`` when ``|q| > 1`` and ``inf`` otherwise (the profile then
    does not fall to half its maximum on one side).
    """
    width = np.asarray(width, dtype=np.float64)
    if peak_type == "pseudovoigt":
        return tch_fwhm_eta(np.asarray(width_g, dtype=np.float64), width)[0]
    if peak_type == "fano":
        q = np.asarray(q, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = 1.0 / q
            return np.where(np.abs(q) > 1.0, width * (1.0 + p * p) / np.abs(1.0 - p * p), np.inf)
    return width.copy()
