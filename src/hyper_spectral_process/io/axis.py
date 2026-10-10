"""Conversions between wavelength, photon energy and Raman shift axes.

A spectrum recorded with detector pixels of constant width in wavelength is
a density per unit wavelength, ``I_lambda``. Per unit energy ``E = K / lambda``
it is

    I_E(E) = I_lambda(lambda(E)) * |d lambda / d E| = I_lambda * lambda**2 / K,

which conserves the integrated intensity. Lineshapes defined in energy (for
example a Lorentzian exciton band) describe ``I_E``, so a broad
photoluminescence band fitted on an energy axis should include this
Jacobian. ``K`` is ``h c = 1239.841984 eV nm`` for photon energy in eV and
``1e7`` for wavenumbers in cm^-1.

Every conversion makes the Jacobian an explicit, required choice because the
right answer depends on what the stored values are (counts per detector pixel
of constant width in wavelength are a density per unit wavelength; values
already resampled onto an energy grid are not).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .spectrum_map import StandardSpectrumMap

HC_EV_NM = 1239.841984  # h*c in eV*nm (CODATA 2018)
CM1_NM = 1.0e7          # wavenumber [cm^-1] = 1e7 / wavelength [nm]

_ENERGY_CONSTANTS = {"eV": HC_EV_NM, "cm^-1": CM1_NM}


def energy_constant(unit: str) -> float:
    """``K`` in ``E = K / lambda[nm]`` for ``unit`` ``"eV"`` or ``"cm^-1"``."""
    if unit not in _ENERGY_CONSTANTS:
        raise ValueError(f"energy unit must be 'eV' or 'cm^-1'; got {unit!r}")
    return _ENERGY_CONSTANTS[unit]


def wavelength_jacobian(wavelength_nm: np.ndarray, unit: str, reference_nm: Optional[float] = None) -> np.ndarray:
    """``|d lambda / d E| = lambda**2 / K`` on a wavelength grid.

    With ``reference_nm`` the factor is divided by its value there,
    ``(lambda / reference_nm)**2``: the shape correction is the same but the
    intensities keep their scale near ``reference_nm``.
    """
    lam = np.asarray(wavelength_nm, dtype=np.float64)
    if np.any(lam <= 0):
        raise ValueError("wavelengths must be positive")
    if reference_nm is None:
        return lam**2 / energy_constant(unit)
    if not reference_nm > 0:
        raise ValueError("reference_nm must be positive")
    energy_constant(unit)
    return (lam / float(reference_nm)) ** 2


def wavelength_to_energy(ssm: StandardSpectrumMap, unit: str = "eV", *, jacobian: bool,
                         reference_nm: Optional[float] = None) -> StandardSpectrumMap:
    """Convert a wavelength map (``ssm.unit == "nm"``) to photon energy.

    Parameters
    ----------
    unit : ``"eV"`` or ``"cm^-1"`` (absolute wavenumber).
    jacobian : ``True`` multiplies the intensities by
        :func:`wavelength_jacobian` (density per unit wavelength -> density
        per unit energy); ``False`` only relabels the axis.
    reference_nm : see :func:`wavelength_jacobian`.

    Returns a map on the increasing energy axis ``K / lambda``.
    """
    if ssm.unit != "nm":
        raise ValueError(f"axis unit must be 'nm' for a wavelength conversion; got {ssm.unit!r}")
    k = energy_constant(unit)
    lam = ssm.axis
    spectra = ssm.spectra * wavelength_jacobian(lam, unit, reference_nm) if jacobian else ssm.spectra
    return StandardSpectrumMap.from_unsorted(spectra, ssm.xy, k / lam, unit)


def raman_shift_to_wavelength(ssm: StandardSpectrumMap, laser_nm: float, *, jacobian: bool) -> StandardSpectrumMap:
    """Convert a Raman-shift map (``"cm^-1"``) to absolute wavelength in nm.

    ``lambda = 1e7 / (1e7 / laser_nm - shift)``. ``jacobian=True`` multiplies
    by ``|d shift / d lambda| = 1e7 / lambda**2`` (density per unit shift ->
    density per unit wavelength); ``False`` only relabels, which is right when
    the values are counts per detector pixel of constant width in wavelength.
    """
    if ssm.unit != "cm^-1":
        raise ValueError(f"axis unit must be 'cm^-1' for a Raman-shift conversion; got {ssm.unit!r}")
    if not laser_nm > 0:
        raise ValueError("laser_nm must be positive")
    absolute = CM1_NM / laser_nm - ssm.axis
    if np.any(absolute <= 0):
        raise ValueError("Raman shifts beyond the excitation wavenumber have no wavelength")
    lam = CM1_NM / absolute
    spectra = ssm.spectra * (CM1_NM / lam**2) if jacobian else ssm.spectra
    return StandardSpectrumMap.from_unsorted(spectra, ssm.xy, lam, "nm")


def wavelength_to_raman_shift(ssm: StandardSpectrumMap, laser_nm: float, *, jacobian: bool) -> StandardSpectrumMap:
    """Convert a wavelength map to Raman shift ``1e7 / laser_nm - 1e7 / lambda`` in cm^-1.

    ``jacobian=True`` multiplies by ``|d lambda / d shift| = lambda**2 / 1e7``.
    """
    if ssm.unit != "nm":
        raise ValueError(f"axis unit must be 'nm' for a wavelength conversion; got {ssm.unit!r}")
    if not laser_nm > 0:
        raise ValueError("laser_nm must be positive")
    lam = ssm.axis
    spectra = ssm.spectra * (lam**2 / CM1_NM) if jacobian else ssm.spectra
    return StandardSpectrumMap.from_unsorted(spectra, ssm.xy, CM1_NM / laser_nm - CM1_NM / lam, "cm^-1")
