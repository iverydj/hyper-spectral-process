"""Smoothing stage (``smooth`` and the optional second pass ``smooth2nd``)."""
from __future__ import annotations

from ..io import StandardSpectrumMap
from ._common import savgol_passes


def smooth_savgol(ssm: StandardSpectrumMap, window: int, polyorder: int = 3, *, mode: str = "interp",
                  strength: int = 1) -> StandardSpectrumMap:
    """Savitzky-Golay smoothing applied ``strength`` times.

    Parameters
    ----------
    window : odd window length in samples (the filter works on sample
        indices, so the smoothing width in axis units scales with the step).
    polyorder : polynomial order, smaller than ``window``.
    mode : edge handling of :func:`scipy.signal.savgol_filter`.
    strength : number of passes (>= 1). Repeated passes narrow the pass band
        and broaden sharp bands further.
    """
    if int(strength) != strength or strength < 1:
        raise ValueError(f"strength must be a positive integer; got {strength}")
    return ssm.replace(spectra=savgol_passes(ssm.spectra, window, polyorder, int(strength), mode))
