"""Small helpers shared by the preprocessing stages."""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def check_savgol(window: int, polyorder: int, n_points: int, what: str = "window") -> None:
    """Raise unless ``window`` is an odd integer with ``polyorder < window <= n_points``.

    The original code silently changed invalid windows (even -> +1, shorter
    than 5 -> 5, longer than the spectrum -> shortened); here they are errors.
    """
    if int(window) != window or window < 3 or window % 2 == 0:
        raise ValueError(f"{what} must be an odd integer >= 3; got {window}")
    if int(polyorder) != polyorder or not 0 <= polyorder < window:
        raise ValueError(f"polyorder must be an integer in [0, {what}); got {polyorder}")
    if window > n_points:
        raise ValueError(f"{what}={window} exceeds the spectrum length {n_points}")


def savgol_passes(spectra: np.ndarray, window: int, polyorder: int, passes: int, mode: str = "interp") -> np.ndarray:
    """Apply a Savitzky-Golay filter ``passes`` times along the last axis (0 passes: copy)."""
    if int(passes) != passes or passes < 0:
        raise ValueError(f"the number of passes must be a non-negative integer; got {passes}")
    out = np.array(spectra, dtype=np.float64, copy=True)
    if passes == 0:
        return out
    check_savgol(window, polyorder, out.shape[-1])
    for _ in range(int(passes)):
        out = savgol_filter(out, int(window), int(polyorder), axis=-1, mode=mode)
    return out


def modified_zscore(a: np.ndarray) -> np.ndarray:
    """Robust z-score ``0.6745 (a - median) / MAD`` of every row of ``a``.

    Raises when the median absolute deviation of a row is zero (for example
    low-count integer data in which most differences are 0): no outlier
    threshold can be defined there. The original code returned all-zero
    scores in that case, i.e. silently detected nothing.
    """
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    med = np.median(a, axis=1, keepdims=True)
    mad = np.median(np.abs(a - med), axis=1, keepdims=True)
    zero = np.flatnonzero(mad[:, 0] == 0)
    if zero.size:
        raise ValueError(f"median absolute deviation is zero in {zero.size} spectra (rows {zero[:10].tolist()}); "
                         "the robust z-score is undefined there. Remove these spectra or smooth/bin first")
    return 0.6745 * (a - med) / mad


def range_mask(axis: np.ndarray, ranges) -> np.ndarray:
    """Boolean mask of the axis samples inside any of the ``(lo, hi)`` ranges."""
    mask = np.zeros(axis.size, dtype=bool)
    for lo, hi in ranges:
        if not lo < hi:
            raise ValueError(f"range ({lo}, {hi}) must satisfy lo < hi")
        mask |= (axis >= lo) & (axis <= hi)
    return mask
