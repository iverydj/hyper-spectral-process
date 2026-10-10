"""Normalization stage: shift every spectrum's minimum to 0, then scale.

Modes:

* ``"instance"``: each spectrum by its own maximum (every spectrum peaks at 1).
* ``"global"``: all spectra by the largest value of the map (relative
  intensities between spectra are kept).
* ``"peak"``: each spectrum by the height of the tallest peak inside
  ``peak_window`` (:func:`peak_height`), so a chosen reference band is always
  1 and other bands may exceed 1.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..io import StandardSpectrumMap

NORMALIZE_MODES = ("instance", "global", "peak")


def peak_height(axis: np.ndarray, y: np.ndarray, window: tuple[float, float], refine: int = 2) -> float:
    """Height of the tallest sample of ``y`` inside ``window`` (axis units).

    ``refine = k > 0`` fits a parabola through the maximum and its ``k``
    neighbours on each side and returns the vertex value, which keeps
    single-sample noise out of the scale factor. If the maximum is closer than
    ``k`` samples to the spectrum edge, or the parabola is not concave, the
    sample maximum is returned.
    """
    inside = (axis >= float(window[0])) & (axis <= float(window[1]))
    if not inside.any():
        raise ValueError(f"peak window {window} contains no samples of the axis [{axis[0]}, {axis[-1]}]")
    idx = np.flatnonzero(inside)
    k = int(idx[int(np.argmax(y[inside]))])
    refine = int(refine)
    if refine <= 0 or k - refine < 0 or k + refine >= y.size:
        return float(y[k])
    sl = slice(k - refine, k + refine + 1)
    c = np.polyfit(axis[sl] - axis[k], y[sl], 2)
    if c[0] >= 0:
        return float(y[k])
    vertex = -c[1] / (2.0 * c[0])
    return float(np.polyval(c, vertex))


def normalize(ssm: StandardSpectrumMap, mode: str = "instance", *, round_decimals: Optional[int] = None,
              peak_window: Optional[tuple[float, float]] = None, peak_refine: int = 2) -> StandardSpectrumMap:
    """Shift each spectrum so its minimum is 0, then scale it (see module docstring).

    Parameters
    ----------
    mode : ``"instance"``, ``"global"`` or ``"peak"``.
    round_decimals : optionally round the result (``3`` gives 0.001 precision).
    peak_window : ``(xmin, xmax)`` of the reference band; required for ``"peak"``.
    peak_refine : parabola half-width in samples for :func:`peak_height`.

    Raises ``ValueError`` when a scale factor is not positive (for example a
    flat spectrum); the original left such spectra unscaled.
    """
    if mode not in NORMALIZE_MODES:
        raise ValueError(f"mode must be one of {NORMALIZE_MODES}; got {mode!r}")
    if mode == "peak" and peak_window is None:
        raise ValueError("mode='peak' requires peak_window=(xmin, xmax)")
    y = ssm.spectra - ssm.spectra.min(axis=1, keepdims=True)
    if mode == "instance":
        scale = y.max(axis=1)
    elif mode == "global":
        scale = np.full(y.shape[0], y.max())
    else:
        scale = np.array([peak_height(ssm.axis, row, peak_window, peak_refine) for row in y])
    bad = np.flatnonzero(~(scale > 0))
    if bad.size:
        raise ValueError(f"normalization scale is not positive for {bad.size} spectra (first rows {bad[:5].tolist()})")
    y = y / scale[:, None]
    if round_decimals is not None:
        y = np.round(y, int(round_decimals))
    return ssm.replace(spectra=y)
