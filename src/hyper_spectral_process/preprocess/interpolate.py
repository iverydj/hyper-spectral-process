"""Interpolation stage: resample onto a uniform axis.

The target axis is built from the current axis, not from the ROI settings:
it starts at the smallest integer strictly greater than the axis minimum
and advances by ``step`` up to the last point not beyond the axis maximum.
Integer start points make maps measured with slightly different axes
share one grid.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..io import StandardSpectrumMap


def interpolation_axis(source_axis: np.ndarray, step: float, round_decimals: Optional[int] = None) -> np.ndarray:
    """Target axis ``floor(min) + 1 + k * step`` for ``k = 0, 1, ...`` while ``<= max``.

    ``round_decimals`` rounds the target axis values (not the intensities).
    """
    src = np.asarray(source_axis, dtype=np.float64)
    if not (math.isfinite(step) and step > 0):
        raise ValueError(f"interpolation step must be a positive finite number; got {step}")
    start = float(math.floor(src.min()) + 1)
    n = int(np.floor((src.max() - start) / step)) + 1 if start <= src.max() else 0
    if n < 2:
        raise ValueError(f"interpolation axis needs at least 2 points (start {start}, axis max {src.max()}, "
                         f"step {step})")
    axis = start + np.arange(n) * float(step)
    if round_decimals is not None:
        axis = np.round(axis, int(round_decimals))
    if np.any(np.diff(axis) <= 0):
        raise ValueError("rounding made the interpolation axis non-increasing; use more decimals")
    return axis


def interpolate(ssm: StandardSpectrumMap, step: float, round_decimals: Optional[int] = None) -> StandardSpectrumMap:
    """Linear interpolation of every spectrum onto :func:`interpolation_axis`."""
    target = interpolation_axis(ssm.axis, step, round_decimals)
    out = np.array([np.interp(target, ssm.axis, y) for y in ssm.spectra])
    return ssm.replace(spectra=out, axis=target)
