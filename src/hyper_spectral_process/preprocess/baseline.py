"""Baseline stage: estimate a smooth background and subtract it.

Methods (pipeline ``baseline.mode``):

* ``"minline"`` (:func:`minline_baseline`): minima of sliding windows joined
  by straight lines.
* ``"peakstrip"`` (:func:`peakstrip_baseline`): iterative peak stripping with
  Savitzky-Golay windows of increasing length, optionally with protected
  peak ranges.
* ``"anchors"`` (:func:`anchor_baseline`, an option added in this package):
  least-squares background through band-free anchor ranges, as a polynomial
  and/or a scaled reference spectrum.

The two minimum-based methods follow the lower envelope of the noise, so
they sit slightly below the mean background and clip the wings of broad
bands; the anchor method follows the mean of the anchor samples.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from ..io import StandardSpectrumMap
from ._common import range_mask

BASELINE_MODES = ("minline", "peakstrip", "anchors")


# ------------------------------------------------------------------ minline
def minline_baseline(x: np.ndarray, y: np.ndarray, window: int, step: int) -> np.ndarray:
    """Baseline through the minima of the windows ``[s, s + window)``, ``s = 0, step, 2 step, ...``.

    Windows that run past the end of the spectrum are shortened. The minima
    are joined linearly and extrapolated linearly beyond the first and the
    last one. Raises ``ValueError`` when ``window`` exceeds the spectrum or
    fewer than two distinct minima are found (the original then returned NaN).
    """
    n = y.size
    if int(window) != window or not 2 <= window <= n:
        raise ValueError(f"minline window must be an integer in [2, {n}]; got {window}")
    if int(step) != step or step < 1:
        raise ValueError(f"minline step must be a positive integer; got {step}")
    nodes = np.unique([s + int(np.argmin(y[s:min(s + int(window), n)])) for s in range(0, n, int(step))])
    if nodes.size < 2:
        raise ValueError(f"minline found {nodes.size} distinct minimum; decrease window ({window}) or step ({step})")
    return interp1d(x[nodes], y[nodes], kind="linear", fill_value="extrapolate", assume_sorted=True)(x)


# ---------------------------------------------------------------- peakstrip
def protect_mask(axis: np.ndarray, peak_positions: Sequence[float], half_width: float) -> np.ndarray:
    """Samples within ``half_width`` (axis units) of any of ``peak_positions``."""
    if not half_width > 0:
        raise ValueError(f"protect half width must be positive; got {half_width}")
    return range_mask(axis, [(p - half_width, p + half_width) for p in peak_positions])


def fill_masked(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace ``y[mask]`` by linear interpolation (in sample index) from the nearest unmasked samples.

    Every run of masked samples is bridged from the unmasked samples on its
    two sides, so adjacent or overlapping masked ranges never take their end
    values from each other; a run at an edge takes the nearest unmasked value.
    """
    if mask.all():
        raise ValueError("cannot interpolate: every sample is masked")
    idx = np.arange(y.size)
    out = y.copy()
    out[mask] = np.interp(idx[mask], idx[~mask], y[~mask])
    return out


def strip_windows(window_min: int, window_max: int) -> list[int]:
    """Window lengths ``window_min, window_min + 2, ...`` up to ``window_max``, each made odd.

    As in the original, an even ``window_max`` is lowered by one and an even
    window is raised by one.
    """
    if int(window_min) != window_min or int(window_max) != window_max or window_min < 1:
        raise ValueError(f"peakstrip windows must be positive integers; got ({window_min}, {window_max})")
    top = int(window_max) - 1 if int(window_max) % 2 == 0 else int(window_max)
    windows = [w + 1 if w % 2 == 0 else w for w in range(int(window_min), top + 1, 2)]
    if not windows:
        raise ValueError(f"peakstrip needs window_min <= window_max; got ({window_min}, {window_max})")
    return windows


def peakstrip_baseline(y: np.ndarray, window_min: int, window_max: int, polyorder: int = 0,
                       protect: Optional[np.ndarray] = None) -> np.ndarray:
    """Iterative peak stripping.

    The spectrum is padded on both sides with ``ceil(M / 2)`` copies of its end
    values. Then, for the windows of :func:`strip_windows`, it is repeatedly
    replaced by ``min(s, savgol(s, w, polyorder))``; the area removed by each
    iteration is recorded and the loop stops at the first local minimum of
    that area (the result of that iteration is the baseline). If no minimum
    occurs, the last iteration is used. Samples in the boolean ``protect``
    mask are first bridged by :func:`fill_masked`, so the baseline cannot
    follow a protected peak.
    """
    s = fill_masked(y, protect) if protect is not None and protect.any() else np.asarray(y, dtype=np.float64)
    pad = int(np.ceil(0.5 * s.size))
    s = np.concatenate([np.full(pad, s[0]), s, np.full(pad, s[-1])])
    windows = strip_windows(window_min, window_max)
    if windows[-1] > s.size:
        raise ValueError(f"peakstrip window_max={window_max} exceeds the padded length {s.size}")
    if not 0 <= polyorder < windows[0]:
        raise ValueError(f"peakstrip polyorder must be in [0, {windows[0]}); got {polyorder}")
    areas, stripped = [], []
    for w in windows:
        b = np.minimum(s, savgol_filter(s, w, int(polyorder), mode="interp"))
        areas.append(float(trapezoid(s - b)))
        stripped.append(b)
        s = b
        if len(areas) > 3 and areas[-2] < areas[-3] and areas[-2] < areas[-1]:
            return stripped[-2][pad:-pad]
    return stripped[-1][pad:-pad]


# ------------------------------------------------------------------- anchors
def anchor_baseline(spectra: np.ndarray, axis: np.ndarray, anchors: Sequence[tuple[float, float]],
                    degree: Optional[int] = 1, reference: Optional[np.ndarray] = None) -> np.ndarray:
    """Least-squares background ``(N, M)`` through the samples inside the ``anchors`` ranges.

    The background of every spectrum is a polynomial of ``degree`` in the axis
    (``None``: no polynomial) plus, when given, ``reference`` (a background
    spectrum on the same axis, e.g. measured off the sample) times a free
    scale. The coefficients are fitted to the anchor samples only, so the
    form between and beyond the anchors is imposed. Band intensity that
    reaches into an anchor is attributed to the background: place the
    anchors where the bands have decayed.
    """
    if not anchors:
        raise ValueError("anchors needs at least one (lo, hi) range")
    mask = range_mask(axis, anchors)
    for lo, hi in anchors:
        if not np.any((axis >= lo) & (axis <= hi)):
            raise ValueError(f"anchor range ({lo}, {hi}) contains no sample of the axis [{axis[0]}, {axis[-1]}]")
    columns = []
    if reference is not None:
        reference = np.asarray(reference, dtype=np.float64)
        if reference.shape != axis.shape or not np.isfinite(reference).all():
            raise ValueError(f"reference must be a finite array of the axis length {axis.size}")
        columns.append(reference)
    if degree is not None:
        if int(degree) != degree or degree < 0:
            raise ValueError(f"degree must be None or a non-negative integer; got {degree}")
        lo, hi = axis[mask].min(), axis[mask].max()
        t = (axis - 0.5 * (lo + hi)) / (0.5 * (hi - lo)) if hi > lo else axis - lo
        columns += [t**k for k in range(int(degree) + 1)]
    if not columns:
        raise ValueError("the anchor baseline needs a polynomial degree, a reference, or both")
    design = np.stack(columns, axis=1)
    if np.linalg.matrix_rank(design[mask]) < design.shape[1]:
        raise ValueError(f"{int(mask.sum())} anchor samples cannot determine {design.shape[1]} coefficients")
    coef, *_ = np.linalg.lstsq(design[mask], np.asarray(spectra, dtype=np.float64)[:, mask].T, rcond=None)
    return (design @ coef).T


# ------------------------------------------------------------------- driver
def subtract_baseline(ssm: StandardSpectrumMap, mode: str, *, minline_window: int = 500, minline_step: int = 150,
                      ps_window_min: int = 31, ps_window_max: int = 151, ps_polyorder: int = 0,
                      protect_peaks: Optional[Sequence[float]] = None, protect_half_width: float = 15.0,
                      anchors: Sequence[tuple[float, float]] = (), anchor_degree: Optional[int] = 1,
                      anchor_reference: Optional[np.ndarray] = None, clip_negative: bool = False
                      ) -> tuple[StandardSpectrumMap, np.ndarray]:
    """Estimate and subtract a baseline; returns ``(corrected map, baseline (N, M))``.

    Parameters
    ----------
    mode : ``"minline"``, ``"peakstrip"`` or ``"anchors"``.
    minline_window, minline_step : :func:`minline_baseline` (samples).
    ps_window_min, ps_window_max, ps_polyorder : :func:`peakstrip_baseline` (samples).
    protect_peaks, protect_half_width : peak centres and half width (axis units)
        protected from the peak-stripping baseline (``"peakstrip"`` only).
    anchors, anchor_degree, anchor_reference : :func:`anchor_baseline` (``"anchors"`` only).
    clip_negative : set negative values of the corrected spectra to 0.
    """
    if mode not in BASELINE_MODES:
        raise ValueError(f"baseline mode must be one of {BASELINE_MODES}; got {mode!r}")
    if protect_peaks and mode != "peakstrip":
        raise ValueError("protect_peaks applies to mode='peakstrip' only")
    if mode != "anchors" and (anchors or anchor_reference is not None):
        raise ValueError("anchors and anchor_reference apply to mode='anchors' only")
    if mode == "minline":
        base = np.array([minline_baseline(ssm.axis, y, minline_window, minline_step) for y in ssm.spectra])
    elif mode == "peakstrip":
        mask = protect_mask(ssm.axis, protect_peaks, protect_half_width) if protect_peaks else None
        base = np.array([peakstrip_baseline(y, ps_window_min, ps_window_max, ps_polyorder, mask)
                         for y in ssm.spectra])
    else:
        base = anchor_baseline(ssm.spectra, ssm.axis, anchors, anchor_degree, anchor_reference)
    corrected = ssm.spectra - base
    if clip_negative:
        corrected = np.maximum(corrected, 0.0)
    return ssm.replace(spectra=corrected), base
