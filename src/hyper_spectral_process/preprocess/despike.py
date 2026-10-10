"""Despike stage: remove cosmic-ray spikes.

Three methods (pipeline ``despike.mode``):

* ``"simple"`` (:func:`despike_simple`): absolute jumps ``|y[k+1] - y[k]|``
  above a threshold in data units, filled by linear interpolation.
* ``"wh"`` (:func:`despike_whitaker_hayes`): Whitaker-Hayes detection on the
  modified z-score of the first difference (Whitaker & Hayes, Chemometr.
  Intell. Lab. Syst. 179, 82 (2018)); flagged samples are replaced by the
  mean of their unflagged neighbours. By default a spike must be an isolated
  pair of opposite jumps (``max_width``); ``max_width=None`` uses the
  published rule (every large difference).
* ``"wh2"`` (:func:`despike_two_pass`): ``"wh"`` followed by a residual
  z-score pass against a Savitzky-Golay reference (:func:`despike_residual`),
  which catches remnants of wide spikes.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.signal import savgol_filter

from ..io import StandardSpectrumMap
from ._common import check_savgol, modified_zscore, range_mask


# ------------------------------------------------------------------- simple
def _runs(idx: np.ndarray) -> list[tuple[int, int]]:
    """Consecutive runs ``(start, end)`` of a sorted integer array."""
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]])
    return list(zip(starts.tolist(), ends.tolist()))


def despike_simple(ssm: StandardSpectrumMap, threshold: float) -> StandardSpectrumMap:
    """Replace jumps larger than ``threshold`` (data units) by linear interpolation.

    For every run of consecutive difference indices ``start .. end`` with
    ``|y[k+1] - y[k]| > threshold``, the samples between the first and the last
    jump, ``start + 1 .. end``, are replaced by the straight line from
    ``y[start]`` to ``y[end + 1]``. A single jump (a step) changes nothing.
    """
    if not threshold > 0:
        raise ValueError(f"threshold must be positive; got {threshold}")
    out = ssm.spectra.copy()
    for row in out:
        jumps = np.flatnonzero(np.abs(np.diff(row)) > threshold)
        for start, end in _runs(jumps):
            n = end - start
            if n > 0:
                row[start + 1:end + 1] = np.linspace(row[start], row[end + 1], n + 2)[1:-1]
    return ssm.replace(spectra=out)


# ---------------------------------------------------------- Whitaker-Hayes
def whitaker_hayes_mask(spectra: np.ndarray, threshold: float = 7.0, max_width: Optional[int] = 2) -> np.ndarray:
    """Spike mask ``(N, M)`` from the Whitaker-Hayes test.

    The first differences ``d_k = y[k+1] - y[k]`` of every spectrum are scored
    with the modified z-score.

    ``max_width=None`` is the published rule (Whitaker & Hayes 2018, and the
    original code): sample ``k`` is flagged for every difference with
    ``|z_k| > threshold``. Every large difference counts, so the flanks and
    the top of a band sampled by few points are flagged too (more so at low
    thresholds such as the original default 3.5).

    With an integer ``max_width`` (the default) a spike is

    * a jump ``i`` with ``|z_i| > threshold``,
    * followed within ``max_width`` samples by a jump ``j`` of the opposite
      sign with ``|z_j| > threshold``,
    * with ordinary differences just outside the pair (``|z_{i-1}|`` and
      ``|z_{j+1}| <= threshold``): the spike stands on a background that is
      flat at the noise level. The farthest ``j`` that satisfies this is taken.

    The samples ``i + 1 .. j`` between the two jumps are flagged. The pair
    condition keeps the steep flanks of real bands from being flagged and the
    isolation condition keeps the top of a sharp, strong band (whose flanks
    are themselves large jumps) from being flagged. A spike on the steep
    flank of a sharp band, a spike wider than ``max_width`` and a spike on the
    first or last sample are not detected (the residual pass of ``"wh2"``
    catches wide spikes).
    """
    if max_width is not None and (int(max_width) != max_width or max_width < 1):
        raise ValueError(f"max_width must be None or a positive integer; got {max_width}")
    if not threshold > 0:
        raise ValueError(f"threshold must be positive; got {threshold}")
    spectra = np.atleast_2d(spectra)
    z = modified_zscore(np.diff(spectra, axis=1))
    mask = np.zeros(spectra.shape, dtype=bool)
    if max_width is None:
        mask[:, :-1] = np.abs(z) > threshold
        return mask
    n_diff = z.shape[1]
    for r, i in zip(*np.nonzero(np.abs(z) > threshold)):
        if i > 0 and abs(z[r, i - 1]) > threshold:
            continue                          # the jump continues a run of large differences
        sign = 1.0 if z[r, i] > 0 else -1.0
        for j in range(min(i + max_width, n_diff - 1), i, -1):     # farthest closing jump first
            if -sign * z[r, j] > threshold and (j + 1 >= n_diff or abs(z[r, j + 1]) <= threshold):
                mask[r, i + 1:j + 1] = True
                break
    return mask


def _replace_by_neighbour_mean(y: np.ndarray, flagged: np.ndarray, m: int,
                               reference: np.ndarray | None = None) -> np.ndarray:
    """Replace flagged samples by the mean of the unflagged samples within ``+/- m``.

    A sample without unflagged neighbours takes the ``reference`` value when
    one is given (the rule of the residual pass) and raises otherwise.
    """
    out = y.copy()
    for k in np.flatnonzero(flagged):
        lo, hi = max(0, k - m), min(y.size, k + m + 1)
        good = ~flagged[lo:hi]
        if good.any():
            out[k] = y[lo:hi][good].mean()
        elif reference is not None:
            out[k] = reference[k]
        else:
            raise ValueError(f"sample {k} has no unflagged neighbour within +/-{m} samples; increase m")
    return out


def despike_whitaker_hayes(ssm: StandardSpectrumMap, threshold: float = 7.0, m: int = 3, *,
                           max_width: Optional[int] = 2, protect: Sequence[tuple[float, float]] = ()
                           ) -> StandardSpectrumMap:
    """Whitaker-Hayes despiking.

    Parameters
    ----------
    threshold : modified z-score of the first difference above which a jump counts.
    m : flagged samples are replaced by the mean of the unflagged samples
        within ``+/- m`` samples.
    max_width : widest spike (in samples) that is detected by the pair rule, or
        ``None`` for the published rule; see :func:`whitaker_hayes_mask`.
    protect : axis ranges ``(lo, hi)`` that are never modified (for known
        strong, narrow bands whose top would look like a spike). Empty by default.
    """
    if int(m) != m or m < 1:
        raise ValueError(f"m must be a positive integer; got {m}")
    mask = whitaker_hayes_mask(ssm.spectra, threshold, max_width) & ~range_mask(ssm.axis, protect)[None, :]
    out = np.array([_replace_by_neighbour_mean(y, f, int(m)) if f.any() else y
                    for y, f in zip(ssm.spectra, mask)])
    return ssm.replace(spectra=out)


# ---------------------------------------------------------- residual pass
def residual_reference(ssm: StandardSpectrumMap, window: int = 9, polyorder: int = 3) -> np.ndarray:
    """Savitzky-Golay reference curve ``(N, M)`` of the residual despike pass."""
    check_savgol(window, polyorder, ssm.n_points, "ref_window")
    return savgol_filter(ssm.spectra, int(window), int(polyorder), axis=1, mode="interp")


def despike_residual(ssm: StandardSpectrumMap, ref_window: int = 9, ref_polyorder: int = 3, threshold: float = 6.0,
                     m: int = 2, replace: str = "ref", protect: Sequence[tuple[float, float]] = ()
                     ) -> StandardSpectrumMap:
    """Residual z-score despiking.

    A weak Savitzky-Golay smooth of every spectrum is the reference; samples
    whose residual ``y - reference`` has a modified z-score ``|z| > threshold``
    are flagged and replaced either by the reference value (``replace="ref"``)
    or by the mean of the unflagged samples within ``+/- m`` (``"mean"``;
    a sample with no unflagged neighbour there takes the reference value).
    The reference is computed from the spectrum including the spike, so with
    ``"ref"`` the replaced value is pulled towards the spike. The tops of real
    bands narrower than about ``ref_window`` samples are also flagged.
    """
    if replace not in ("ref", "mean"):
        raise ValueError("replace must be 'ref' or 'mean'")
    if not threshold > 0:
        raise ValueError(f"threshold must be positive; got {threshold}")
    ref = residual_reference(ssm, ref_window, ref_polyorder)
    mask = (np.abs(modified_zscore(ssm.spectra - ref)) > threshold) & ~range_mask(ssm.axis, protect)[None, :]
    if replace == "ref":
        out = np.where(mask, ref, ssm.spectra)
    else:
        if int(m) != m or m < 1:
            raise ValueError(f"m must be a positive integer; got {m}")
        out = np.array([_replace_by_neighbour_mean(y, f, int(m), r) if f.any() else y
                        for y, f, r in zip(ssm.spectra, mask, ref)])
    return ssm.replace(spectra=out)


def despike_two_pass(ssm: StandardSpectrumMap, *, wh_threshold: float = 7.0, wh_m: int = 3,
                     wh_max_width: Optional[int] = 2,
                     ref_window: int = 9, ref_polyorder: int = 3, resid_threshold: float = 6.0, resid_m: int = 2,
                     resid_replace: str = "ref", protect: Sequence[tuple[float, float]] = ()
                     ) -> tuple[StandardSpectrumMap, StandardSpectrumMap, np.ndarray]:
    """Whitaker-Hayes pass followed by the residual pass.

    Returns ``(pass1, pass2, reference)``: the map after each pass and the
    Savitzky-Golay reference of the second pass (for QA plots).
    """
    pass1 = despike_whitaker_hayes(ssm, wh_threshold, wh_m, max_width=wh_max_width, protect=protect)
    reference = residual_reference(pass1, ref_window, ref_polyorder)
    pass2 = despike_residual(pass1, ref_window, ref_polyorder, resid_threshold, resid_m, resid_replace, protect)
    return pass1, pass2, reference
