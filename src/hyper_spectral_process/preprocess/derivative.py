"""Derivative stage: Savitzky-Golay derivatives saved as extra variants.

For every order ``k`` the spectra are optionally pre-smoothed, then
differentiated with a Savitzky-Golay filter (``deriv=k``), then optionally
post-smoothed. The derivative is taken of the data as they are at the
position of ``"deriv"`` in the stage order; stages listed after it act on the
non-derivative spectra only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import savgol_filter

from ..io import StandardSpectrumMap
from ._common import check_savgol, savgol_passes


@dataclass(frozen=True)
class SmoothBlock:
    """Optional Savitzky-Golay smoothing around a derivative.

    ``strength`` passes of a ``window``/``polyorder`` filter; nothing is done
    when ``enable`` is False or ``strength`` is 0.
    """

    enable: bool = False
    strength: int = 0
    window: int = 11
    polyorder: int = 3

    @property
    def passes(self) -> int:
        return int(self.strength) if self.enable else 0


@dataclass(frozen=True)
class DerivativeSteps:
    """The intermediate results of :func:`savgol_derivative`."""

    final: StandardSpectrumMap                 # after post-smoothing (or the plain derivative)
    plain: StandardSpectrumMap                 # Savitzky-Golay derivative before post-smoothing
    pre: Optional[StandardSpectrumMap] = None  # input after pre-smoothing, if any
    post: Optional[StandardSpectrumMap] = None  # same as ``final`` when post-smoothing ran


def axis_delta(axis: np.ndarray, use_axis_delta: bool = True, delta_override: float = 1.0) -> float:
    """Sample spacing for the derivative: the median axis step, or ``delta_override``.

    With ``use_axis_delta=False`` (or a non-positive median step) the
    derivative is taken per ``delta_override`` axis units per sample, e.g.
    per sample for ``1.0``.
    """
    if use_axis_delta:
        step = float(np.median(np.diff(axis)))
        if step > 0:
            return step
    if not delta_override > 0:
        raise ValueError(f"delta_override must be positive; got {delta_override}")
    return float(delta_override)


def savgol_derivative(ssm: StandardSpectrumMap, order: int, window: int = 11, polyorder: int = 3, *,
                      delta: float = 1.0, pre: SmoothBlock = SmoothBlock(), post: SmoothBlock = SmoothBlock()
                      ) -> DerivativeSteps:
    """Savitzky-Golay derivative of order ``order`` with optional pre/post smoothing.

    ``delta`` is the sample spacing (see :func:`axis_delta`). The filter
    assumes uniform spacing; interpolate a non-uniform axis first if a true
    ``d^k y / dx^k`` is needed.
    """
    if int(order) != order or order < 1:
        raise ValueError(f"derivative order must be a positive integer; got {order}")
    check_savgol(window, polyorder, ssm.n_points)
    if polyorder < order:
        raise ValueError(f"polyorder ({polyorder}) must be >= the derivative order ({order})")
    y = ssm.spectra
    pre_map = None
    if pre.passes:
        y = savgol_passes(y, pre.window, pre.polyorder, pre.passes)
        pre_map = ssm.replace(spectra=y)
    d = savgol_filter(y, int(window), int(polyorder), deriv=int(order), delta=float(delta), axis=1, mode="interp")
    plain = ssm.replace(spectra=d)
    if not post.passes:
        return DerivativeSteps(final=plain, plain=plain, pre=pre_map)
    post_map = ssm.replace(spectra=savgol_passes(d, post.window, post.polyorder, post.passes))
    return DerivativeSteps(final=post_map, plain=plain, pre=pre_map, post=post_map)
