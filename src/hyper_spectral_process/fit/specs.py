"""Peak specification (:class:`PeakSpec`), stage plan (:class:`StageSpec`) and :class:`FitConfig`.

A peak is described by initial values of ``pos``, ``width`` and ``height``
and, for each of them, a *limit* (the natural scale of its deviation) and a
*limit_reg* (the strength of the soft spring prior, see
:mod:`.objective`). Type-specific parameters go into ``extra``:

=====================  ========================================================
type                   parameters (``width`` meaning; ``extra`` keys)
=====================  ========================================================
``gaussian``           FWHM
``lorentzian``         FWHM
``pseudovoigt``        TCH pseudo-Voigt: ``width`` = Lorentzian FWHM Gamma_L;
                       ``width_g`` = Gaussian FWHM Gamma_G (default ``width``),
                       ``width_g_limit``, ``width_g_limit_reg`` (default: those of width)
``pseudovoigt_fixed``  shared FWHM; ``eta`` (required, fixed, 0 = Gaussian, 1 = Lorentzian)
``fano``               resonance FWHM; ``q`` (default 5), ``q_range`` (hard
                       range, default (-1e6, 1e6)), ``q_limit`` (default 1),
                       ``q_limit_reg`` (default 0)
=====================  ========================================================
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping, Optional, Sequence, Union

PEAK_TYPES = ("gaussian", "lorentzian", "pseudovoigt", "pseudovoigt_fixed", "fano")
_ALIASES = {
    "gaussian": ("gaussian", "gauss", "g"),
    "lorentzian": ("lorentzian", "lorentz", "cauchy", "l"),
    "pseudovoigt": ("pseudovoigt", "pseudo-voigt", "pseudo_voigt", "pv", "pvoigt", "voigt"),
    "pseudovoigt_fixed": ("pseudovoigt_fixed", "pseudo-voigt-fixed", "pvfixed", "pv_fixed", "pvf"),
    "fano": ("fano", "bwf", "breit-wigner-fano", "breit_wigner_fano", "breitwignerfano"),
}
_EXTRA_KEYS = {
    "gaussian": (), "lorentzian": (),
    "pseudovoigt": ("width_g", "width_g_limit", "width_g_limit_reg"),
    "pseudovoigt_fixed": ("eta",),
    "fano": ("q", "q_range", "q_limit", "q_limit_reg"),
}


def _norm_token(token: str) -> str:
    return str(token).strip().lower().replace("-", "").replace("_", "")


def canonical_peak_type(token: str) -> str:
    """Canonical type name for ``token`` (case, ``-`` and ``_`` ignored; aliases such as ``"pv"``, ``"bwf"``)."""
    t = _norm_token(token)
    for name, aliases in _ALIASES.items():
        if t in {_norm_token(a) for a in aliases}:
            return name
    raise ValueError(f"unsupported peak type {token!r}; allowed: {PEAK_TYPES} (and their aliases)")


def _positive(name: str, value: float) -> None:
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be positive and finite; got {value}")


def _nonnegative(name: str, value: float) -> None:
    if not (math.isfinite(value) and value >= 0):
        raise ValueError(f"{name} must be non-negative and finite; got {value}")


@dataclass(frozen=True)
class PeakSpec:
    """One peak: initial values, spring-prior limits and type-specific parameters.

    Parameters
    ----------
    type : peak type (see the module docstring); stored in canonical form.
    pos, width, height : initial position (axis units), width (axis units,
        meaning per type) and height (data units).
    pos_limit, width_limit, height_limit : natural scale of the deviation from
        the anchor value; the spring penalty equals ``*_limit_reg`` at one limit.
    pos_limit_reg, width_limit_reg, height_limit_reg : spring strengths (0 = free).
    extra : type-specific parameters (see the module docstring).
    name : label used in tables and plots (optional).
    """

    type: str
    pos: float
    width: float
    height: float
    pos_limit: float
    pos_limit_reg: float
    width_limit: float
    width_limit_reg: float
    height_limit: float
    height_limit_reg: float
    extra: Optional[Mapping] = None
    name: str = ""

    def __post_init__(self) -> None:
        t = canonical_peak_type(self.type)
        object.__setattr__(self, "type", t)
        if not math.isfinite(self.pos) or not math.isfinite(self.height):
            raise ValueError(f"pos and height must be finite; got {self.pos}, {self.height}")
        _positive("width", self.width)
        for n in ("pos_limit", "width_limit", "height_limit"):
            _positive(n, getattr(self, n))
        for n in ("pos_limit_reg", "width_limit_reg", "height_limit_reg"):
            _nonnegative(n, getattr(self, n))
        extra = dict(self.extra or {})
        unknown = set(extra) - set(_EXTRA_KEYS[t])
        if unknown:
            raise ValueError(f"extra keys {sorted(unknown)} do not apply to type {t!r} "
                             f"(allowed: {_EXTRA_KEYS[t]})")
        if t == "pseudovoigt_fixed":
            if "eta" not in extra or not 0.0 <= float(extra["eta"]) <= 1.0:
                raise ValueError("type 'pseudovoigt_fixed' requires extra={'eta': value in [0, 1]}")
        if t == "pseudovoigt":
            _positive("width_g", float(extra.get("width_g", self.width)))
            _positive("width_g_limit", float(extra.get("width_g_limit", self.width_limit)))
            _nonnegative("width_g_limit_reg", float(extra.get("width_g_limit_reg", self.width_limit_reg)))
        if t == "fano":
            lo, hi = (float(v) for v in extra.get("q_range", (-1e6, 1e6)))
            if not lo <= hi:
                raise ValueError(f"q_range must satisfy low <= high; got {(lo, hi)}")
            if not lo <= float(extra.get("q", 5.0)) <= hi:
                raise ValueError(f"initial q={extra.get('q', 5.0)} lies outside q_range {(lo, hi)}")
            _positive("q_limit", float(extra.get("q_limit", 1.0)))
            _nonnegative("q_limit_reg", float(extra.get("q_limit_reg", 0.0)))
        object.__setattr__(self, "extra", extra or None)

    # ------------------------------------------------------- typed accessors
    def get(self, key: str, default=None):
        """Value of an ``extra`` key with the documented default."""
        defaults = {"width_g": self.width, "width_g_limit": self.width_limit,
                    "width_g_limit_reg": self.width_limit_reg, "q": 5.0, "q_range": (-1e6, 1e6),
                    "q_limit": 1.0, "q_limit_reg": 0.0}
        return (self.extra or {}).get(key, defaults.get(key, default))

    @property
    def n_params(self) -> int:
        """Number of fitted parameters: 4 for ``pseudovoigt`` and ``fano``, 3 otherwise."""
        return 4 if self.type in ("pseudovoigt", "fano") else 3

    def with_values(self, **changes) -> "PeakSpec":
        """Copy with new values (e.g. refined ``pos``/``width``/``height``; ``extra`` merged key by key)."""
        extra = changes.pop("extra", None)
        merged = dict(self.extra or {})
        if extra:
            merged.update(extra)
        return replace(self, extra=merged or None, **changes)


@dataclass(frozen=True)
class StageSpec:
    """One optimisation stage: ``iters`` Adam steps at learning rate ``lr``.

    ``active`` is ``"all"`` or a sequence of peak indices (in the order used
    for fitting, i.e. after sorting by position when
    ``FitConfig.sort_peaks_by_pos``); inactive peaks contribute nothing to
    the model. With ``recenter_reg`` the spring anchors are moved to the
    current parameter values when the stage starts, so the prior pulls
    towards the result of the previous stage rather than the initial guess.
    """

    name: str
    iters: int
    lr: float
    active: Union[str, tuple] = "all"
    recenter_reg: bool = True

    def __post_init__(self) -> None:
        if int(self.iters) != self.iters or self.iters < 0:
            raise ValueError(f"iters must be a non-negative integer; got {self.iters}")
        _positive("lr", self.lr)
        if isinstance(self.active, str):
            if self.active != "all":
                raise ValueError(f"active must be 'all' or a sequence of peak indices; got {self.active!r}")
        else:
            idx = tuple(int(i) for i in self.active)
            if len(set(idx)) != len(idx) or not idx:
                raise ValueError(f"active must list distinct peak indices; got {self.active}")
            object.__setattr__(self, "active", idx)


@dataclass(frozen=True)
class FitConfig:
    """What to fit and how.

    Parameters
    ----------
    peaks : the peak specifications.
    stages : the stage plan (at least one stage).
    sort_peaks_by_pos : order the peaks by initial position (stage
        ``active`` indices refer to this order).
    base_init : initial value of the constant baseline ``base``.
    fit_baseline : fit ``base``; False keeps it at ``base_init``.
    x_fit_range : ``(xmin, xmax)`` of the fit window (either may be None);
        clipped to the data.
    downweight_x_ranges : axis ranges whose residuals are multiplied by
        ``downweight_factor`` in the loss (0 ignores them).
    height_sign_reg : weight of the penalty on negative heights.
    x_label, y_label : axis labels for plots.
    """

    peaks: tuple
    stages: tuple
    sort_peaks_by_pos: bool = True
    base_init: float = 0.0
    fit_baseline: bool = False
    x_fit_range: Optional[tuple] = None
    downweight_x_ranges: Optional[tuple] = None
    downweight_factor: float = 0.2
    height_sign_reg: float = 5.0
    x_label: str = "x"
    y_label: str = "Intensity"

    def __post_init__(self) -> None:
        peaks, stages = tuple(self.peaks), tuple(self.stages)
        if not peaks:
            raise ValueError("FitConfig needs at least one peak")
        if not stages:
            raise ValueError("FitConfig needs at least one StageSpec")
        for st in stages:
            if st.active != "all" and (min(st.active) < 0 or max(st.active) >= len(peaks)):
                raise ValueError(f"stage {st.name!r}: active indices {st.active} out of range for {len(peaks)} peaks")
        if not 0.0 <= self.downweight_factor <= 1.0:
            raise ValueError(f"downweight_factor must be in [0, 1]; got {self.downweight_factor}")
        for lo, hi in self.downweight_x_ranges or ():
            if not lo < hi:
                raise ValueError(f"downweight range ({lo}, {hi}) must satisfy lo < hi")
        _nonnegative("height_sign_reg", self.height_sign_reg)
        object.__setattr__(self, "peaks", peaks)
        object.__setattr__(self, "stages", stages)

    def fit_peaks(self) -> tuple:
        """Peaks in fitting order (sorted by ``pos`` if ``sort_peaks_by_pos``)."""
        return tuple(sorted(self.peaks, key=lambda p: p.pos)) if self.sort_peaks_by_pos else self.peaks

    def with_peaks(self, peaks: Sequence[PeakSpec]) -> "FitConfig":
        return replace(self, peaks=tuple(peaks))

    def n_params(self) -> int:
        """Number of fitted parameters per spectrum (the ``k`` of the BIC)."""
        return sum(p.n_params for p in self.peaks) + (1 if self.fit_baseline else 0)
