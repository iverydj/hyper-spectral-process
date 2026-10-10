"""Map-level preparation of a fit: pixel selection, low-signal skip, axis space, normalisation, window.

The steps, in order (:func:`prepare_map`):

1. *Selection* of the rows to fit (``all``, explicit ``indices`` or ``random``).
2. *Low-signal skip*: rows whose maximum is below ``skip_low_signal_factor``
   times the mean of all row maxima are not fitted.
3. *Axis space* (optional): a wavelength map (unit ``"nm"``) is converted to
   photon energy (``"eV"``) or wavenumber (``"cm^-1"``) with the intensity
   Jacobian ``(lambda / lambda_ref)^2`` (``lambda_ref`` = centre of the
   wavelength fit window); the peak specifications, the fit window and the
   down-weighted ranges given in nm are converted too.
4. *Normalisation* (optional): every row is mapped to [0, 1] by its own
   minimum and range over the full axis.
5. *Fit window* and per-sample weights (``downweight_x_ranges``).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from ..io import StandardSpectrumMap, energy_constant, wavelength_to_energy
from .specs import FitConfig, PeakSpec

AXIS_SPACES = ("native", "eV", "cm^-1")


@dataclass(frozen=True)
class MapFitOptions:
    """How :func:`~.fit_map.fit_map` treats the map (independent of the peak model).

    Parameters
    ----------
    select_mode : ``"all"``, ``"indices"`` (rows in ``indices``) or ``"random"``
        (``k_random`` rows drawn with ``seed``).
    normalize_01 : map every row to [0, 1] (minimum 0, maximum 1) before fitting.
    unnormalize_results : convert heights, baselines, areas, data and
        reconstruction back to the input intensity scale.
    skip_low_signal, skip_low_signal_factor : do not fit rows whose maximum is
        below ``factor * mean(row maxima)``.
    fit_axis_space : ``"native"`` or, for a map in nm, ``"eV"`` / ``"cm^-1"``.
    batch_size : spectra per optimisation batch.
    devices : torch devices, e.g. ``("cuda:0",)``; batches are spread over them.
    """

    select_mode: str = "all"
    indices: tuple = ()
    k_random: int = 5
    seed: int = 42
    normalize_01: bool = True
    unnormalize_results: bool = False
    skip_low_signal: bool = True
    skip_low_signal_factor: float = 0.1
    fit_axis_space: str = "native"
    batch_size: int = 256
    devices: tuple = ("cpu",)

    def __post_init__(self) -> None:
        if self.select_mode not in ("all", "indices", "random"):
            raise ValueError(f"select_mode must be 'all', 'indices' or 'random'; got {self.select_mode!r}")
        if self.fit_axis_space not in AXIS_SPACES:
            raise ValueError(f"fit_axis_space must be one of {AXIS_SPACES}; got {self.fit_axis_space!r}")
        if self.unnormalize_results and not self.normalize_01:
            raise ValueError("unnormalize_results=True requires normalize_01=True")


@dataclass(frozen=True)
class PreparedMap:
    """Data ready for fitting (all ``N`` rows; ``valid`` marks the rows to fit)."""

    ssm: StandardSpectrumMap           # converted/normalised map on the full axis
    fit_mask: np.ndarray               # (M,) samples inside the fit window
    weights: np.ndarray                # (L,) loss weights on the fit window
    valid: np.ndarray                  # (N,) selected and not low-signal
    low_signal: np.ndarray             # (N,)
    norm_min: Optional[np.ndarray]     # (N,) per-row offset of normalize_01
    norm_range: Optional[np.ndarray]   # (N,) per-row scale of normalize_01
    cfg: FitConfig                     # configuration in the fit space
    conversion: Optional[dict]         # axis conversion record (None for native)

    @property
    def x(self) -> np.ndarray:
        return self.ssm.axis[self.fit_mask]

    @property
    def y(self) -> np.ndarray:
        return self.ssm.spectra[:, self.fit_mask]


def select_rows(n_total: int, opts: MapFitOptions) -> np.ndarray:
    """Boolean mask of the rows chosen by ``opts.select_mode``."""
    sel = np.zeros(n_total, dtype=bool)
    if opts.select_mode == "all":
        sel[:] = True
    elif opts.select_mode == "indices":
        idx = np.asarray(opts.indices, dtype=int)
        if idx.size == 0 or idx.min() < 0 or idx.max() >= n_total:
            raise ValueError(f"indices must be non-empty and inside [0, {n_total}); got {opts.indices}")
        sel[idx] = True
    else:
        k = min(int(opts.k_random), n_total)
        sel[np.random.default_rng(opts.seed).choice(n_total, size=k, replace=False)] = True
    return sel


def low_signal_mask(spectra: np.ndarray, factor: float) -> np.ndarray:
    """Rows whose maximum is below ``factor * mean(row maxima)``."""
    row_max = spectra.max(axis=1)
    return row_max < factor * row_max.mean()


def fit_window_mask(axis: np.ndarray, x_range: Optional[tuple]) -> np.ndarray:
    """Samples inside ``x_range = (xmin, xmax)`` (either may be None), clipped to the axis."""
    if x_range is None:
        return np.ones(axis.size, dtype=bool)
    lo = axis[0] if x_range[0] is None else max(float(x_range[0]), axis[0])
    hi = axis[-1] if x_range[1] is None else min(float(x_range[1]), axis[-1])
    mask = (axis >= lo) & (axis <= hi)
    if mask.sum() < 2:
        raise ValueError(f"fit window {x_range} contains fewer than two samples of the axis [{axis[0]}, {axis[-1]}]")
    return mask


def downweight_weights(x: np.ndarray, ranges: Optional[tuple], factor: float) -> np.ndarray:
    """Weights 1, multiplied by ``factor`` inside every range of ``ranges``."""
    w = np.ones(x.size)
    for lo, hi in ranges or ():
        w[(x >= lo) & (x <= hi)] *= factor
    if not w.any():
        raise ValueError("every sample of the fit window has zero weight")
    return w


# ------------------------------------------------------------ axis space
def peak_to_energy(peak: PeakSpec, unit: str, reference_nm: float) -> PeakSpec:
    """Convert a peak given in nm to the energy axis ``unit`` (``"eV"`` or ``"cm^-1"``).

    ``pos`` maps exactly (``K / lambda``); widths and their limits are scaled
    by ``|dE/d lambda| = K / lambda^2`` at the peak; heights and their limit
    by the relative Jacobian ``(lambda / reference_nm)^2``. Dimensionless
    values (regs, ``q``, ``eta``) are unchanged.
    """
    k = energy_constant(unit)
    lam = float(peak.pos)
    if lam <= 0:
        raise ValueError(f"peak position {lam} nm must be positive")
    scale = k / lam**2
    jac = (lam / reference_nm) ** 2
    extra = dict(peak.extra or {})
    for key in ("width_g", "width_g_limit"):
        if key in extra:
            extra[key] = float(extra[key]) * scale
    return replace(peak, pos=k / lam, width=peak.width * scale, pos_limit=peak.pos_limit * scale,
                   width_limit=peak.width_limit * scale, height=peak.height * jac,
                   height_limit=peak.height_limit * jac, extra=extra or None)


def peak_to_wavelength(peak: PeakSpec, unit: str, reference_nm: float) -> PeakSpec:
    """Inverse of :func:`peak_to_energy` (used to report refined peaks in the input unit)."""
    k = energy_constant(unit)
    e = float(peak.pos)
    if e <= 0:
        raise ValueError(f"peak position {e} must be positive")
    lam = k / e
    scale = lam**2 / k
    jac = (lam / reference_nm) ** 2
    extra = dict(peak.extra or {})
    for key in ("width_g", "width_g_limit"):
        if key in extra:
            extra[key] = float(extra[key]) * scale
    return replace(peak, pos=lam, width=peak.width * scale, pos_limit=peak.pos_limit * scale,
                   width_limit=peak.width_limit * scale, height=peak.height / jac,
                   height_limit=peak.height_limit / jac, extra=extra or None)


def _range_to_energy(rng: tuple, k: float) -> tuple:
    lo, hi = rng
    return (None if hi is None else k / float(hi), None if lo is None else k / float(lo))


def config_to_energy(cfg: FitConfig, unit: str, reference_nm: float) -> FitConfig:
    """Convert peaks, fit window and down-weighted ranges of ``cfg`` from nm to ``unit``."""
    k = energy_constant(unit)
    return replace(cfg, peaks=tuple(peak_to_energy(p, unit, reference_nm) for p in cfg.peaks),
                   x_fit_range=None if cfg.x_fit_range is None else _range_to_energy(cfg.x_fit_range, k),
                   downweight_x_ranges=None if not cfg.downweight_x_ranges else
                   tuple(_range_to_energy(r, k) for r in cfg.downweight_x_ranges))


def prepare_map(ssm: StandardSpectrumMap, cfg: FitConfig, opts: MapFitOptions) -> PreparedMap:
    """Apply the preparation steps of the module docstring."""
    valid = select_rows(ssm.n_spectra, opts)
    low = low_signal_mask(ssm.spectra, opts.skip_low_signal_factor) if opts.skip_low_signal \
        else np.zeros(ssm.n_spectra, dtype=bool)
    valid &= ~low
    if not valid.any():
        raise ValueError("no spectrum left to fit after selection and the low-signal skip")
    conversion = None
    if opts.fit_axis_space != "native":
        if ssm.unit != "nm":
            raise ValueError(f"fit_axis_space={opts.fit_axis_space!r} needs a map in nm; its unit is {ssm.unit!r}")
        window = fit_window_mask(ssm.axis, cfg.x_fit_range)
        reference_nm = 0.5 * (ssm.axis[window][0] + ssm.axis[window][-1])
        ssm = wavelength_to_energy(ssm, opts.fit_axis_space, jacobian=True, reference_nm=reference_nm)
        cfg = config_to_energy(cfg, opts.fit_axis_space, reference_nm)
        conversion = {"from": "nm", "to": opts.fit_axis_space, "jacobian": "(lambda / reference_nm)**2",
                      "reference_nm": reference_nm}
    norm_min = norm_range = None
    if opts.normalize_01:
        norm_min = ssm.spectra.min(axis=1)
        norm_range = ssm.spectra.max(axis=1) - norm_min
        flat = np.flatnonzero(valid & ~(norm_range > 0))
        if flat.size:
            raise ValueError(f"{flat.size} selected spectra are flat (rows {flat[:5].tolist()}); cannot normalise")
        spectra = (ssm.spectra - norm_min[:, None]) / np.where(norm_range > 0, norm_range, 1.0)[:, None]
        spectra[low] = 0.0
        ssm = ssm.replace(spectra=spectra)
    fit_mask = fit_window_mask(ssm.axis, cfg.x_fit_range)
    weights = downweight_weights(ssm.axis[fit_mask], cfg.downweight_x_ranges, cfg.downweight_factor)
    return PreparedMap(ssm=ssm, fit_mask=fit_mask, weights=weights, valid=valid, low_signal=low,
                       norm_min=norm_min, norm_range=norm_range, cfg=cfg, conversion=conversion)
