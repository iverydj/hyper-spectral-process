"""The fit result container :class:`FitSpectrumMap`, its NPZ format and derived quantities.

NPZ keys (schema ``FitSpectrumMap.v4``; strings are stored as unicode
arrays, so the file loads with ``allow_pickle=False``)::

    __schema__, axis (L,), unit, xy (N, 2), spectra_original (N, L),
    peak_types (P,), params_pos / params_width / params_height / params_eta (N, P),
    params_base (N,), valid_mask (N,), metadata_json,
    recon (N, L), params_q (N, P), params_width_g (N, P),       (empty arrays if absent)
    params_area (N, P), loss_final (N,), composite_active_mask (N, P)   (optional)

Rows that were not fitted have ``valid_mask = False`` and NaN parameters.
``params_width`` is the FWHM for Gaussian, Lorentzian and fixed-eta
pseudo-Voigt peaks, the Lorentzian FWHM for TCH pseudo-Voigt peaks (whose
Gaussian FWHM is ``params_width_g``) and the resonance FWHM for Fano peaks.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .lineshapes import peak_area, peak_fwhm
from .model import PeakModel

SCHEMA = "FitSpectrumMap.v4"


@dataclass(frozen=True)
class FitSpectrumMap:
    """Per-spectrum fit parameters of a map, the fitted data and the reconstruction.

    ``axis``/``unit`` are those of the fit (the fit window, in the fit
    space). ``metadata`` records the configuration of the fit (peak names,
    stage plan, normalisation, axis conversion, ...).
    """

    axis: np.ndarray
    unit: str
    xy: np.ndarray
    spectra_original: np.ndarray
    peak_types: tuple
    pos: np.ndarray
    width: np.ndarray
    height: np.ndarray
    eta: np.ndarray
    base: np.ndarray
    valid_mask: np.ndarray
    metadata: dict = field(default_factory=dict)
    recon: Optional[np.ndarray] = None
    q: Optional[np.ndarray] = None
    width_g: Optional[np.ndarray] = None
    area: Optional[np.ndarray] = None
    loss_final: Optional[np.ndarray] = None
    active_mask: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        n, p = np.shape(self.pos)
        for name in ("width", "height", "eta"):
            if np.shape(getattr(self, name)) != (n, p):
                raise ValueError(f"{name} must have shape ({n}, {p})")
        if len(self.peak_types) != p:
            raise ValueError(f"peak_types has {len(self.peak_types)} entries for {p} peaks")
        if np.shape(self.spectra_original) != (n, np.size(self.axis)) or np.shape(self.xy) != (n, 2):
            raise ValueError("spectra_original, xy and axis do not match the parameters")
        object.__setattr__(self, "peak_types", tuple(str(t) for t in self.peak_types))

    @property
    def n_spectra(self) -> int:
        return int(self.pos.shape[0])

    @property
    def n_peaks(self) -> int:
        return int(self.pos.shape[1])

    @property
    def peak_names(self) -> tuple:
        names = self.metadata.get("peak_names") or [""] * self.n_peaks
        return tuple(n or f"P{j + 1}" for j, n in enumerate(names))

    def peak_index(self, name: str) -> int:
        """Column of the peak called ``name``."""
        if name not in self.peak_names:
            raise KeyError(f"no peak named {name!r}; peaks: {self.peak_names}")
        return self.peak_names.index(name)

    def replace(self, **changes) -> "FitSpectrumMap":
        return replace(self, **changes)

    # ------------------------------------------------------------- model
    def components(self, rows=None, axis: Optional[np.ndarray] = None) -> np.ndarray:
        """Peak curves ``height * profile`` ``(n, P, L)`` of the given rows (default: all) on ``axis``.

        Peaks whose parameters are NaN (rows not fitted, composite slots not
        used) give NaN curves.
        """
        rows = np.arange(self.n_spectra) if rows is None else np.atleast_1d(rows)
        x = self.axis if axis is None else np.asarray(axis, dtype=np.float64)
        model = PeakModel(self.peak_types, np.zeros(self.n_peaks), x, torch.device("cpu"))
        zeros = np.zeros_like(self.pos)

        def t(a):
            return torch.as_tensor(np.asarray(a, dtype=np.float32)[rows][:, :, None])

        with torch.no_grad():
            peaks = model.peaks(t(self.pos), t(self.width), t(self.height),
                                t(self.width_g if self.width_g is not None else zeros),
                                t(self.q if self.q is not None else zeros), eta=t(self.eta))
        return peaks.numpy().astype(np.float64)

    def reconstruct(self, rows=None, axis: Optional[np.ndarray] = None) -> np.ndarray:
        """Model ``base + sum of peaks`` ``(n, L)`` from the stored parameters.

        Unused composite slots (NaN) are left out; rows not fitted are NaN.
        """
        rows = np.arange(self.n_spectra) if rows is None else np.atleast_1d(rows)
        comp = self.components(rows, axis)
        if self.active_mask is not None:
            comp = np.where(self.active_mask[rows][:, :, None], comp, 0.0)
        return self.base[rows][:, None] + comp.sum(axis=1)

    def peak_areas(self) -> np.ndarray:
        """Analytic area of every peak ``(N, P)`` (:func:`~.lineshapes.peak_area`; NaN for Fano)."""
        out = np.full(self.pos.shape, np.nan)
        for j, t in enumerate(self.peak_types):
            out[:, j] = peak_area(t, self.height[:, j], self.width[:, j], self.eta[:, j],
                                  None if self.width_g is None else self.width_g[:, j])
        return out

    def peak_fwhms(self) -> np.ndarray:
        """Profile FWHM of every peak ``(N, P)`` (:func:`~.lineshapes.peak_fwhm`)."""
        out = np.full(self.pos.shape, np.nan)
        for j, t in enumerate(self.peak_types):
            out[:, j] = peak_fwhm(t, self.width[:, j], None if self.width_g is None else self.width_g[:, j],
                                  None if self.q is None else self.q[:, j])
        return out

    # --------------------------------------------------------------- I/O
    def save(self, path: str | Path) -> None:
        """Write the NPZ file described in the module docstring."""
        def f32(a):
            return np.array([], dtype=np.float32) if a is None else np.asarray(a, dtype=np.float32)

        extra = {}
        for key, val in (("params_area", self.area), ("loss_final", self.loss_final)):
            if val is not None:
                extra[key] = f32(val)
        if self.active_mask is not None:
            extra["composite_active_mask"] = np.asarray(self.active_mask, dtype=bool)
        np.savez_compressed(
            path, __schema__=np.array(SCHEMA), axis=np.asarray(self.axis, dtype=np.float64), unit=np.array(self.unit),
            xy=np.asarray(self.xy, dtype=np.float64), spectra_original=f32(self.spectra_original),
            peak_types=np.array(self.peak_types), params_pos=f32(self.pos), params_width=f32(self.width),
            params_height=f32(self.height), params_eta=f32(self.eta), params_base=f32(self.base),
            valid_mask=np.asarray(self.valid_mask, dtype=bool), metadata_json=np.array(json.dumps(self.metadata)),
            recon=f32(self.recon), peaks_3d=np.array([], dtype=np.float32), params_q=f32(self.q),
            params_width_g=f32(self.width_g), **extra)

    @classmethod
    def load(cls, path: str | Path, *, allow_pickle: bool = False) -> "FitSpectrumMap":
        """Read a file written by :meth:`save`.

        Files written by the original code store strings as object arrays;
        pass ``allow_pickle=True`` for such files from a trusted source.
        """
        with np.load(path, allow_pickle=allow_pickle) as z:
            schema = str(z["__schema__"])
            if schema not in ("FitSpectrumMap.v1", SCHEMA):
                raise ValueError(f"{path}: unsupported schema {schema!r}")

            def opt(key):
                if key not in z.files or z[key].size == 0:
                    return None
                return z[key]

            return cls(axis=z["axis"], unit=str(z["unit"]), xy=z["xy"], spectra_original=z["spectra_original"],
                       peak_types=tuple(str(t) for t in z["peak_types"]), pos=z["params_pos"],
                       width=z["params_width"], height=z["params_height"], eta=z["params_eta"],
                       base=z["params_base"], valid_mask=z["valid_mask"].astype(bool),
                       metadata=json.loads(str(z["metadata_json"])), recon=opt("recon"), q=opt("params_q"),
                       width_g=opt("params_width_g"), area=opt("params_area"), loss_final=opt("loss_final"),
                       active_mask=None if opt("composite_active_mask") is None
                       else z["composite_active_mask"].astype(bool))


# ---------------------------------------------------------------- metrics
def fit_metrics(fsm: FitSpectrumMap) -> dict:
    """Per-spectrum quality numbers of the stored reconstruction (NaN where not fitted).

    ``wrmse``: root mean squared residual; ``wnrmse``: wRMSE divided by the
    data range; ``wr2``: ``1 - SS_res / SS_tot`` (set to 0 when negative, as in
    the original); ``bias``: mean of ``recon - data``. All samples have weight
    1 here, whatever the fit weights were.
    """
    if fsm.recon is None:
        raise ValueError("the result has no reconstruction")
    y, yhat = np.asarray(fsm.spectra_original, np.float64), np.asarray(fsm.recon, np.float64)
    err = yhat - y
    with np.errstate(invalid="ignore", divide="ignore"):
        wrmse = np.sqrt(np.mean(err**2, axis=1))
        yrange = np.ptp(y, axis=1)
        yrange = np.where(yrange > 1e-12, yrange, np.maximum(np.abs(y).max(axis=1), 1.0))
        ss_res = np.sum(err**2, axis=1)
        ss_tot = np.sum((y - y.mean(axis=1, keepdims=True)) ** 2, axis=1)
        wr2 = np.where(ss_tot <= 1e-12, 1.0, np.where(ss_res <= ss_tot, 1.0 - ss_res / ss_tot, 0.0))
    out = {"wrmse": wrmse, "wnrmse": wrmse / yrange, "wr2": wr2, "bias": err.mean(axis=1)}
    invalid = ~np.asarray(fsm.valid_mask, dtype=bool)
    for v in out.values():
        v[invalid] = np.nan
    return out


def write_params_csv(fsm: FitSpectrumMap, path: str | Path) -> None:
    """One row per spectrum: validity, coordinates, baseline and every peak's type/pos/width/height/eta/area."""
    area = fsm.peak_areas()
    headers = ["idx", "valid", "coord_x", "coord_y", "baseline"]
    for j, name in enumerate(fsm.peak_names):
        headers += [f"{name}_{k}" for k in ("type", "pos", "width", "height", "eta", "area")]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        for i in range(fsm.n_spectra):
            row = [i, int(fsm.valid_mask[i]), f"{fsm.xy[i, 0]:.8g}", f"{fsm.xy[i, 1]:.8g}", _num(fsm.base[i])]
            for j, t in enumerate(fsm.peak_types):
                row += [t] + [_num(a[i, j]) for a in (fsm.pos, fsm.width, fsm.height, fsm.eta, area)]
            w.writerow(row)


def write_metrics_csv(fsm: FitSpectrumMap, path: str | Path) -> None:
    """One row per spectrum: wRMSE, wNRMSE, wR2, residual bias and the data range."""
    m = fit_metrics(fsm)
    y = np.asarray(fsm.spectra_original, np.float64)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "valid", "coord_x", "coord_y", "wRMSE", "wNRMSE", "wR2", "residual_bias", "y_min", "y_max"])
        for i in range(fsm.n_spectra):
            w.writerow([i, int(fsm.valid_mask[i]), f"{fsm.xy[i, 0]:.8g}", f"{fsm.xy[i, 1]:.8g}"]
                       + [_num(v) for v in (m["wrmse"][i], m["wnrmse"][i], m["wr2"][i], m["bias"][i],
                                            y[i].min(), y[i].max())])


def _num(v) -> str:
    v = float(v)
    return f"{v:.10g}" if np.isfinite(v) else ""
