"""Fit every spectrum of a map: :func:`fit_map`.

Steps: :func:`~.prepare.prepare_map` -> optional initial guess on an
averaged spectrum (:mod:`.initial_guess`) -> batched staged optimisation of
every valid row (:func:`~.engine.fit_batches`) -> optional un-normalisation
-> :class:`~.result.FitSpectrumMap`.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from ..io import StandardSpectrumMap
from .engine import fit_batches
from .initial_guess import InitialGuess, InitialGuessOptions, fit_initial_guess
from .lineshapes import peak_area
from .model import PeakTable
from .prepare import MapFitOptions, PreparedMap, prepare_map
from .result import FitSpectrumMap
from .specs import FitConfig


@dataclass(frozen=True)
class MapFit:
    """Output of :func:`fit_map`."""

    result: FitSpectrumMap
    loss_history: np.ndarray          # (n_batches, n_iter) mean loss of every batch
    stage_ends: tuple
    peaks: tuple                      # peaks used as start values (fitting order, fit space)
    input_peaks: tuple                # the configured peaks in the fit space (fitting order)
    initial_guess: Optional[InitialGuess]
    prepared: PreparedMap


def _metadata(cfg: FitConfig, opts: MapFitOptions, prepared: PreparedMap, peaks: tuple, n_fit: int,
              seconds: float) -> dict:
    return {
        "peak_names": [p.name for p in peaks],
        "peaks": [asdict(p) for p in peaks],
        "stages": [asdict(s) for s in cfg.stages],
        "fit_baseline": cfg.fit_baseline, "base_init": cfg.base_init, "height_sign_reg": cfg.height_sign_reg,
        "x_fit_range": cfg.x_fit_range, "downweight_x_ranges": cfg.downweight_x_ranges,
        "downweight_factor": cfg.downweight_factor,
        "normalize_01": opts.normalize_01, "unnormalized": opts.unnormalize_results,
        "skip_low_signal": opts.skip_low_signal, "skip_low_signal_factor": opts.skip_low_signal_factor,
        "fit_axis_space": opts.fit_axis_space, "axis_conversion": prepared.conversion,
        "n_params": cfg.n_params(), "n_fitted": n_fit, "batch_size": opts.batch_size,
        "devices": list(opts.devices), "fit_seconds": seconds,
    }


def fit_map(ssm: StandardSpectrumMap, cfg: FitConfig, opts: MapFitOptions = MapFitOptions(), *,
            initial_guess: Optional[InitialGuessOptions] = None, initial_guess_rows: Optional[np.ndarray] = None
            ) -> MapFit:
    """Fit the peak model of ``cfg`` to every selected spectrum of ``ssm``.

    Parameters
    ----------
    ssm : the map (typically preprocessed).
    cfg : peaks, stage plan and fit window.
    opts : selection, normalisation, axis space, batching and devices.
    initial_guess : if given, the peaks are first refined on an averaged
        spectrum (see :mod:`.initial_guess`) and every pixel starts from the
        refined values; ``initial_guess_rows`` fixes the averaged rows.

    Returns a :class:`MapFit`; ``result`` has one row per input spectrum
    (rows not fitted: ``valid_mask`` False, NaN parameters).
    """
    t0 = time.perf_counter()
    prepared = prepare_map(ssm, cfg, opts)
    fcfg = prepared.cfg
    input_peaks = fcfg.fit_peaks()
    guess = None
    peaks = input_peaks
    if initial_guess is not None or initial_guess_rows is not None:
        guess = fit_initial_guess(prepared, initial_guess or InitialGuessOptions(), device=opts.devices[0],
                                  rows=initial_guess_rows)
        peaks = guess.peaks
    rows = np.flatnonzero(prepared.valid)
    table = PeakTable.from_peaks(peaks)
    run_cfg = fcfg.with_peaks(peaks)
    res, hist, ends = fit_batches(prepared.x, prepared.y[rows], prepared.weights, table, run_cfg,
                                  devices=opts.devices, batch_size=opts.batch_size)
    n, p, length = ssm.n_spectra, len(peaks), prepared.x.size

    def full(values, shape):
        out = np.full(shape, np.nan)
        out[rows] = values
        return out

    out = {k: full(res[k], (n, p)) for k in ("pos", "width", "height", "width_g", "q", "eta")}
    base, recon, loss = full(res["base"], (n,)), full(res["recon"], (n, length)), full(res["loss_final"], (n,))
    area = np.full((n, p), np.nan)
    for j, t in enumerate(table.types):
        area[:, j] = peak_area(t, out["height"][:, j], out["width"][:, j], out["eta"][:, j], out["width_g"][:, j])
    data = prepared.y.copy()
    if opts.unnormalize_results:
        s, o = prepared.norm_range, prepared.norm_min
        out["height"] *= s[:, None]
        area *= s[:, None]
        base = base * s + o
        recon = recon * s[:, None] + o[:, None]
        data = data * s[:, None] + o[:, None]
    any_tch = "pseudovoigt" in table.types
    any_fano = "fano" in table.types
    for key, ptype in (("width_g", "pseudovoigt"), ("q", "fano")):
        other = np.flatnonzero([t != ptype for t in table.types])
        out[key][np.ix_(rows, other)] = 0.0       # not applicable to these peak types
    result = FitSpectrumMap(
        axis=prepared.x, unit=prepared.ssm.unit, xy=ssm.xy, spectra_original=data, peak_types=table.types,
        pos=out["pos"], width=out["width"], height=out["height"], eta=out["eta"], base=base,
        valid_mask=prepared.valid.copy(), recon=recon, q=out["q"] if any_fano else None,
        width_g=out["width_g"] if any_tch else None, area=area, loss_final=loss,
        metadata=_metadata(run_cfg, opts, prepared, peaks, rows.size, time.perf_counter() - t0))
    return MapFit(result=result, loss_history=hist, stage_ends=ends, peaks=peaks, input_peaks=input_peaks,
                  initial_guess=guess, prepared=prepared)
