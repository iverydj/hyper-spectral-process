"""Model variants: fit several peak sets and choose one per pixel by BIC.

1. :func:`fit_variants` fits every variant (a named tuple of peaks) to the
   same map, window and options; each variant gets its own initial guess on
   the same averaged spectrum.
2. :func:`select_variants` computes for every pixel and variant

       BIC = L ln(SSR / L) + k ln(L)

   from the stored data and reconstruction on the ``L`` window samples, with
   ``k`` the number of fitted parameters (3 per Gaussian, Lorentzian or
   fixed-eta pseudo-Voigt peak, 4 per TCH pseudo-Voigt or Fano peak, plus 1
   for a fitted baseline). Optionally a drift penalty is added:

       penalty = alpha * sum_j max(0, |pos_j - pos_j,input| / pos_limit_j - 1)^2,

   which penalises a variant whose peaks moved further than their
   ``pos_limit`` from the input positions (a sign that the variant cannot
   describe the pixel). The variant with the lowest value wins.
3. :func:`composite_fit` assembles the winners into one
   :class:`~.result.FitSpectrumMap` whose peak slots are those of the largest
   variant; every variant peak is mapped to the slot with the closest input
   position. Slots a pixel's winner does not use are NaN and
   ``active_mask`` tells which slots are used.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from ..io import StandardSpectrumMap
from .fit_map import MapFit, fit_map
from .initial_guess import InitialGuessOptions, average_rows
from .prepare import MapFitOptions, prepare_map
from .result import FitSpectrumMap
from .specs import FitConfig, PeakSpec

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class VariantSelection:
    """Per-pixel choice between ``K`` variants (arrays ``(K, N)`` unless noted)."""

    names: tuple
    variant_id: np.ndarray        # (N,) index of the winner, -1 where any variant is invalid
    k_params: np.ndarray          # (K,)
    bic: np.ndarray               # BIC including the drift penalty (inf where invalid)
    bic_raw: np.ndarray           # BIC without the penalty
    wrmse: np.ndarray
    drift_penalty: np.ndarray
    drift_max: np.ndarray         # largest |pos - pos_input| / pos_limit of each pixel
    valid_mask: np.ndarray        # (N,) valid in every variant
    drift_penalty_alpha: float

    def delta_bic(self) -> np.ndarray:
        """``BIC(runner-up) - BIC(winner)`` of every pixel (NaN where invalid); small = ambiguous."""
        s = np.sort(self.bic, axis=0)
        out = np.full(self.variant_id.shape, np.nan)
        out[self.valid_mask] = (s[1] - s[0])[self.valid_mask]
        return out

    def save(self, path: str | Path, xy: np.ndarray) -> None:
        np.savez_compressed(path, variant_id=self.variant_id, variant_names=np.array(self.names),
                            k_active=self.k_params, bic_per_variant=self.bic, bic_raw_per_variant=self.bic_raw,
                            wrmse_per_variant=self.wrmse, drift_pen_per_variant=self.drift_penalty,
                            drift_max_per_variant=self.drift_max, drift_penalty_alpha=self.drift_penalty_alpha,
                            valid_mask=self.valid_mask, xy=np.asarray(xy, dtype=np.float64))


def fit_variants(ssm: StandardSpectrumMap, variants: Sequence[tuple], cfg: FitConfig,
                 opts: MapFitOptions = MapFitOptions(), *,
                 initial_guess: Optional[InitialGuessOptions] = InitialGuessOptions()) -> dict:
    """Fit every ``(name, peaks)`` variant with ``cfg`` (whose own peaks are replaced); returns ``{name: MapFit}``.

    With ``initial_guess`` all variants are refined on the same averaged spectrum.
    """
    names = [n for n, _ in variants]
    if len(set(names)) != len(names) or len(names) < 2:
        raise ValueError(f"need at least two variants with distinct names; got {names}")
    rows = None
    if initial_guess is not None:
        rows = average_rows(prepare_map(ssm, cfg.with_peaks(variants[0][1]), opts), initial_guess)
    fits = {}
    for name, peaks in variants:
        log.info("variant %r: %d peaks", name, len(peaks))
        fits[name] = fit_map(ssm, cfg.with_peaks(tuple(peaks)), opts, initial_guess_rows=rows)
    return fits


def _n_params(fsm: FitSpectrumMap) -> int:
    k = sum(4 if t in ("pseudovoigt", "fano") else 3 for t in fsm.peak_types)
    return k + (1 if fsm.metadata.get("fit_baseline") else 0)


def select_variants(fits: Mapping[str, FitSpectrumMap], *,
                    input_peaks: Optional[Mapping[str, Sequence[PeakSpec]]] = None,
                    drift_penalty_alpha: float = 0.0) -> VariantSelection:
    """Choose the variant with the lowest (drift-penalised) BIC for every pixel (see the module docstring).

    ``input_peaks`` maps a variant name to its input peaks in the fit space
    and fitting order (``MapFit.input_peaks``); required when
    ``drift_penalty_alpha > 0``.
    """
    names = list(fits)
    if len(names) < 2:
        raise ValueError("variant selection needs at least two variants")
    fsms = [fits[n] for n in names]
    ref = fsms[0]
    for n, f in zip(names, fsms):
        if f.axis.shape != ref.axis.shape or not np.allclose(f.axis, ref.axis) or f.n_spectra != ref.n_spectra:
            raise ValueError(f"variant {n!r} was fitted on a different window or map")
        if f.recon is None:
            raise ValueError(f"variant {n!r} has no reconstruction")
        if bool(f.metadata.get("fit_baseline")) != bool(ref.metadata.get("fit_baseline")):
            raise ValueError("all variants must use the same fit_baseline setting")
    if drift_penalty_alpha < 0:
        raise ValueError("drift_penalty_alpha must be >= 0")
    if drift_penalty_alpha > 0 and input_peaks is None:
        raise ValueError("drift_penalty_alpha > 0 requires input_peaks")
    k_var, n = len(names), ref.n_spectra
    length = ref.axis.size
    valid = np.logical_and.reduce([np.asarray(f.valid_mask, dtype=bool) for f in fsms])
    bic_raw = np.full((k_var, n), np.inf)
    wrmse = np.full((k_var, n), np.nan)
    pen = np.zeros((k_var, n))
    drift_max = np.zeros((k_var, n))
    k_params = np.array([_n_params(f) for f in fsms])
    for v, (name, f) in enumerate(zip(names, fsms)):
        diff = np.asarray(f.spectra_original, np.float64) - np.asarray(f.recon, np.float64)
        mse = np.nanmean(diff * diff, axis=1)
        wrmse[v] = np.sqrt(mse)
        with np.errstate(divide="ignore", invalid="ignore"):
            b = length * np.log(np.clip(mse, 1e-300, None)) + k_params[v] * np.log(length)
        b[~np.asarray(f.valid_mask, dtype=bool)] = np.inf
        bic_raw[v] = b
        if drift_penalty_alpha > 0:
            peaks = list(input_peaks[name])
            if len(peaks) != f.n_peaks:
                raise ValueError(f"variant {name!r}: {len(peaks)} input peaks for {f.n_peaks} fitted peaks")
            for j, p in enumerate(peaks):
                drift = np.nan_to_num(np.abs(f.pos[:, j] - p.pos) / p.pos_limit, nan=0.0)
                pen[v] += np.maximum(0.0, drift - 1.0) ** 2
                drift_max[v] = np.maximum(drift_max[v], drift)
            pen[v] *= drift_penalty_alpha
    bic_adj = np.where(np.isfinite(bic_raw), bic_raw + pen, np.inf)
    variant_id = np.full(n, -1, dtype=np.int64)
    variant_id[valid] = np.argmin(bic_adj[:, valid], axis=0)
    _warn(names, wrmse, bic_adj, valid)
    return VariantSelection(names=tuple(names), variant_id=variant_id, k_params=k_params, bic=bic_adj,
                            bic_raw=bic_raw, wrmse=wrmse, drift_penalty=pen, drift_max=drift_max, valid_mask=valid,
                            drift_penalty_alpha=float(drift_penalty_alpha))


def _warn(names, wrmse, bic, valid) -> None:
    if not valid.any():
        log.warning("no pixel is valid in every variant")
        return
    med = np.nanmedian(wrmse[:, valid], axis=1)
    for name, m in zip(names, med):
        if m > 5.0 * med.min():
            log.warning("variant %r: median wRMSE %.4g is more than 5x the best (%.4g); possibly degenerate",
                        name, m, med.min())
    s = np.sort(bic[:, valid], axis=0)
    tie = float(np.mean(s[1] - s[0] < 1.0))
    if tie >= 0.05:
        log.warning("%.1f %% of the pixels have a BIC margin below 1 between the two best variants", 100 * tie)


def slot_map(input_peaks: Mapping[str, Sequence[PeakSpec]]) -> tuple[str, dict]:
    """Reference variant (most peaks) and, per variant, the reference slot of each of its peaks.

    ``input_peaks`` are in fitting order (the column order of the results).
    Every peak is matched, in that order, to the unused reference slot with
    the closest input position; a peak and its slot must have the same type.
    """
    names = list(input_peaks)
    ref = names[int(np.argmax([len(input_peaks[n]) for n in names]))]
    ref_peaks = list(input_peaks[ref])
    out = {}
    for n in names:
        used, slots = set(), []
        for p in input_peaks[n]:
            cand = [(abs(p.pos - ref_peaks[s].pos), s) for s in range(len(ref_peaks)) if s not in used]
            if not cand:
                raise ValueError(f"variant {n!r} has more peaks than the reference variant {ref!r}")
            s = min(cand)[1]
            if ref_peaks[s].type != p.type:
                raise ValueError(f"variant {n!r}: peak at {p.pos:g} ({p.type}) maps to slot {s} of type "
                                 f"{ref_peaks[s].type}; composite slots need matching types")
            used.add(s)
            slots.append(s)
        out[n] = slots
    return ref, out


def composite_fit(fits: Mapping[str, FitSpectrumMap], selection: VariantSelection,
                  input_peaks: Mapping[str, Sequence[PeakSpec]]) -> FitSpectrumMap:
    """One result holding every pixel's winning variant (see the module docstring).

    ``input_peaks`` are the variants' input peaks in fitting order (sorted by
    position), as in :func:`select_variants`.
    """
    names = list(selection.names)
    ref_name, slots = slot_map({n: input_peaks[n] for n in names})
    ref = fits[ref_name]
    n, p = ref.n_spectra, ref.n_peaks
    arrays = {k: np.full((n, p), np.nan) for k in ("pos", "width", "height", "eta", "q", "width_g", "area")}
    base, recon = np.full(n, np.nan), np.full(ref.spectra_original.shape, np.nan)
    active = np.zeros((n, p), dtype=bool)
    valid = np.zeros(n, dtype=bool)
    has = {"q": any(fits[m].q is not None for m in names), "width_g": any(fits[m].width_g is not None for m in names)}
    for v, name in enumerate(names):
        f = fits[name]
        px = (selection.variant_id == v) & selection.valid_mask
        if not px.any():
            continue
        area = f.area if f.area is not None else f.peak_areas()
        for j, s in enumerate(slots[name]):
            for key, src in (("pos", f.pos), ("width", f.width), ("height", f.height), ("eta", f.eta),
                             ("area", area), ("q", f.q), ("width_g", f.width_g)):
                arrays[key][px, s] = 0.0 if src is None else src[px, j]
            active[px, s] = True
        base[px], recon[px], valid[px] = f.base[px], f.recon[px], True
    meta = dict(ref.metadata)
    meta.update({"composite": True, "variant_names": names, "reference_variant": ref_name,
                 "slot_map": {k: list(v) for k, v in slots.items()},
                 "peak_names": [q.name for q in input_peaks[ref_name]]})
    return FitSpectrumMap(axis=ref.axis, unit=ref.unit, xy=ref.xy, spectra_original=ref.spectra_original,
                          peak_types=ref.peak_types, pos=arrays["pos"], width=arrays["width"],
                          height=arrays["height"], eta=arrays["eta"], base=base, valid_mask=valid, metadata=meta,
                          recon=recon, q=arrays["q"] if has["q"] else None,
                          width_g=arrays["width_g"] if has["width_g"] else None, area=arrays["area"],
                          active_mask=active)


def input_peaks_of(fits: Mapping[str, MapFit]) -> dict:
    """``{name: MapFit.input_peaks}`` for :func:`select_variants` and :func:`composite_fit`."""
    return {n: f.input_peaks for n, f in fits.items()}
