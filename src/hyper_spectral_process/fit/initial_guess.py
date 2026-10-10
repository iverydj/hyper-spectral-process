"""Initial guess: fit the peak model to an averaged spectrum before fitting every pixel.

The average of a set of prepared spectra (all valid rows, explicit rows or
a random subset) is fitted with the full stage plan; the fitted ``pos``,
``width``, ``height`` (and ``width_g``, ``q``) replace the initial values of
the peak specifications, while their limits and regs are kept. The map fit
then starts every pixel from these refined values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .engine import BatchFitter, resolve_devices
from .model import PeakTable
from .prepare import PreparedMap, peak_to_wavelength
from .specs import FitConfig, PeakSpec


@dataclass(frozen=True)
class InitialGuessOptions:
    """Which prepared spectra are averaged: ``"all"`` valid rows, ``"indices"`` or ``"random"``."""

    select_mode: str = "random"
    indices: tuple = (0, 10, 20)
    k_random: int = 20
    seed: int = 123

    def __post_init__(self) -> None:
        if self.select_mode not in ("all", "indices", "random"):
            raise ValueError(f"select_mode must be 'all', 'indices' or 'random'; got {self.select_mode!r}")


@dataclass(frozen=True)
class InitialGuess:
    """Result of :func:`fit_initial_guess` (arrays on the fit window)."""

    rows: np.ndarray            # averaged rows
    x: np.ndarray
    y: np.ndarray               # averaged spectrum
    recon_before: np.ndarray    # model with the input peaks
    recon: np.ndarray           # model after the fit
    components: np.ndarray      # (P, L) fitted peak curves
    peaks: tuple                # refined PeakSpecs (fitting order, fit space)
    loss_history: np.ndarray
    stage_ends: tuple
    ssr: float
    bic: float
    n_params: int


def average_rows(prepared: PreparedMap, opts: InitialGuessOptions) -> np.ndarray:
    """Rows of ``prepared`` used for the averaged spectrum (only valid rows are eligible)."""
    pool = np.flatnonzero(prepared.valid)
    if opts.select_mode == "all":
        return pool
    if opts.select_mode == "indices":
        rows = np.asarray(opts.indices, dtype=int)
        bad = rows[~np.isin(rows, pool)]
        if rows.size == 0 or bad.size:
            raise ValueError(f"initial-guess indices {bad.tolist()} are not valid rows to fit")
        return rows
    k = min(int(opts.k_random), pool.size)
    return np.sort(np.random.default_rng(opts.seed).choice(pool, size=k, replace=False))


def bic(ssr: float, n_data: int, n_params: int) -> float:
    """``n ln(SSR / n) + k ln(n)``; ``inf`` when ``SSR <= 0``."""
    if not ssr > 0 or n_data <= 0:
        return float("inf")
    return float(n_data * np.log(ssr / n_data) + n_params * np.log(n_data))


def refined_peaks(peaks: tuple, res: dict, j_row: int = 0) -> tuple:
    """Copies of ``peaks`` (fitting order) with the fitted values of row ``j_row`` of ``res``."""
    out = []
    for j, p in enumerate(peaks):
        extra = {}
        if p.type == "pseudovoigt":
            extra["width_g"] = float(res["width_g"][j_row, j])
        elif p.type == "fano":
            extra["q"] = float(res["q"][j_row, j])
        out.append(p.with_values(pos=float(res["pos"][j_row, j]), width=float(res["width"][j_row, j]),
                                 height=float(res["height"][j_row, j]), extra=extra))
    return tuple(out)


def fit_initial_guess(prepared: PreparedMap, opts: InitialGuessOptions = InitialGuessOptions(), *,
                      device: str = "cpu", rows: Optional[np.ndarray] = None) -> InitialGuess:
    """Fit the averaged spectrum of ``prepared`` and return refined peaks.

    ``rows`` overrides the selection of ``opts`` (used to share one average
    between model variants).
    """
    cfg: FitConfig = prepared.cfg
    rows = average_rows(prepared, opts) if rows is None else np.asarray(rows)
    y = prepared.y[rows].mean(axis=0, keepdims=True)
    peaks = cfg.fit_peaks()
    fitter = BatchFitter(prepared.x, y, prepared.weights, PeakTable.from_peaks(peaks), cfg,
                         resolve_devices([device])[0])
    with torch.no_grad():
        before = fitter.forward()[0].cpu().numpy()[0].astype(np.float64)
    hist, ends = fitter.fit()
    res = fitter.results()
    with torch.no_grad():
        comps = fitter.forward()[1].cpu().numpy()[0].astype(np.float64)
    ssr = float(np.sum((y[0] - res["recon"][0]) ** 2))
    k = cfg.n_params()
    return InitialGuess(rows=rows, x=prepared.x, y=y[0], recon_before=before, recon=res["recon"][0],
                        components=comps, peaks=refined_peaks(peaks, res), loss_history=hist, stage_ends=ends,
                        ssr=ssr, bic=bic(ssr, y.shape[1], k), n_params=k)


def format_peaks_block(peaks, *, conversion: Optional[dict] = None, variable: str = "PEAKS") -> str:
    """Python source of a ``PeakSpec`` list, ready to paste into a script.

    Refined peaks of an energy-space fit are converted back to the input
    unit (nm) when ``conversion`` (the ``PreparedMap.conversion`` record) is
    given. Values are rounded for readability (positions and widths to 4
    significant digits, heights to 3 decimals); regs are kept exactly.
    """
    lines = [f"{variable} = ["]
    for p in peaks:
        if conversion is not None:
            p = peak_to_wavelength(p, conversion["to"], conversion["reference_nm"])
        extra = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in (p.extra or {}).items()}
        args = [f'type="{p.type}"', f"pos={p.pos:.6g}", f"width={p.width:.4g}", f"height={p.height:.3f}",
                f"pos_limit={p.pos_limit:.4g}", f"pos_limit_reg={p.pos_limit_reg!r}",
                f"width_limit={p.width_limit:.4g}", f"width_limit_reg={p.width_limit_reg!r}",
                f"height_limit={p.height_limit:.4g}", f"height_limit_reg={p.height_limit_reg!r}"]
        if extra:
            args.append(f"extra={extra!r}")
        if p.name:
            args.append(f'name="{p.name}"')
        lines.append("    PeakSpec(" + ", ".join(args) + "),")
    lines.append("]")
    return "\n".join(lines)
