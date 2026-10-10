"""NT-MDT comma-separated map exports.

Every non-empty row is ``x, y, a_0, I_0, a_1, I_1, ..., a_{M-1}, I_{M-1}``:
the axis values are repeated in every row.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .spectrum_map import StandardSpectrumMap


def read_ntmdt_txt(path: str | Path, unit: str = "nm", *, axis_atol: float = 1e-6) -> StandardSpectrumMap:
    """Read an NT-MDT map export.

    Parameters
    ----------
    path : the text file (no header).
    unit : axis unit stored in the returned map.
    axis_atol : largest absolute difference tolerated between the axis of a
        row and that of the first row.

    Raises ``ValueError`` on a field count that is not ``2 + 2 M`` or differs
    between rows, on non-numeric values and on axis mismatches.
    """
    if axis_atol < 0:
        raise ValueError("axis_atol must be >= 0")
    path = Path(path)
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="strict").splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        raise ValueError(f"{path}: no data rows")
    rows = []
    for k, line in enumerate(lines, start=1):
        try:
            rows.append(np.array([float(t) for t in line.split(",")], dtype=np.float64))
        except ValueError as exc:
            raise ValueError(f"{path} line {k}: non-numeric value ({exc})") from None
    n_fields = rows[0].size
    if n_fields < 6 or (n_fields - 2) % 2:
        raise ValueError(f"{path} line 1: expected 2 + 2*M values with M >= 2, got {n_fields}")
    axis = rows[0][2::2]
    xy = np.empty((len(rows), 2))
    spectra = np.empty((len(rows), axis.size))
    for k, r in enumerate(rows):
        if r.size != n_fields:
            raise ValueError(f"{path} line {k + 1}: expected {n_fields} values, got {r.size}")
        if not np.allclose(r[2::2], axis, rtol=0.0, atol=axis_atol):
            diff = float(np.max(np.abs(r[2::2] - axis)))
            raise ValueError(f"{path} line {k + 1}: axis differs from line 1 by up to {diff}")
        xy[k] = r[:2]
        spectra[k] = r[3::2]
    return StandardSpectrumMap.from_unsorted(spectra, xy, axis, unit)
