"""HORIBA LabSpec map exports (wide text table).

Layout: the first row holds the ``M`` axis values (in LabSpec exports it
starts with two empty cells); every following row is ``x  y  I_1 ... I_M``.
Cells may be separated by tabs, spaces, commas or semicolons. Empty lines and
lines starting with ``#`` or ``//`` are skipped.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .spectrum_map import StandardSpectrumMap, raster_indices


def _tokens(line: str) -> list[str]:
    return line.replace("﻿", "").replace(",", " ").replace(";", " ").split()


def _floats(tokens: list[str], where: str) -> np.ndarray:
    try:
        return np.array([float(t) for t in tokens], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"{where}: non-numeric cell ({exc})") from None


def read_horiba_txt(path: str | Path, unit: str = "cm^-1") -> StandardSpectrumMap:
    """Read a HORIBA LabSpec map export.

    Parameters
    ----------
    path : the text file.
    unit : axis unit stored in the returned map (the export does not record it).

    Returns
    -------
    StandardSpectrumMap with the axis sorted increasing (duplicate axis values
    merged by their mean). Raises ``ValueError`` when a row does not contain
    exactly ``M + 2`` numbers.
    """
    path = Path(path)
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="strict").splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith(("#", "//"))]
    if len(lines) < 2:
        raise ValueError(f"{path}: expected an axis header row and at least one data row")
    axis = _floats(_tokens(lines[0]), f"{path} header row")
    m = axis.size
    if m < 2:
        raise ValueError(f"{path}: header row has fewer than two axis values")
    rows = np.empty((len(lines) - 1, m + 2), dtype=np.float64)
    for k, line in enumerate(lines[1:]):
        values = _floats(_tokens(line), f"{path} data row {k + 1}")
        if values.size != m + 2:
            raise ValueError(f"{path} data row {k + 1}: expected {m + 2} values (x, y, {m} intensities), "
                             f"got {values.size}")
        rows[k] = values
    return StandardSpectrumMap.from_unsorted(rows[:, 2:], rows[:, :2], axis, unit)


def _fmt(value: float, max_decimals: int) -> str:
    v = float(value)
    if math.isfinite(v) and v.is_integer():
        return str(int(v))
    s = f"{v:.{max_decimals}f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


def write_horiba_txt(ssm: StandardSpectrumMap, path: str | Path, *, frame_order: str = "x_fast",
                     max_decimals: int = 7) -> None:
    """Write ``ssm`` in the LabSpec layout read by :func:`read_horiba_txt`.

    Rows are written in raster order: ``frame_order="x_fast"`` (y outer, x
    inner) or ``"y_fast"``. The coordinates must form a complete raster.
    Values are written with at most ``max_decimals`` decimals (trailing zeros
    removed), tab-separated, with CRLF line endings.
    """
    xs, ys, ix, iy = raster_indices(ssm.xy)
    if xs.size * ys.size != ssm.n_spectra:
        raise ValueError(f"coordinates form an incomplete raster ({xs.size} x {ys.size} nodes "
                         f"for {ssm.n_spectra} spectra)")
    if frame_order == "x_fast":
        order = np.lexsort((ix, iy))
    elif frame_order == "y_fast":
        order = np.lexsort((iy, ix))
    else:
        raise ValueError("frame_order must be 'x_fast' or 'y_fast'")
    with open(path, "w", encoding="ascii", newline="") as fh:
        fh.write("\t\t" + "\t".join(_fmt(v, max_decimals) for v in ssm.axis) + "\r\n")
        for i in order:
            cells = [_fmt(ssm.xy[i, 0], max_decimals), _fmt(ssm.xy[i, 1], max_decimals)]
            cells += [_fmt(v, max_decimals) for v in ssm.spectra[i]]
            fh.write("\t".join(cells) + "\r\n")
