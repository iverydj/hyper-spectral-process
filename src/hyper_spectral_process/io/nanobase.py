"""Nanobase mapping CSV exports.

Metadata lines precede an ``AXISDATA=`` marker; the row after it is the
header (``X, Y, Z, XINDEX, YINDEX, ZINDEX, <axis values...>``) and every
following row is one spectrum.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .spectrum_map import StandardSpectrumMap

_INDEX_COLUMNS = ("Z", "XINDEX", "YINDEX", "ZINDEX")


def read_nanobase_csv(path: str | Path, unit: str = "cm^-1", *, x_column: str = "X", y_column: str = "Y",
                      swap_xy: bool = False, negate_x: bool = False, negate_y: bool = False) -> StandardSpectrumMap:
    """Read a Nanobase mapping CSV.

    Parameters
    ----------
    path : the CSV file.
    unit : axis unit stored in the returned map.
    x_column, y_column : header names of the stage coordinates.
    swap_xy, negate_x, negate_y : orientation corrections applied to the
        coordinates after reading (swap first, then negate). The default
        returns the coordinates exactly as stored. The original converter
        applied ``swap_xy=True, negate_x=True`` for the instrument it was
        written for.

    Columns whose header parses as a number form the spectral axis. Every
    cell must be numeric; empty cells raise ``ValueError``.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = [[c.strip() for c in r] for r in csv.reader(fh)]
    marker = next((i for i, r in enumerate(rows) if r and r[0].upper().startswith("AXISDATA")), None)
    if marker is None or marker + 1 >= len(rows):
        raise ValueError(f"{path}: 'AXISDATA=' marker followed by a header row not found")
    header = rows[marker + 1]
    for col in (x_column, y_column):
        if col not in header:
            raise ValueError(f"{path}: header has no column {col!r}")
    ix, iy = header.index(x_column), header.index(y_column)
    skip = {ix, iy} | {header.index(c) for c in _INDEX_COLUMNS if c in header}
    spec_cols, axis = [], []
    for j, h in enumerate(header):
        if j in skip:
            continue
        try:
            axis.append(float(h))
        except ValueError:
            continue
        spec_cols.append(j)
    if len(spec_cols) < 2:
        raise ValueError(f"{path}: header has fewer than two numeric axis columns")
    data = [r for r in rows[marker + 2:] if any(r)]
    if not data:
        raise ValueError(f"{path}: no data rows after the header")
    spectra = np.empty((len(data), len(spec_cols)))
    xy = np.empty((len(data), 2))
    for k, r in enumerate(data):
        where = f"{path} data row {k + 1}"
        if len(r) <= max(spec_cols[-1], ix, iy):
            raise ValueError(f"{where}: {len(r)} cells, expected at least {spec_cols[-1] + 1}")
        try:
            xy[k] = (float(r[ix]), float(r[iy]))
            spectra[k] = [float(r[j]) for j in spec_cols]
        except ValueError as exc:
            raise ValueError(f"{where}: non-numeric cell ({exc})") from None
    if swap_xy:
        xy = xy[:, ::-1].copy()
    if negate_x:
        xy[:, 0] = -xy[:, 0]
    if negate_y:
        xy[:, 1] = -xy[:, 1]
    return StandardSpectrumMap.from_unsorted(spectra, xy, axis, unit)
