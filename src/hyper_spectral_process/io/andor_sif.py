"""Andor Solis ``.sif`` kinetic series recorded as a raster map.

SIF files store frames but no stage coordinates, so the raster is described
by the caller. Requires the optional package ``sif_parser``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .spectrum_map import StandardSpectrumMap


def symmetric_raster(x_half_size: float, y_half_size: float, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates ``linspace(-x_half_size, x_half_size, nx)`` and the same for y.

    This is the raster convention of the original SIF converter (a scan
    centred on the stage origin).
    """
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be positive")
    return (np.linspace(-float(x_half_size), float(x_half_size), int(nx)),
            np.linspace(-float(y_half_size), float(y_half_size), int(ny)))


def frames_to_xy(n_frames: int, x_coords: Sequence[float], y_coords: Sequence[float],
                 frame_order: str = "x_fast") -> np.ndarray:
    """Stage coordinates ``(n_frames, 2)`` of a raster acquisition.

    ``frame_order="x_fast"``: frame ``k`` is at ``(x[k % nx], y[k // nx])``
    (x changes fastest). ``"y_fast"``: ``(x[k // ny], y[k % ny])``. Pass the
    coordinate arrays reversed to describe a scan that runs towards
    decreasing x or y.
    """
    xs, ys = np.asarray(x_coords, dtype=np.float64), np.asarray(y_coords, dtype=np.float64)
    nx, ny = xs.size, ys.size
    if n_frames != nx * ny:
        raise ValueError(f"{n_frames} frames but the raster has {nx} x {ny} = {nx * ny} nodes")
    k = np.arange(n_frames)
    if frame_order == "x_fast":
        ix, iy = k % nx, k // nx
    elif frame_order == "y_fast":
        ix, iy = k // ny, k % ny
    else:
        raise ValueError("frame_order must be 'x_fast' or 'y_fast'")
    return np.column_stack([xs[ix], ys[iy]])


def read_andor_sif(path: str | Path, x_coords: Sequence[float], y_coords: Sequence[float], *,
                   frame_order: str = "x_fast", unit: str = "nm") -> StandardSpectrumMap:
    """Read a SIF kinetic series of full-vertical-binning spectra.

    Parameters
    ----------
    path : the ``.sif`` file.
    x_coords, y_coords : raster coordinates (see :func:`symmetric_raster`).
    frame_order : acquisition order of the frames, see :func:`frames_to_xy`.
    unit : unit of the wavelength calibration stored in the file.

    Every frame must be a single spectrum (shape ``(1, M)``); the axis is the
    calibration stored in the file.
    """
    try:
        import sif_parser
        from sif_parser.utils import extract_calibration
    except ImportError as exc:  # optional dependency
        raise ImportError("read_andor_sif requires the optional package 'sif_parser' "
                          "(pip install sif_parser)") from exc
    data, info = sif_parser.np_open(str(path))
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 3 and data.shape[1] == 1:
        data = data[:, 0, :]
    if data.ndim != 2:
        raise ValueError(f"{path}: expected frames of shape (1, M) (full vertical binning); got {data.shape}")
    calibration = extract_calibration(info)
    if calibration is None:
        raise ValueError(f"{path}: the file contains no wavelength calibration")
    axis = np.asarray(calibration, dtype=np.float64)
    if axis.shape != (data.shape[1],):
        raise ValueError(f"{path}: calibration length {axis.shape} differs from spectrum length {data.shape[1]}")
    xy = frames_to_xy(data.shape[0], x_coords, y_coords, frame_order)
    return StandardSpectrumMap.from_unsorted(data, xy, axis, unit)
