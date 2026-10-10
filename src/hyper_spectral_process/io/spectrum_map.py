"""The :class:`StandardSpectrumMap` container and its NPZ file format.

A spectrum map is ``N`` spectra sampled on one shared spectral axis of length
``M``, plus the stage coordinates of every spectrum. The NPZ layout is
described in ``docs/data_format.md``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace as _dc_replace
from pathlib import Path

import numpy as np

REQUIRED_KEYS = ("spectra", "xy", "axis")


@dataclass(frozen=True)
class StandardSpectrumMap:
    """``N`` spectra on a common, strictly increasing spectral axis.

    Attributes
    ----------
    spectra : (N, M) float64
        Intensity of spectrum ``i`` at ``axis[j]``. Must be finite.
    xy : (N, 2) float64
        Stage coordinates ``(x, y)`` of every spectrum, in instrument units
        (often micrometres). No origin or orientation is assumed.
    axis : (M,) float64
        Spectral axis (Raman shift, wavelength, photon energy, ...), strictly
        increasing, at least two samples.
    unit : str
        Axis unit label, e.g. ``"cm^-1"``, ``"nm"`` or ``"eV"``.

    Use :meth:`from_unsorted` to build a map from an axis in arbitrary order.
    """

    spectra: np.ndarray
    xy: np.ndarray
    axis: np.ndarray
    unit: str = ""

    def __post_init__(self) -> None:
        spectra = np.asarray(self.spectra, dtype=np.float64)
        xy = np.asarray(self.xy, dtype=np.float64)
        axis = np.asarray(self.axis, dtype=np.float64)
        if spectra.ndim != 2:
            raise ValueError(f"spectra must be 2-D (N, M); got shape {spectra.shape}")
        n, m = spectra.shape
        if xy.shape != (n, 2):
            raise ValueError(f"xy must have shape ({n}, 2); got {xy.shape}")
        if axis.shape != (m,):
            raise ValueError(f"axis must have shape ({m},); got {axis.shape}")
        if m < 2:
            raise ValueError("axis must contain at least two samples")
        if not np.isfinite(axis).all():
            raise ValueError("axis contains non-finite values")
        if np.any(np.diff(axis) <= 0):
            raise ValueError("axis must be strictly increasing; use StandardSpectrumMap.from_unsorted() "
                             "to sort it and merge duplicate values")
        if not np.isfinite(xy).all():
            raise ValueError("xy contains non-finite values")
        bad_rows = np.flatnonzero(~np.isfinite(spectra).all(axis=1))
        if bad_rows.size:
            raise ValueError(f"spectra contain non-finite values in {bad_rows.size} rows "
                             f"(first: {bad_rows[:5].tolist()})")
        object.__setattr__(self, "spectra", spectra)
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "unit", str(self.unit))

    # ------------------------------------------------------------------ basics
    @property
    def n_spectra(self) -> int:
        """Number of spectra ``N``."""
        return int(self.spectra.shape[0])

    @property
    def n_points(self) -> int:
        """Number of axis samples ``M``."""
        return int(self.spectra.shape[1])

    def replace(self, **changes) -> "StandardSpectrumMap":
        """Copy with some fields replaced, e.g. ``ssm.replace(spectra=new)``."""
        return _dc_replace(self, **changes)

    def select(self, rows) -> "StandardSpectrumMap":
        """Subset of spectra by boolean mask ``(N,)`` or integer indices."""
        rows = np.asarray(rows)
        if rows.dtype == bool and rows.shape != (self.n_spectra,):
            raise ValueError(f"boolean mask must have shape ({self.n_spectra},); got {rows.shape}")
        return self.replace(spectra=self.spectra[rows], xy=self.xy[rows])

    @classmethod
    def from_unsorted(cls, spectra, xy, axis, unit: str = "") -> "StandardSpectrumMap":
        """Build a map from an axis in any order.

        The axis is sorted in increasing order and columns that share an axis
        value are merged by their mean, as the original StandardSpectrumMap
        specification prescribes.
        """
        axis = np.asarray(axis, dtype=np.float64)
        spectra = np.asarray(spectra, dtype=np.float64)
        if spectra.ndim != 2 or axis.shape != (spectra.shape[1],):
            raise ValueError(f"spectra {spectra.shape} and axis {axis.shape} do not match")
        order = np.argsort(axis, kind="mergesort")
        axis, spectra = axis[order], spectra[:, order]
        uniq, inverse, counts = np.unique(axis, return_inverse=True, return_counts=True)
        if uniq.size != axis.size:
            merged = np.zeros((spectra.shape[0], uniq.size))
            np.add.at(merged, (slice(None), inverse), spectra)
            spectra = merged / counts[None, :]
        return cls(spectra, xy, uniq, unit)

    # --------------------------------------------------------------------- NPZ
    def save(self, path: str | Path, *, compressed: bool = True) -> None:
        """Write the map as NPZ with keys ``spectra``, ``xy``, ``axis``, ``unit``."""
        saver = np.savez_compressed if compressed else np.savez
        saver(path, spectra=self.spectra, xy=self.xy, axis=self.axis, unit=np.array(self.unit))

    @classmethod
    def load(cls, path: str | Path) -> "StandardSpectrumMap":
        """Read an NPZ written by :meth:`save` (the ``unit`` key is optional)."""
        with np.load(path, allow_pickle=False) as data:
            missing = [k for k in REQUIRED_KEYS if k not in data.files]
            if missing:
                raise ValueError(f"{path}: missing required key(s) {missing}")
            unit = str(data["unit"]) if "unit" in data.files else ""
            return cls(data["spectra"], data["xy"], data["axis"], unit)


# ---------------------------------------------------------------------- raster
def raster_indices(xy: np.ndarray, decimals: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Locate every spectrum on the raster spanned by its coordinates.

    Coordinates are rounded to ``decimals`` before the unique x and y values
    are taken. Returns ``(xs, ys, ix, iy)``: the sorted unique coordinates and
    the column and row index of every spectrum. Nodes without a spectrum are
    allowed (they become NaN in :func:`to_image`); two spectra on the same
    node raise ``ValueError``.
    """
    xy = np.asarray(xy, dtype=np.float64)
    xs, ix = np.unique(np.round(xy[:, 0], decimals), return_inverse=True)
    ys, iy = np.unique(np.round(xy[:, 1], decimals), return_inverse=True)
    flat = iy * xs.size + ix
    if np.unique(flat).size != flat.size:
        raise ValueError("coordinates contain duplicate raster positions")
    return xs, ys, ix, iy


def to_image(values: np.ndarray, xy: np.ndarray, decimals: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Arrange per-spectrum ``values`` ``(N,)`` on the raster of ``xy``.

    Returns ``(image, xs, ys)`` with ``image[iy, ix]`` (rows = y, columns = x)
    and NaN where the raster has no spectrum.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (np.asarray(xy).shape[0],):
        raise ValueError(f"values must have shape ({np.asarray(xy).shape[0]},); got {values.shape}")
    xs, ys, ix, iy = raster_indices(xy, decimals)
    image = np.full((ys.size, xs.size), np.nan)
    image[iy, ix] = values
    return image, xs, ys
