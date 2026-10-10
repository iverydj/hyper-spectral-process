"""Maps: put one value per spectrum on the stage raster.

:func:`plot_map` draws values on the raster spanned by the coordinates
(:func:`~hyper_spectral_process.io.to_image`; nodes without a spectrum or
with NaN are grey) or, with ``layout="scatter"``, as square markers at the
coordinates (for scans that are not rectangular).
"""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from ..io import StandardSpectrumMap, to_image

AGGREGATES = ("sum", "mean", "max")


def band_intensity(ssm: StandardSpectrumMap, xmin: Optional[float] = None, xmax: Optional[float] = None,
                   aggregate: str = "sum") -> np.ndarray:
    """``sum``, ``mean`` or ``max`` of every spectrum over ``xmin <= axis <= xmax`` ``(N,)``."""
    if aggregate not in AGGREGATES:
        raise ValueError(f"aggregate must be one of {AGGREGATES}")
    lo = ssm.axis[0] if xmin is None else float(xmin)
    hi = ssm.axis[-1] if xmax is None else float(xmax)
    inside = (ssm.axis >= lo) & (ssm.axis <= hi)
    if not inside.any():
        raise ValueError(f"range [{lo}, {hi}] contains no sample of the axis [{ssm.axis[0]}, {ssm.axis[-1]}]")
    y = ssm.spectra[:, inside]
    return {"sum": y.sum, "mean": y.mean, "max": y.max}[aggregate](axis=1)


def _edges(v: np.ndarray) -> np.ndarray:
    if v.size == 1:
        return np.array([v[0] - 0.5, v[0] + 0.5])
    mid = 0.5 * (v[:-1] + v[1:])
    return np.concatenate([[v[0] - (mid[0] - v[0])], mid, [v[-1] + (v[-1] - mid[-1])]])


def plot_map(ax: plt.Axes, xy: np.ndarray, values: np.ndarray, *, cmap: str = "viridis",
             vmin: Optional[float] = None, vmax: Optional[float] = None, percentiles: Optional[tuple] = None,
             label: str = "", layout: str = "grid", colorbar: bool = True, marker_size: float = 8.0):
    """Draw per-spectrum ``values`` ``(N,)`` at the coordinates ``xy`` ``(N, 2)``.

    Parameters
    ----------
    percentiles : ``(lo, hi)`` sets the colour limits to these percentiles
        of the finite values (overrides ``vmin``/``vmax``).
    layout : ``"grid"`` (raster; duplicate positions raise) or ``"scatter"``.
    Returns the mappable (for a shared colour bar).
    """
    values = np.asarray(values, dtype=np.float64)
    if percentiles is not None:
        finite = values[np.isfinite(values)]
        if finite.size:
            vmin, vmax = np.percentile(finite, percentiles)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("lightgray")
    if layout == "grid":
        img, xs, ys = to_image(values, xy)
        art = ax.pcolormesh(_edges(xs), _edges(ys), np.ma.masked_invalid(img), cmap=cm, vmin=vmin, vmax=vmax,
                            shading="flat")
    elif layout == "scatter":
        m = np.isfinite(values)
        art = ax.scatter(xy[m, 0], xy[m, 1], c=values[m], cmap=cm, vmin=vmin, vmax=vmax, s=marker_size, marker="s")
    else:
        raise ValueError("layout must be 'grid' or 'scatter'")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if colorbar:
        plt.colorbar(art, ax=ax, fraction=0.046, pad=0.04, label=label)
    return art


def plot_band_intensity(ssm: StandardSpectrumMap, xmin: Optional[float] = None, xmax: Optional[float] = None, *,
                        aggregate: str = "sum", cmap: str = "CMRmap", ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Integrated-intensity map of a spectral range (the original heatmap tool)."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 5.6))
    plot_map(ax, ssm.xy, band_intensity(ssm, xmin, xmax, aggregate), cmap=cmap, label=f"{aggregate} intensity")
    lo = ssm.axis[0] if xmin is None else xmin
    hi = ssm.axis[-1] if xmax is None else xmax
    ax.set_title(f"{aggregate} over [{lo:g}, {hi:g}] {ssm.unit}")
    return fig if fig is not None else ax.figure
