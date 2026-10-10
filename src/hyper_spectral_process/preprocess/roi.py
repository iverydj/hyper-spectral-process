"""ROI stage: keep a range of the spectral axis without resampling."""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..io import StandardSpectrumMap


def crop_roi(ssm: StandardSpectrumMap, xmin: Optional[float] = None, xmax: Optional[float] = None, *,
             round_decimals: Optional[int] = None) -> StandardSpectrumMap:
    """Keep the samples with ``xmin <= axis <= xmax``.

    Parameters
    ----------
    xmin, xmax : axis limits in axis units; ``None`` means the end of the axis.
        A range that extends beyond the data is clipped to the data.
    round_decimals : optionally round the intensities to this many decimals
        (``2`` gives 0.01 precision).

    Raises ``ValueError`` when ``xmin > xmax`` or when fewer than two samples
    remain.
    """
    lo = ssm.axis[0] if xmin is None else float(xmin)
    hi = ssm.axis[-1] if xmax is None else float(xmax)
    if lo > hi:
        raise ValueError(f"ROI needs xmin <= xmax; got ({xmin}, {xmax})")
    keep = (ssm.axis >= lo) & (ssm.axis <= hi)
    if keep.sum() < 2:
        raise ValueError(f"ROI [{lo}, {hi}] keeps fewer than two samples of the axis "
                         f"[{ssm.axis[0]}, {ssm.axis[-1]}]")
    spectra = ssm.spectra[:, keep]
    if round_decimals is not None:
        spectra = np.round(spectra, int(round_decimals))
    return ssm.replace(spectra=spectra, axis=ssm.axis[keep])
