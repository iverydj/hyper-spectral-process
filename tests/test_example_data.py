"""The measured example maps load with the documented layout."""
from pathlib import Path

import numpy as np
import pytest

from hyper_spectral_process.io import StandardSpectrumMap, raman_shift_to_wavelength

DATA = Path(__file__).resolve().parents[1] / "examples" / "data" / "mos2_wrinkled"


@pytest.mark.parametrize("name,lo,hi", [("raman_map.npz", 139.9, 651.6), ("pl_map.npz", 285.5, 5628.5)])
def test_example_maps(name, lo, hi):
    m = StandardSpectrumMap.load(DATA / name)
    assert m.spectra.shape == (7128, 1024) and m.unit == "cm^-1"
    assert lo < m.axis[0] < m.axis[-1] < hi
    xs, ys = np.unique(m.xy[:, 0].round(6)), np.unique(m.xy[:, 1].round(6))
    assert xs.size * ys.size == 7128


def test_pl_pixels_are_uniform_in_wavelength():
    lam = raman_shift_to_wavelength(StandardSpectrumMap.load(DATA / "pl_map.npz"), 532.0, jacobian=False).axis
    step = np.diff(lam)
    assert np.ptp(step) / step.mean() < 0.01
