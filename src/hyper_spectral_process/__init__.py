"""hyper_spectral_process: preprocessing and GPU-batched multi-peak fitting of hyperspectral maps.

Subpackages: :mod:`.io` (data container, readers, axis conversions),
:mod:`.preprocess` (stage-ordered preprocessing), :mod:`.fit` (peak
fitting), :mod:`.plot` (maps and figures) and :mod:`.synthetic` (test maps).
"""
from .io import StandardSpectrumMap

__version__ = "0.1.0"
__all__ = ["StandardSpectrumMap", "__version__"]
