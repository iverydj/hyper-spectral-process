"""The multi-peak model: per-peak constants (:class:`PeakTable`) and its evaluation in torch.

The model of one spectrum is

    recon(x) = base + sum_j height_j * profile_j(x; pos_j, width_j, ...),

with a constant ``base`` and the unit-height profiles of :mod:`.lineshapes`.
All parameters have shape ``(B, P, 1)`` (``base``: ``(B, 1)``) so that ``B``
spectra with ``P`` peaks each are evaluated as one ``(B, P, L)`` tensor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

from .lineshapes import fano, pseudo_voigt, pseudo_voigt_tch, tch_fwhm_eta
from .specs import PeakSpec

#: Profile family of each peak type: ``mix`` = eta-weighted Gaussian/Lorentzian with one FWHM.
FAMILY = {"gaussian": "mix", "lorentzian": "mix", "pseudovoigt_fixed": "mix", "pseudovoigt": "tch", "fano": "fano"}


@dataclass(frozen=True)
class PeakTable:
    """Per-peak constants of a fit as ``(P,)`` float arrays, in fitting order."""

    types: tuple
    names: tuple
    pos: np.ndarray
    width: np.ndarray
    height: np.ndarray
    width_g: np.ndarray      # TCH pseudo-Voigt Gaussian FWHM (0 for other types)
    q: np.ndarray            # Fano asymmetry (0 for other types)
    eta: np.ndarray          # fixed mixing of the 'mix' family (G 0, L 1, fixed pV eta)
    q_lo: np.ndarray
    q_hi: np.ndarray
    limits: dict             # parameter -> (P,) natural scale of the spring
    regs: dict               # parameter -> (P,) spring strength

    @classmethod
    def from_peaks(cls, peaks: Sequence[PeakSpec]) -> "PeakTable":
        peaks = tuple(peaks)
        tch = [p.type == "pseudovoigt" for p in peaks]
        fan = [p.type == "fano" for p in peaks]

        def arr(values):
            return np.asarray(list(values), dtype=np.float64)

        eta = arr({"gaussian": 0.0, "lorentzian": 1.0}.get(p.type, float(p.get("eta", 0.0))) for p in peaks)
        return cls(
            types=tuple(p.type for p in peaks), names=tuple(p.name for p in peaks),
            pos=arr(p.pos for p in peaks), width=arr(p.width for p in peaks), height=arr(p.height for p in peaks),
            width_g=arr(p.get("width_g") if t else 0.0 for p, t in zip(peaks, tch)),
            q=arr(p.get("q") if f else 0.0 for p, f in zip(peaks, fan)),
            eta=eta,
            q_lo=arr(p.get("q_range")[0] if f else 0.0 for p, f in zip(peaks, fan)),
            q_hi=arr(p.get("q_range")[1] if f else 0.0 for p, f in zip(peaks, fan)),
            limits={"pos": arr(p.pos_limit for p in peaks), "width": arr(p.width_limit for p in peaks),
                    "height": arr(p.height_limit for p in peaks),
                    "width_g": arr(p.get("width_g_limit") if t else 1.0 for p, t in zip(peaks, tch)),
                    "q": arr(p.get("q_limit") if f else 1.0 for p, f in zip(peaks, fan))},
            regs={"pos": arr(p.pos_limit_reg for p in peaks), "width": arr(p.width_limit_reg for p in peaks),
                  "height": arr(p.height_limit_reg for p in peaks),
                  "width_g": arr(p.get("width_g_limit_reg") if t else 0.0 for p, t in zip(peaks, tch)),
                  "q": arr(p.get("q_limit_reg") if f else 0.0 for p, f in zip(peaks, fan))},
        )

    @property
    def n_peaks(self) -> int:
        return len(self.types)


class PeakModel:
    """Evaluate peaks of the given types on one axis, on one device.

    ``eta`` ``(P,)`` is the fixed mixing of the 'mix'-family peaks (ignored
    for the others).
    """

    def __init__(self, types: Sequence[str], eta: np.ndarray, x: np.ndarray, device: torch.device):
        self.device = device
        self.x = torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)[None, None, :]  # (1, 1, L)
        self.eta = torch.as_tensor(np.asarray(eta, dtype=np.float32), device=device)[None, :, None]
        groups = []
        for fam in ("mix", "tch", "fano"):
            idx = [j for j, t in enumerate(types) if FAMILY[t] == fam]
            if idx:
                groups.append((fam, idx))
        self.groups = [(fam, torch.as_tensor(idx, device=device)) for fam, idx in groups]
        perm = [j for _, idx in groups for j in idx]
        self.inverse = torch.as_tensor(np.argsort(perm), device=device)
        self.single = len(groups) == 1

    def profiles(self, pos, width, width_g, q, eta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Unit-height profiles ``(B, P, L)``; ``eta`` (``(B, P, 1)``) overrides the fixed mixing."""
        eta = self.eta if eta is None else eta
        parts = []
        for fam, idx in self.groups:
            def sel(t):
                return t if self.single else t.index_select(1, idx)
            if fam == "mix":
                prof = pseudo_voigt(self.x, sel(pos), sel(width), sel(eta))
            elif fam == "tch":
                prof = pseudo_voigt_tch(self.x, sel(pos), sel(width), sel(width_g))
            else:
                prof = fano(self.x, sel(pos), sel(width), sel(q))
            parts.append(prof)
        if self.single:
            return parts[0]
        return torch.cat(parts, dim=1).index_select(1, self.inverse)

    def peaks(self, pos, width, height, width_g, q, active: Optional[torch.Tensor] = None,
              eta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``height * profile`` of every peak ``(B, P, L)``; peaks outside ``active`` (mask ``(P,)``) are zero."""
        h = height if active is None else height * active[None, :, None]
        return h * self.profiles(pos, width, width_g, q, eta)


def effective_eta(table: PeakTable, width: np.ndarray, width_g: np.ndarray) -> np.ndarray:
    """``eta`` of every peak ``(N, P)``: fixed for the 'mix' family, from TCH for ``pseudovoigt``, 0 for Fano."""
    eta = np.broadcast_to(table.eta, width.shape).astype(np.float64).copy()
    for j, t in enumerate(table.types):
        if t == "pseudovoigt":
            eta[:, j] = tch_fwhm_eta(width_g[:, j], width[:, j])[1]
        elif t == "fano":
            eta[:, j] = 0.0
    return eta
