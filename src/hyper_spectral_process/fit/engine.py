"""Staged Adam optimisation of a batch of spectra (:class:`BatchFitter`) and its distribution over devices.

Every spectrum has its own parameters; a batch of ``B`` spectra is one
``(B, P, L)`` tensor problem. The loss history is accumulated on the device
and copied to the host once per batch, so the optimisation never waits for
the host and several devices run concurrently (one worker thread each).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence

import numpy as np
import torch

from .model import PeakModel, PeakTable, effective_eta
from .objective import data_term, prior_term
from .specs import FitConfig, StageSpec

PARAM_NAMES = ("pos", "width", "height", "width_g", "q")
WIDTH_MIN = 1e-4      # widths are clamped to at least this after every step
HEIGHT_MIN = -1e3     # heights are clamped to at least this after every step


class BatchFitter:
    """Fit ``B`` spectra on one axis with the staged plan of a :class:`FitConfig`.

    Parameters
    ----------
    x : (L,) fit axis.
    y : (B, L) spectra.
    weights : (L,) per-sample loss weights.
    table : peak constants in fitting order (initial values, limits, regs).
    cfg : the fit configuration (stages, baseline, height-sign penalty).
    device : torch device.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray, table: PeakTable, cfg: FitConfig,
                 device: torch.device):
        self.cfg, self.table, self.device = cfg, table, device
        self.model = PeakModel(table.types, table.eta, x, device)
        f32 = dict(dtype=torch.float32, device=device)
        self.y = torch.as_tensor(np.asarray(y, dtype=np.float32), device=device)
        self.w = torch.as_tensor(np.asarray(weights, dtype=np.float32), device=device)[None, :]
        b = self.y.shape[0]

        def per_peak(a):
            return torch.as_tensor(a, **f32)[None, :, None]

        self.params = {n: torch.nn.Parameter(per_peak(getattr(table, n)).repeat(b, 1, 1)) for n in PARAM_NAMES}
        self.base = torch.nn.Parameter(torch.full((b, 1), float(cfg.base_init), **f32))
        self.anchors = {n: p.detach().clone() for n, p in self.params.items()}
        self.limits = {n: per_peak(v) for n, v in table.limits.items()}
        self.regs = {n: per_peak(v) for n, v in table.regs.items()}
        self.q_lo, self.q_hi = per_peak(table.q_lo), per_peak(table.q_hi)

    # ------------------------------------------------------------- model
    def _active(self, stage: Optional[StageSpec]) -> Optional[torch.Tensor]:
        if stage is None or stage.active == "all":
            return None
        mask = torch.zeros(self.table.n_peaks, dtype=torch.float32, device=self.device)
        mask[list(stage.active)] = 1.0
        return mask

    def forward(self, active: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """``(recon (B, L), peaks (B, P, L))``."""
        p = self.params
        peaks = self.model.peaks(p["pos"], p["width"], p["height"], p["width_g"], p["q"], active)
        return self.base + peaks.sum(dim=1), peaks

    def per_spectrum_loss(self, recon: torch.Tensor) -> torch.Tensor:
        """Objective of every spectrum ``(B,)`` (see :mod:`.objective`)."""
        return data_term(recon, self.y, self.w) + prior_term(self.params, self.anchors, self.limits, self.regs,
                                                            self.cfg.height_sign_reg)

    # -------------------------------------------------------- optimisation
    def recenter(self) -> None:
        """Move the spring anchors to the current parameter values."""
        with torch.no_grad():
            for n, p in self.params.items():
                self.anchors[n].copy_(p)

    def _project(self) -> None:
        with torch.no_grad():
            self.params["width"].clamp_(min=WIDTH_MIN)
            self.params["width_g"].clamp_(min=WIDTH_MIN)
            self.params["height"].clamp_(min=HEIGHT_MIN)
            q = self.params["q"]
            q.copy_(torch.maximum(torch.minimum(q, self.q_hi), self.q_lo))

    def fit(self) -> tuple[np.ndarray, tuple[int, ...]]:
        """Run every stage; returns the mean loss per iteration and the cumulative stage ends."""
        stages = self.cfg.stages
        history = torch.empty(sum(int(s.iters) for s in stages), dtype=torch.float32, device=self.device)
        ends, k = [], 0
        self._project()
        for st in stages:
            if st.recenter_reg:
                self.recenter()
            active = self._active(st)
            opt_params = list(self.params.values()) + ([self.base] if self.cfg.fit_baseline else [])
            opt = torch.optim.Adam(opt_params, lr=float(st.lr))
            for _ in range(int(st.iters)):
                opt.zero_grad(set_to_none=True)
                recon, _ = self.forward(active)
                per = self.per_spectrum_loss(recon)
                per.sum().backward()
                opt.step()
                self._project()
                history[k] = per.detach().mean()
                k += 1
            if st.iters:
                ends.append(k)
        return history.cpu().numpy().astype(np.float64), tuple(ends)

    def results(self) -> dict:
        """Fitted parameters ``(B, P)``, ``eta``, ``base (B,)``, ``recon (B, L)`` and the final loss ``(B,)``."""
        with torch.no_grad():
            recon, _ = self.forward()
            loss = self.per_spectrum_loss(recon)
        out = {n: p.detach().cpu().numpy()[:, :, 0].astype(np.float64) for n, p in self.params.items()}
        out["eta"] = effective_eta(self.table, out["width"], out["width_g"])
        out["base"] = self.base.detach().cpu().numpy()[:, 0].astype(np.float64)
        out["recon"] = recon.cpu().numpy().astype(np.float64)
        out["loss_final"] = loss.cpu().numpy().astype(np.float64)
        return out


# ------------------------------------------------------------------ devices
def resolve_devices(devices: Sequence[str]) -> list[torch.device]:
    """Check that every requested device exists (``"cpu"``, ``"cuda"``, ``"cuda:k"``); repeats are allowed."""
    devices = [devices] if isinstance(devices, str) else list(devices)
    if not devices:
        raise ValueError("at least one device is required")
    out = []
    for d in devices:
        dev = torch.device(d)
        if dev.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(f"device {d!r} requested but CUDA is not available")
            index = torch.cuda.current_device() if dev.index is None else dev.index
            if index >= torch.cuda.device_count():
                raise RuntimeError(f"device {d!r} requested but only {torch.cuda.device_count()} CUDA device(s) "
                                   "are visible")
            dev = torch.device("cuda", index)
        elif dev.type != "cpu":
            raise ValueError(f"unsupported device {d!r}")
        out.append(dev)
    return out


def fit_batches(x: np.ndarray, y: np.ndarray, weights: np.ndarray, table: PeakTable, cfg: FitConfig, *,
                devices: Sequence[str] = ("cpu",), batch_size: int = 256) -> tuple[dict, np.ndarray, tuple]:
    """Fit every row of ``y`` in batches of ``batch_size``, spread over ``devices``.

    Batch ``b`` runs on ``devices[b % len(devices)]``; every device has one
    worker thread. Returns ``(results, histories, stage_ends)``: the
    concatenated :meth:`BatchFitter.results`, the mean-loss history of every
    batch ``(n_batches, n_iter)`` and the stage ends.
    """
    if int(batch_size) != batch_size or batch_size < 1:
        raise ValueError(f"batch_size must be a positive integer; got {batch_size}")
    devs = resolve_devices(devices)
    n = y.shape[0]
    if n == 0:
        raise ValueError("no spectra to fit")
    starts = list(range(0, n, int(batch_size)))
    results: list = [None] * len(starts)

    def work(k_dev: int) -> None:
        for b in range(k_dev, len(starts), len(devs)):
            rows = slice(starts[b], starts[b] + int(batch_size))
            fitter = BatchFitter(x, y[rows], weights, table, cfg, devs[k_dev])
            hist, ends = fitter.fit()
            results[b] = (fitter.results(), hist, ends)

    if len(devs) == 1:
        work(0)
    else:
        with ThreadPoolExecutor(max_workers=len(devs)) as pool:
            for fut in [pool.submit(work, k) for k in range(len(devs))]:
                fut.result()
    merged = {key: np.concatenate([r[0][key] for r in results]) for key in results[0][0]}
    histories = np.stack([r[1] for r in results])
    return merged, histories, results[0][2]
