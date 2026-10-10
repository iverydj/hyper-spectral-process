"""The fit objective: weighted mean squared residual plus the soft spring prior.

For one spectrum with residual ``r = recon - y`` and per-sample weights ``w``

    loss = sum_l w_l r_l^2 / sum_l w_l
         + mean_j S(pos_j) + mean_j S(width_j) + mean_j S(height_j)
         + height_sign_reg * mean_j max(0, -height_j)
         + mean_j S(q_j) + mean_j S(width_g_j),

where the spring prior of a parameter ``theta`` with anchor ``theta_0``,
natural scale ``limit`` and strength ``reg`` is

    S(theta) = reg * (d^2 + max(0, |d| - 1)),   d = (theta - theta_0) / limit.

The quadratic term always acts; the linear term adds a wall beyond one
``limit``; ``S = reg`` at ``|d| = 1``. Peaks without a ``q`` or ``width_g``
have ``reg = 0`` there. The anchors are the initial values, moved to the
current values at the start of every stage with ``recenter_reg``.

The objective of a batch is the **sum** of the per-spectrum losses, so the
gradient of each spectrum's parameters does not depend on how many spectra
share the batch.
"""
from __future__ import annotations

import torch


def spring(value: torch.Tensor, anchor: torch.Tensor, limit: torch.Tensor, reg: torch.Tensor) -> torch.Tensor:
    """``reg * (d^2 + relu(|d| - 1))`` with ``d = (value - anchor) / limit`` (elementwise)."""
    d = (value - anchor) / limit
    return reg * (d * d + torch.relu(d.abs() - 1.0))


def data_term(recon: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Weighted mean squared residual of every spectrum ``(B,)``; ``w`` has shape ``(1, L)``."""
    return ((recon - y) ** 2 * w).sum(dim=-1) / (w.sum(dim=-1) + 1e-12)


def prior_term(params: dict, anchors: dict, limits: dict, regs: dict, height_sign_reg: float) -> torch.Tensor:
    """Spring prior and negative-height penalty of every spectrum ``(B,)``.

    ``params`` and ``anchors`` map ``pos``, ``width``, ``height``, ``width_g``,
    ``q`` to ``(B, P, 1)`` tensors; ``limits`` and ``regs`` to ``(1, P, 1)``.
    """
    total = height_sign_reg * torch.relu(-params["height"]).mean(dim=(1, 2))
    for name in ("pos", "width", "height", "q", "width_g"):
        total = total + spring(params[name], anchors[name], limits[name], regs[name]).mean(dim=(1, 2))
    return total
