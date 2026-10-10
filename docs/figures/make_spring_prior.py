"""Figure spring_prior.png: the soft spring prior, its effect on a fit, and the staged optimisation."""
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch

from _common import BLUE, GREY, ORANGE, PURPLE, RED, lorentz, save
from hyper_spectral_process.fit import BatchFitter, FitConfig, PeakSpec, PeakTable, StageSpec, spring


@dataclass(frozen=True)
class AppConfig:
    name: str = "spring_prior.png"
    dpi: int = 150
    seed: int = 7


def one_fit(x, y, reg, stages):
    p = PeakSpec("lorentzian", 382.0, 5.0, 0.8, pos_limit=2.0, pos_limit_reg=reg, width_limit=3.0, width_limit_reg=0.0,
                 height_limit=0.5, height_limit_reg=0.0)
    cfg = FitConfig((p,), stages)
    f = BatchFitter(x, y[None, :], np.ones(x.size), PeakTable.from_peaks((p,)), cfg, torch.device("cpu"))
    hist, ends = f.fit()
    return f.results()["pos"][0, 0], hist, ends


def main(cfg: AppConfig) -> None:
    rng = np.random.default_rng(cfg.seed)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))
    ax = axes[0]
    d = torch.linspace(-2.5, 2.5, 501)
    for reg, c in ((1.0, BLUE), (0.5, ORANGE)):
        s = spring(d, torch.zeros(1), torch.ones(1), torch.tensor([reg])).numpy()
        ax.plot(d.numpy(), s, color=c, label=f"reg = {reg:g}")
    ax.plot(d.numpy(), (d**2).numpy(), color=GREY, ls=":", label="quadratic part only (reg = 1)")
    ax.axvline(1, color=RED, ls="--", lw=0.7)
    ax.axvline(-1, color=RED, ls="--", lw=0.7)
    ax.set_xlabel("d = (value - anchor) / limit")
    ax.set_ylabel("penalty")
    ax.set_title("(a) S = reg (d$^2$ + max(0, |d| - 1)):\nequals reg at one limit, extra wall beyond")
    ax.legend()

    ax = axes[1]
    x = np.linspace(360, 410, 200)
    y = lorentz(x, 385.0, 4.0, 1.0) + rng.normal(0, 0.02, x.size)
    stages = (StageSpec("coarse", 1500, 1e-2),)
    regs = np.array([0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    pos = [one_fit(x, y, r, stages)[0] for r in regs]
    ax.semilogx(np.where(regs > 0, regs, 1e-7), pos, "o-", color=PURPLE)
    ax.axhline(385.0, color=BLUE, ls=":", label="true position")
    ax.axhline(382.0, color=RED, ls=":", label="initial value = anchor")
    ax.set_xlabel("pos_limit_reg (pos_limit = 2)")
    ax.set_ylabel("fitted position")
    ax.set_title("(b) the prior trades data misfit for closeness to the anchor\n(data MSE ~ 4e-4 here; leftmost: reg 0)")
    ax.legend()

    ax = axes[2]
    stages = (StageSpec("coarse", 600, 2e-2), StageSpec("fine", 600, 2e-3), StageSpec("polish", 400, 5e-4))
    _, hist, ends = one_fit(x, y, 1e-4, stages)
    ax.semilogy(hist, color=BLUE)
    start = 0
    for e, st in zip(ends, stages):
        ax.axvline(e - 0.5, color=ORANGE, ls="--", lw=0.8)
        ax.text(0.5 * (start + e), hist.max() * 0.6, f"{st.name}\nlr {st.lr:g}", ha="center", fontsize=8)
        start = e
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss (data + prior)")
    ax.set_title("(c) staged Adam: new optimiser per stage; with recenter_reg\nthe anchors move to the stage-start values")
    fig.tight_layout()
    save(fig, cfg.name, cfg.dpi)


if __name__ == "__main__":
    main(AppConfig())
