"""Figure batching.png: one (B, P, L) problem per batch, batches spread over devices; batch-size invariance."""
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch

from _common import BLUE, GREEN, ORANGE, RED, arrow, box, lorentz, save
from hyper_spectral_process.fit import BatchFitter, FitConfig, PeakSpec, PeakTable, StageSpec


@dataclass(frozen=True)
class AppConfig:
    name: str = "batching.png"
    dpi: int = 150
    seed: int = 8


def main(cfg: AppConfig) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    a.set_axis_off()
    a.set_title("every pixel has its own parameters; a batch is one tensor problem", fontsize=10)
    box(a, 0.0, 0.62, 0.22, 0.28, "N selected pixels\n(rows of the map)", GREEN)
    for k, c in enumerate((BLUE, ORANGE, BLUE, ORANGE)):
        box(a, 0.30, 0.78 - k * 0.17, 0.2, 0.12, f"batch {k}: rows\n{k * 256}-{k * 256 + 255}", c, 8)
        arrow(a, 0.225, 0.76, 0.30, 0.84 - k * 0.17)
    box(a, 0.62, 0.70, 0.36, 0.2, "worker thread 1 -> device 1\nbatches 0, 2, 4, ...", BLUE, 8.5)
    box(a, 0.62, 0.42, 0.36, 0.2, "worker thread 2 -> device 2\nbatches 1, 3, 5, ...", ORANGE, 8.5)
    a.text(0.0, 0.25, "in each batch: pos, width, height, width_g, q are (B, P, 1) tensors, base is (B, 1);\n"
           "the model evaluates all B x P peaks on the L window samples as one (B, P, L) tensor;\n"
           "the objective is the SUM of the per-pixel losses, so a pixel's gradient does not depend on B;\n"
           "the loss history stays on the device until the batch ends (no per-iteration host sync).",
           transform=a.transAxes, fontsize=8.5, va="top")

    rng = np.random.default_rng(cfg.seed)
    x = np.linspace(360, 410, 200)
    y = lorentz(x, 385.0, 4.0, 1.0)[None, :] + rng.normal(0, 0.02, (256, x.size))
    p = PeakSpec("lorentzian", 383.0, 5.0, 0.8, 2.0, 0.0, 3.0, 0.0, 0.5, 0.0)
    conf = FitConfig((p,), (StageSpec("s", 1500, 1e-2),))
    table = PeakTable.from_peaks((p,))

    def fit(rows):
        f = BatchFitter(x, y[rows], np.ones(x.size), table, conf, torch.device("cpu"))
        f.fit()
        return f.results()["pos"][:, 0]

    together = fit(slice(0, 256))[:16]                         # the first 16 pixels inside a batch of 256
    alone = np.array([fit(slice(i, i + 1))[0] for i in range(16)])   # the same pixels, one per batch
    b.plot(alone, together, "o", color=BLUE)
    lim = (min(alone.min(), together.min()) - 0.01, max(alone.max(), together.max()) + 0.01)
    b.plot(lim, lim, color=RED, lw=0.8, ls=":")
    b.set_xlabel("position, pixel fitted alone (B = 1)")
    b.set_ylabel("position, same pixel in a batch of B = 256")
    b.set_title(f"batch-size invariance: largest difference {np.max(np.abs(alone - together)):.1e}", fontsize=10)
    fig.tight_layout()
    save(fig, cfg.name, cfg.dpi)


if __name__ == "__main__":
    main(AppConfig())
