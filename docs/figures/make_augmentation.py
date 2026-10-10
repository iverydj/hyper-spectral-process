"""Figure augmentation.png: train/valid/test split and mixup with white or low-pass noise."""
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from _common import BLUE, GREEN, ORANGE, PURPLE, RED, gauss, save
from hyper_spectral_process.preprocess import lowpass_noise, mixup, split_assignments


@dataclass(frozen=True)
class AppConfig:
    name: str = "augmentation.png"
    dpi: int = 150
    seed: int = 5


def main(cfg: AppConfig) -> None:
    rng = np.random.default_rng(cfg.seed)
    x = np.linspace(0, 100, 300)
    pool = np.vstack([gauss(x, 30 + 6 * k, 8, 1.0) + gauss(x, 70, 10, 0.3 + 0.2 * k) for k in range(4)])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))

    ax = axes[0]
    labels = split_assignments(60, 0.6, 0.2, 0.2, seed=cfg.seed)
    colors = {"train": BLUE, "valid": ORANGE, "test": GREEN}
    for i, lab in enumerate(labels):
        ax.add_patch(plt.Rectangle((i % 12, i // 12), 0.9, 0.9, color=colors[lab]))
    for lab, c in colors.items():
        ax.plot([], [], "s", color=c, label=f"{lab} ({int(np.sum(labels == lab))})")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3)
    ax.set_title("(a) split: 60 spectra, ratios 0.6 / 0.2 / 0.2, seeded shuffle")

    ax = axes[1]
    # draw until an example with visibly mixed weights (largest weight below 0.6) comes up
    e = next(ex[0] for k in range(1000)
             for ex in [mixup(pool, 1, np.random.default_rng(cfg.seed + k), min_sources=3, max_sources=3,
                              noise_std=0.02)[2]] if ex[0]["weights"].max() < 0.6)
    for s, w in zip(e["sources"], e["weights"]):
        ax.plot(x, pool[s], lw=1.0, alpha=0.45, label=f"source {s}, weight {w:.2f}")
    ax.plot(x, e["mixed"], "k--", lw=1.3, label="sum of weights x sources")
    ax.plot(x, e["final"], color=RED, lw=1.0, label="+ noise (std 0.02)")
    ax.set_title("(b) mixup: Dirichlet weights, concentration\ndrawn from alpha_range, sources from one split")
    ax.legend(fontsize=7)

    ax = axes[2]
    white = rng.normal(size=x.size)
    low = lowpass_noise(x.size, np.random.default_rng(cfg.seed), sigma_points=4.0)
    ax.plot(x, white, color="0.7", lw=0.8, label="white noise")
    ax.plot(x, low + 4, color=PURPLE, lw=1.0, label="low-pass: Gaussian kernel, sigma 4 samples (offset)")
    ax.set_title("(c) noise modes, both rescaled to noise_std")
    ax.legend(fontsize=7, loc="lower right")
    for a in axes[1:]:
        a.set_xlabel("axis")
    fig.tight_layout()
    save(fig, cfg.name, cfg.dpi)


if __name__ == "__main__":
    main(AppConfig())
