"""Figure pipeline.png: the stage-ordered preprocessing pipeline and the fit that follows it."""
from dataclasses import dataclass

import matplotlib.pyplot as plt

from _common import BLUE, GREEN, ORANGE, PURPLE, RED, arrow, box, save


@dataclass(frozen=True)
class AppConfig:
    name: str = "pipeline.png"
    dpi: int = 150


def main(cfg: AppConfig) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.0))
    ax.set_axis_off()
    ax.text(0.0, 0.97, "1  read", fontsize=11, weight="bold", transform=ax.transAxes)
    box(ax, 0.0, 0.80, 0.13, 0.12, "instrument export\n(LabSpec, Nanobase,\nNT-MDT, Andor SIF)", GREEN, 8)
    arrow(ax, 0.135, 0.86, 0.165, 0.86)
    box(ax, 0.17, 0.80, 0.12, 0.12, "StandardSpectrumMap\n.npz", GREEN, 8)
    ax.text(0.33, 0.97, "2  preprocess: only the stages listed in stage_order run, in that order", fontsize=11,
            weight="bold", transform=ax.transAxes)
    stages = ["split", "roi", "despike", "augmentation", "normalize", "smooth", "baseline", "smooth2nd", "interp",
              "deriv"]
    for k, s in enumerate(stages):
        x = 0.33 + k * 0.067
        box(ax, x, 0.80, 0.054, 0.12, s.replace("augmentation", "augmen-\ntation").replace("smooth2nd", "smooth\n2nd"),
            BLUE, 7.5)
        if k:
            arrow(ax, x - 0.012, 0.86, x - 0.002, 0.86)
    arrow(ax, 0.295, 0.86, 0.33, 0.86)
    rules = ["- any subset in any order, each stage at most once (the order above is one example)",
             "- parameters of a stage never enable it; despike.mode / baseline.mode must be set exactly when listed",
             "- 'split' first and only with split.enable; augmentation mixes spectra inside one split only",
             "- 'deriv' saves derivative variants of the data at its position; later stages act on the d0 spectra",
             "- the whole configuration is validated before any file is read or written"]
    for k, r in enumerate(rules):
        ax.text(0.33, 0.70 - k * 0.055, r, fontsize=8.5, transform=ax.transAxes)
    ax.text(0.33, 0.40, "outputs: npz_proc_d0 / d1 / d2, metadata table, QA image of every stage, per-spectrum "
            "traces, class-wise t-SNE", fontsize=8.5, color="0.3", transform=ax.transAxes)
    ax.text(0.0, 0.30, "3  fit", fontsize=11, weight="bold", transform=ax.transAxes)
    steps = [("select pixels\nlow-signal skip", ORANGE), ("axis space\n(nm -> eV, Jacobian)", ORANGE),
             ("normalise 0-1\nfit window", ORANGE), ("initial guess on\naveraged spectrum", PURPLE),
             ("staged Adam with\nspring prior,\nB pixels per batch", PURPLE), ("FitSpectrumMap\n+ summary plots", GREEN),
             ("variants: BIC per\npixel -> composite", RED)]
    for k, (s, c) in enumerate(steps):
        x = 0.0 + k * 0.142
        box(ax, x, 0.05, 0.125, 0.17, s, c, 8.5)
        if k:
            arrow(ax, x - 0.016, 0.135, x, 0.135)
    save(fig, cfg.name, cfg.dpi)


if __name__ == "__main__":
    main(AppConfig())
