"""Figures qa_trace.png and qa_tsne.png: per-spectrum stage trace and class-wise t-SNE (synthetic data)."""
from dataclasses import dataclass

import numpy as np

from _common import gauss, lorentz, save
from hyper_spectral_process.io import StandardSpectrumMap
from hyper_spectral_process.preprocess import (BaselineParams, DerivativeParams, DespikeParams, NormalizeParams,
                                               PreprocessConfig, RoiParams, run_pipeline)
from hyper_spectral_process.preprocess.qa import default_tsne_keys, plot_trace, plot_tsne_stages, tsne_embeddings


@dataclass(frozen=True)
class AppConfig:
    trace_name: str = "qa_trace.png"
    tsne_name: str = "qa_tsne.png"
    dpi: int = 120
    n_per_class: int = 120
    seed: int = 6


PIPE = PreprocessConfig(stage_order=("roi", "despike", "baseline", "normalize", "deriv"),
                        roi=RoiParams(220, 780), despike=DespikeParams(mode="wh", wh_threshold=7.0, wh_m=3),
                        baseline=BaselineParams(mode="peakstrip", ps_window_min=21, ps_window_max=301),
                        normalize=NormalizeParams(round_decimals=None), derivative=DerivativeParams(orders=(1,)))


def make_class(rng, n, pos2, x):
    """Spectra of one class: a common band, a class-specific band position, random backgrounds and spikes."""
    bg = rng.uniform(20, 80, (n, 1)) * (1 + 0.6 * np.exp(-((x - 300) / 200) ** 2))
    y = bg + lorentz(x, 450, 12, 100) + lorentz(x, pos2 + rng.normal(0, 2, (n, 1)), 15, 40) + rng.normal(0, 2, (n, x.size))
    y[np.arange(n), rng.integers(10, x.size - 10, n)] += 300
    return StandardSpectrumMap(y, np.column_stack([np.arange(n), np.zeros(n)]), x, "cm^-1")


def main(cfg: AppConfig) -> None:
    rng = np.random.default_rng(cfg.seed)
    x = np.linspace(200, 800, 601)
    sources = []
    for name, pos2 in (("class A", 560.0), ("class B", 590.0), ("class C", 620.0)):
        res = run_pipeline(make_class(rng, cfg.n_per_class, pos2, x), PIPE)
        sources.append((name, res.stages))
        if name == "class A":
            fig = plot_trace(res.stages, res.trace_order, 0, baseline_curve=res.baseline_curve)
            save(fig, cfg.trace_name, cfg.dpi)
    keys = default_tsne_keys(PIPE.stage_order, sources[0][1], PIPE.derivative.orders)
    emb = tsne_embeddings(sources, keys, perplexity=30)
    save(plot_tsne_stages(emb, title="Class-wise t-SNE across preprocessing stages (synthetic, perplexity 30)"),
         cfg.tsne_name, cfg.dpi)


if __name__ == "__main__":
    main(AppConfig())
