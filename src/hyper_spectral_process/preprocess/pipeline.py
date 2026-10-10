"""Run the configured stages on one map and keep a snapshot after every stage.

:func:`run_pipeline` is pure: it returns the processed map, the snapshot of
every stage (for QA plots and t-SNE), the baseline curves and the derivative
variants, and writes nothing. Files are written by :mod:`.runner`.

Snapshot keys: ``source``, ``roi``, ``spike`` (or ``spike1``/``spike2`` for
``wh2``), ``augment``, ``norm``, ``smooth``, ``smooth2nd``, ``baseline``,
``interp`` and, for every derivative order ``k``, ``deriv{k}_pre`` (if
pre-smoothed), ``deriv{k}_plain`` and ``deriv{k}``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..io import StandardSpectrumMap
from .augmentation import mixup
from .baseline import subtract_baseline
from .config import PreprocessConfig, validate_config
from .derivative import axis_delta, savgol_derivative
from .despike import despike_simple, despike_two_pass, despike_whitaker_hayes
from .interpolate import interpolate
from .normalize import normalize
from .roi import crop_roi
from .smooth import smooth_savgol


@dataclass
class PipelineResult:
    """Output of :func:`run_pipeline`.

    ``splits``, ``is_augmented`` and ``parent`` are aligned with the rows of
    ``final``: the split label, whether the row is synthetic, and the input row
    it comes from (for a mixup spectrum: its first source). ``derivatives``
    maps a variant name ``"procd{k}"`` to the derivative map, computed at the
    position of ``"deriv"`` in the stage order (rows aligned with
    ``deriv_rows``, a copy of the row bookkeeping at that position).
    """

    final: StandardSpectrumMap
    stages: dict
    trace_order: list
    splits: np.ndarray
    is_augmented: np.ndarray
    parent: np.ndarray
    baseline_curve: Optional[np.ndarray] = None
    despike_reference: Optional[np.ndarray] = None
    derivatives: dict = field(default_factory=dict)
    deriv_rows: Optional[tuple] = None
    mixup_examples: list = field(default_factory=list)


def _augment(ssm: StandardSpectrumMap, splits: np.ndarray, is_aug: np.ndarray, parent: np.ndarray,
             cfg: PreprocessConfig, rng: np.random.Generator):
    aug = cfg.augmentation
    enabled = ["all"] if set(splits.tolist()) == {"all"} else list(aug.splits)
    blocks, new_parent, new_splits, examples = [], [], [], []
    for name in enabled:
        n_new = int(aug.max_new_per_split.get(name, 0))
        if n_new <= 0:
            continue
        pool = np.flatnonzero((splits == name) & ~is_aug)
        y, first, ex = mixup(ssm.spectra[pool], n_new, rng, min_sources=aug.mixup_min_sources,
                             max_sources=aug.mixup_max_sources, alpha_range=tuple(aug.mixup_alpha_range),
                             noise_mode=aug.noise_mode, noise_std=aug.noise_std,
                             lowpass_sigma_points=aug.lowpass_sigma_points,
                             lowpass_kernel_radius=aug.lowpass_kernel_radius, lowpass_pad_mode=aug.lowpass_pad_mode)
        for e in ex:
            e["sources"] = pool[e["sources"]]
            e["split"] = name
        blocks.append(y)
        new_parent.append(parent[pool[first]])
        new_splits += [name] * n_new
        examples += ex
    if not blocks:
        return ssm, splits, is_aug, parent, examples
    added = np.vstack(blocks)
    n0, n_add = ssm.n_spectra, added.shape[0]
    # synthetic spectra get placeholder coordinates (row number, -1)
    new_xy = np.column_stack([np.arange(n0, n0 + n_add, dtype=np.float64), np.full(n_add, -1.0)])
    out = StandardSpectrumMap(np.vstack([ssm.spectra, added]), np.vstack([ssm.xy, new_xy]), ssm.axis, ssm.unit)
    return (out, np.concatenate([splits, np.array(new_splits, dtype=str)]),
            np.concatenate([is_aug, np.ones(n_add, dtype=bool)]), np.concatenate([parent] + new_parent), examples)


def run_pipeline(ssm: StandardSpectrumMap, cfg: PreprocessConfig, *, splits: Optional[np.ndarray] = None,
                 aug_seed: int = 0, skip_augmentation: bool = False) -> PipelineResult:
    """Apply ``cfg.stage_order`` to ``ssm``.

    Parameters
    ----------
    splits : split label of every input row (default: ``"all"`` for every
        row). Compute it with :func:`~.augmentation.split_assignments`; the
        ``split`` stage itself does not change the data.
    aug_seed : seed of the augmentation random generator.
    skip_augmentation : run the same stages without ``augmentation`` (used
        for the no-augmentation reference output).
    """
    validate_config(cfg)
    n = ssm.n_spectra
    splits = np.full(n, "all") if splits is None else np.asarray(splits, dtype=str)
    if splits.shape != (n,):
        raise ValueError(f"splits must have shape ({n},)")
    is_aug, parent = np.zeros(n, dtype=bool), np.arange(n)
    rng = np.random.default_rng(int(aug_seed))
    stages, trace = {"source": ssm}, ["source"]
    res = PipelineResult(final=ssm, stages=stages, trace_order=trace, splits=splits, is_augmented=is_aug,
                         parent=parent)
    cur = ssm

    def keep(key, value):
        stages[key] = value
        trace.append(key)

    for name in cfg.stage_order:
        if name == "split" or (name == "augmentation" and skip_augmentation):
            continue
        if name == "roi":
            r = cfg.roi
            cur = crop_roi(cur, r.xmin, r.xmax, round_decimals=r.round_decimals)
            keep("roi", cur)
        elif name == "despike":
            d = cfg.despike
            if d.mode == "simple":
                cur = despike_simple(cur, d.sf_threshold)
                keep("spike", cur)
            elif d.mode == "wh":
                cur = despike_whitaker_hayes(cur, d.wh_threshold, d.wh_m, max_width=d.wh_max_width, protect=d.protect)
                keep("spike", cur)
            else:
                p1, cur, res.despike_reference = despike_two_pass(
                    cur, wh_threshold=d.wh1_threshold, wh_m=d.wh1_m, wh_max_width=d.wh_max_width,
                    ref_window=d.resid_ref_window, ref_polyorder=d.resid_ref_polyorder,
                    resid_threshold=d.resid_threshold, resid_m=d.resid_m, resid_replace=d.resid_replace,
                    protect=d.protect)
                keep("spike1", p1)
                keep("spike2", cur)
        elif name in ("smooth", "smooth2nd"):
            s = getattr(cfg, name)
            cur = smooth_savgol(cur, s.window, s.polyorder, mode=s.mode, strength=s.strength)
            keep(name, cur)
        elif name == "baseline":
            b = cfg.baseline
            ref = None if b.anchor_reference is None else np.asarray(b.anchor_reference, dtype=np.float64)
            cur, res.baseline_curve = subtract_baseline(
                cur, b.mode, minline_window=b.minline_window, minline_step=b.minline_step,
                ps_window_min=b.ps_window_min, ps_window_max=b.ps_window_max, ps_polyorder=b.ps_polyorder,
                protect_peaks=b.protect_peaks, protect_half_width=b.protect_half_width, anchors=b.anchors,
                anchor_degree=b.anchor_degree, anchor_reference=ref, clip_negative=b.clip_negative)
            keep("baseline", cur)
        elif name == "normalize":
            p = cfg.normalize
            cur = normalize(cur, p.mode, round_decimals=p.round_decimals, peak_window=p.peak_window,
                            peak_refine=p.peak_refine)
            keep("norm", cur)
        elif name == "interp":
            cur = interpolate(cur, cfg.interp.step, cfg.interp.round_decimals)
            keep("interp", cur)
        elif name == "augmentation":
            cur, splits, is_aug, parent, res.mixup_examples = _augment(cur, splits, is_aug, parent, cfg, rng)
            keep("augment", cur)
        elif name == "deriv":
            dcfg = cfg.derivative
            delta = axis_delta(cur.axis, dcfg.use_axis_delta, dcfg.delta_override)
            qa_orders = sorted(set(dcfg.orders) | ({1} if max(dcfg.orders) >= 2 else set()))
            for k in qa_orders:
                st = savgol_derivative(cur, k, dcfg.window, dcfg.polyorder, delta=delta, pre=dcfg.pre_smooth,
                                       post=dcfg.post_for(k))
                if st.pre is not None:
                    keep(f"deriv{k}_pre", st.pre)
                keep(f"deriv{k}_plain", st.plain)
                keep(f"deriv{k}", st.final)
                if k in dcfg.orders:
                    res.derivatives[f"procd{k}"] = st.final
            res.deriv_rows = (splits.copy(), is_aug.copy(), parent.copy())
    res.final, res.splits, res.is_augmented, res.parent = cur, splits, is_aug, parent
    return res
