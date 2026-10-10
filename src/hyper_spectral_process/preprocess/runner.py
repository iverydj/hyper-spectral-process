"""Batch preprocessing of several map files with metadata, NPZ outputs and QA images.

:func:`run_preprocess` validates everything first (configuration, input
records, NPZ schemas, output directory), then runs :func:`~.pipeline.run_pipeline`
on every file and writes::

    <out_root>/<run_id>/
        npz_proc_d0/<source_id>__proc.npz           processed spectra
        npz_proc_d<k>/<source_id>__procd<k>.npz     derivative variants
        (..._trainset/_testset/_valset.npz          per split, when splitting is enabled)
        npz_noaug_d<k>/, npz_noaug_source/          optional no-augmentation reference
        metadata/samples_metadata.csv               one row per output spectrum and variant
        metadata/label_map.json                     class name <-> class id
        metadata/run_config_snapshot.json
        processing_stats.csv
        qa_images/                                  stage QA, traces, t-SNE
"""
from __future__ import annotations

import csv
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..io import StandardSpectrumMap
from . import qa
from .augmentation import split_assignments
from .config import PreprocessConfig, validate_config
from .derivative import axis_delta, savgol_derivative
from .pipeline import PipelineResult, run_pipeline

log = logging.getLogger(__name__)
SOURCE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
BASE_COLUMNS = ["sample_id", "source_id", "source_file", "source_path", "class_name", "class_id", "variant",
                "split", "is_augmented", "parent_sample_id", "axis_min", "axis_max", "axis_step", "spec_len"]


@dataclass(frozen=True)
class InputFile:
    """One input map: a StandardSpectrumMap NPZ, its unique ``source_id`` and its class.

    ``metadata`` holds free-form extra columns (e.g. substrate, analyte)
    copied into ``samples_metadata.csv``.
    """

    path: Path
    source_id: str
    class_name: str
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if not SOURCE_ID_PATTERN.match(self.source_id):
            raise ValueError(f"source_id may contain only letters, digits, '.', '_' and '-': {self.source_id!r}")
        if not str(self.class_name).strip():
            raise ValueError(f"class_name of {self.source_id!r} is empty")
        if self.path.suffix.lower() != ".npz":
            raise ValueError(f"input must be a StandardSpectrumMap .npz file: {self.path}")


def read_input_table(csv_path: str | Path) -> list[InputFile]:
    """Input records from a CSV with columns ``path``, ``source_id``, ``class_name`` (others become metadata).

    Relative paths are resolved from the CSV's directory.
    """
    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = {"path", "source_id", "class_name"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path}: missing column(s) {sorted(missing)}")
        records = []
        for row in reader:
            p = Path(row["path"])
            extra = {k: v for k, v in row.items() if k not in ("path", "source_id", "class_name")}
            records.append(InputFile(p if p.is_absolute() else csv_path.parent / p, row["source_id"].strip(),
                                     row["class_name"].strip(), extra))
    return records


def _stats_row(obj: StandardSpectrumMap, stem: str, stage: str) -> dict:
    return {"file": stem, "stage": stage, "N": obj.n_spectra, "M": obj.n_points, "axis_min": obj.axis[0],
            "axis_max": obj.axis[-1], "mean": obj.spectra.mean(), "std": obj.spectra.std(),
            "min": obj.spectra.min(), "max": obj.spectra.max()}


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _stage_qa(res: PipelineResult, src: StandardSpectrumMap, cfg: PreprocessConfig, qa_dir: Path, stem: str) -> None:
    """Before/after image of every stage, derivative overviews and per-spectrum traces."""
    q, dpi = cfg.qa, cfg.qa.dpi
    order = res.trace_order
    step = 1
    for p in range(1, len(order)):
        prev, cur = res.stages[order[p - 1]], res.stages[order[p]]
        name = order[p]
        if name.endswith("_plain") or name.endswith("_pre"):
            continue
        tag = f"{stem}_{step:02d}"
        if name == "augment":
            fig = qa.plot_mixup_examples(prev, res.mixup_examples, n=q.aug_mixup_n)
        elif name == "baseline":
            fig = qa.plot_spectrum_grid(prev, cur, prev.replace(spectra=res.baseline_curve),
                                        labels=("before", "after", "baseline"), suptitle=f"{step}) baseline")
        elif name == "spike2":
            fig = qa.plot_spectrum_grid(prev, cur, prev.replace(spectra=res.despike_reference), yscale="log",
                                        labels=("pass 1", "pass 2", "reference"), suptitle=f"{step}) despike pass 2")
        elif re.fullmatch(r"deriv\d+", name):
            k = int(name[5:])
            plain = res.stages[f"deriv{k}_plain"]
            fig = qa.plot_spectrum_grid(plain, cur, labels=("plain derivative", "final"),
                                        suptitle=f"{step}) derivative d{k}: plain vs final")
            _save(fig, qa_dir / f"{tag}_deriv{k}_plain_vs_final.png", dpi)
            # reference: plain Savitzky-Golay derivative of the mean source spectrum, no other stage
            mean_src = src.replace(spectra=src.spectra.mean(axis=0, keepdims=True), xy=src.xy[:1])
            d = cfg.derivative
            ref = savgol_derivative(mean_src, k, d.window, d.polyorder,
                                    delta=axis_delta(src.axis, d.use_axis_delta, d.delta_override)).plain
            inside = (ref.axis >= cur.axis[0]) & (ref.axis <= cur.axis[-1])
            fig = qa.plot_overview(cur, f"{stem}: derivative d{k}", n=q.trace_n, seed=q.trace_seed,
                                   reference=(ref.axis[inside], ref.spectra[0, inside]),
                                   reference_label=f"d{k} of the mean source spectrum (no other stage)")
            _save(fig, qa_dir / f"{tag}_deriv{k}_overview.png", dpi)
            fig = qa.plot_total_intensity(cur, f"{stem}: derivative d{k}, summed")
            _save(fig, qa_dir / f"{tag}_deriv{k}_heatmap.png", dpi)
            step += 1
            continue
        else:
            yscale = "log" if name in qa.LOG_STAGES else "linear"
            fig = qa.plot_spectrum_grid(prev, cur, yscale=yscale, seed=q.trace_seed,
                                        suptitle=f"{step}) {order[p - 1]} -> {name}")
        _save(fig, qa_dir / f"{tag}_prev_vs_{name}.png", dpi)
        step += 1
    if q.trace_enable:
        n_src = src.n_spectra
        if q.trace_idxs is not None:
            rows = np.asarray(q.trace_idxs, dtype=int)
            if rows.size != np.unique(rows).size or rows.min() < 0 or rows.max() >= n_src:
                raise ValueError(f"qa.trace_idxs must be distinct rows in [0, {n_src})")
        elif q.trace_mode == "all":
            rows = np.arange(n_src)
        else:
            rows = qa.sample_rows(n_src, q.trace_n, q.trace_seed)
        for i in rows:
            fig = qa.plot_trace(res.stages, order, int(i), baseline_curve=res.baseline_curve,
                                despike_reference=res.despike_reference)
            _save(fig, qa_dir / "trace" / stem / f"{stem}_trace_idx{int(i):06d}.png", dpi)


def _metadata_rows(rec: InputFile, variant: str, obj: StandardSpectrumMap, rows: tuple) -> list[dict]:
    splits, is_aug, parent = rows
    step = float(np.median(np.diff(obj.axis)))
    return [{"sample_id": f"{rec.source_id}__{variant}__idx{i}", "source_id": rec.source_id,
             "source_file": rec.path.name, "source_path": str(rec.path), "class_name": rec.class_name,
             "variant": variant, "split": splits[i], "is_augmented": int(is_aug[i]),
             "parent_sample_id": f"{rec.source_id}__source__idx{int(parent[i])}", "axis_min": obj.axis[0],
             "axis_max": obj.axis[-1], "axis_step": step, "spec_len": obj.n_points, **dict(rec.metadata)}
            for i in range(obj.n_spectra)]


def _save_variant(obj: StandardSpectrumMap, splits: np.ndarray, out_dir: Path, stem: str, cfg: PreprocessConfig):
    out_dir.mkdir(parents=True, exist_ok=True)
    obj.save(out_dir / f"{stem}.npz")
    if cfg.split.enable:
        for name in ("train", "test", "valid"):
            obj.select(splits == name).save(out_dir / f"{stem}_{cfg.split.file_tags[name]}.npz")


def _variants(res: PipelineResult, src: StandardSpectrumMap, include_source: bool, src_rows: tuple) -> dict:
    """``variant name -> (map, (splits, is_augmented, parent))``."""
    out = {}
    if include_source:
        out["source"] = (src, src_rows)
    out["proc"] = (res.final, (res.splits, res.is_augmented, res.parent))
    for name, obj in res.derivatives.items():
        out[name] = (obj, res.deriv_rows)
    return out


def run_preprocess(inputs: Sequence[InputFile], cfg: PreprocessConfig, out_root: str | Path,
                   run_id: Optional[str] = None) -> Path:
    """Preprocess every input map with ``cfg``; returns the run directory (see the module docstring)."""
    t0 = time.perf_counter()
    validate_config(cfg)
    inputs = list(inputs)
    if not inputs:
        raise ValueError("no input files")
    ids = [r.source_id for r in inputs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicated source_id in the inputs: {sorted({i for i in ids if ids.count(i) > 1})}")
    for rec in inputs:
        if not rec.path.exists():
            raise FileNotFoundError(f"input file not found: {rec.path}")
        StandardSpectrumMap.load(rec.path)  # schema check before any output is written
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_run = Path(out_root) / run_id
    if out_run.exists() and any(out_run.iterdir()) and not cfg.output.allow_existing_run_dir:
        raise FileExistsError(f"output run directory exists and is not empty: {out_run}")
    qa_dir, meta_dir = out_run / "qa_images", out_run / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    qa_on = cfg.qa.enable
    label_map = {name: i for i, name in enumerate(sorted({r.class_name for r in inputs}))}
    rows_meta, stats, finals, tsne_sources = [], [], {}, []
    for i_src, rec in enumerate(inputs):
        src = StandardSpectrumMap.load(rec.path)
        stem = rec.source_id
        log.info("%s: N=%d, M=%d, axis %.4g..%.4g %s", stem, src.n_spectra, src.n_points, src.axis[0],
                 src.axis[-1], src.unit)
        split_seed = cfg.random_seed + i_src * 7919
        splits = (split_assignments(src.n_spectra, cfg.split.train, cfg.split.valid, cfg.split.test, split_seed)
                  if cfg.split.enable else np.full(src.n_spectra, "all"))
        res = run_pipeline(src, cfg, splits=splits, aug_seed=split_seed + 137)
        stats += [_stats_row(src, stem, "source"), _stats_row(res.final, stem, "proc")]
        finals[stem] = res.final
        if qa_on:
            _save(qa.plot_overview(src, f"{stem}: source"), qa_dir / f"{stem}_00_source_overlay.png", cfg.qa.dpi)
            _save(qa.plot_total_intensity(src, f"{stem}: source"), qa_dir / f"{stem}_00_source_heatmap.png",
                  cfg.qa.dpi)
            _stage_qa(res, src, cfg, qa_dir, stem)
            _save(qa.plot_overview(res.final, f"{stem}: processed"), qa_dir / f"{stem}_final_overlay.png", cfg.qa.dpi)
            if cfg.qa.tsne_enable:
                tsne_sources.append((rec.class_name, res.stages))
        src_rows = (splits, np.zeros(src.n_spectra, dtype=bool), np.arange(src.n_spectra))
        for variant, (obj, rows) in _variants(res, src, cfg.output.include_source_variant, src_rows).items():
            rows_meta += _metadata_rows(rec, variant, obj, rows)
            if variant != "source" and cfg.output.save_processed_npz:
                order = 0 if variant == "proc" else int(variant[5:])
                _save_variant(obj, rows[0], out_run / f"npz_proc_d{order}", f"{stem}__{variant}", cfg)
        if cfg.output.save_noaug_reference_npz:
            ref = run_pipeline(src, cfg, splits=splits, aug_seed=split_seed + 137, skip_augmentation=True)
            for variant, (obj, rows) in _variants(ref, src, cfg.output.include_source_variant, src_rows).items():
                sub = "npz_noaug_source" if variant == "source" else \
                    f"npz_noaug_d{0 if variant == 'proc' else int(variant[5:])}"
                _save_variant(obj, rows[0], out_run / sub, f"{stem}__{variant}", cfg)
    for row in rows_meta:
        row["class_id"] = label_map[row["class_name"]]
    if cfg.split.enable:
        by_parent: dict = {}
        for row in rows_meta:
            by_parent.setdefault(row["parent_sample_id"], set()).add(row["split"])
        leaks = [k for k, v in by_parent.items() if len(v) > 1]
        if leaks:
            raise RuntimeError(f"split leakage for {len(leaks)} parent spectra")
    extra_cols = sorted({k for r in rows_meta for k in r} - set(BASE_COLUMNS))
    with (meta_dir / "samples_metadata.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=BASE_COLUMNS + extra_cols)
        w.writeheader()
        w.writerows(rows_meta)
    (meta_dir / "label_map.json").write_text(json.dumps(
        {"class_name_to_id": label_map, "id_to_class_name": {str(v): k for k, v in label_map.items()}}, indent=2))
    (meta_dir / "run_config_snapshot.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str))
    with (out_run / "processing_stats.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(stats[0]))
        w.writeheader()
        w.writerows(stats)
    if qa_on:
        if len(finals) > 1:
            _save(qa.plot_cross_file_comparison(finals), qa_dir / "cross_file_comparison.png", cfg.qa.dpi)
        if cfg.qa.tsne_enable:
            _tsne(tsne_sources, cfg, qa_dir / "tsne")
    log.info("done: %d files, %d metadata rows, %.1f s -> %s", len(inputs), len(rows_meta),
             time.perf_counter() - t0, out_run)
    return out_run


def _tsne(sources: list, cfg: PreprocessConfig, out_dir: Path) -> None:
    q = cfg.qa
    keys = list(q.tsne_stage_keys) if q.tsne_stage_keys is not None else \
        qa.default_tsne_keys(cfg.stage_order, sources[0][1], cfg.derivative.orders)
    perplexities = list(dict.fromkeys(float(p) for p in np.atleast_1d(q.tsne_perplexity)))
    for perp in perplexities:
        emb = qa.tsne_embeddings(sources, keys, perplexity=perp, max_iter=q.tsne_max_iter,
                                 random_state=q.tsne_random_state, n_jobs=q.tsne_n_jobs,
                                 thread_limit=q.tsne_thread_limit)
        sub = out_dir / f"perp_{int(round(perp)):03d}" if len(perplexities) > 1 else out_dir
        fig = qa.plot_tsne_stages(emb, max_cols=q.tsne_max_cols, point_size=q.tsne_point_size, alpha=q.tsne_alpha,
                                  title=f"Class-wise t-SNE across preprocessing stages (perplexity {perp:g})")
        _save(fig, sub / "stage_tsne_subplots.png", q.dpi)
        summary = {"stage_keys": keys, "perplexity": perp,
                   "stages": {k: {"n_samples": int(v[0].shape[0]), "kl_divergence": v[2]} for k, v in emb.items()}}
        (sub / "stage_tsne_summary.json").write_text(json.dumps(summary, indent=2))
