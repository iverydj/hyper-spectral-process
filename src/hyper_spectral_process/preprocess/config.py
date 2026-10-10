"""Configuration of the stage-ordered preprocessing pipeline and its validation.

Only the stages listed in :attr:`PreprocessConfig.stage_order` run, in that
order; the parameter blocks of unlisted stages are ignored (they never
enable a stage). :func:`validate_config` rejects ambiguous configurations
before any data is processed. Defaults follow the original run
configuration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping, Optional

from .baseline import BASELINE_MODES
from .derivative import SmoothBlock
from .normalize import NORMALIZE_MODES

STAGES = ("split", "roi", "despike", "augmentation", "normalize", "smooth", "baseline", "smooth2nd", "interp", "deriv")
DESPIKE_MODES = ("simple", "wh", "wh2")


@dataclass(frozen=True)
class RoiParams:
    """``roi``: keep ``xmin <= axis <= xmax`` (:func:`~.roi.crop_roi`)."""
    xmin: Optional[float] = 250.0
    xmax: Optional[float] = 3200.0
    round_decimals: Optional[int] = None


@dataclass(frozen=True)
class DespikeParams:
    """``despike`` (:mod:`.despike`). ``mode`` selects which group of parameters is used."""
    mode: str = "none"                  # "simple" | "wh" | "wh2"; must be set when the stage is listed
    sf_threshold: float = 40.0          # simple: jump threshold in data units
    wh_threshold: float = 3.5           # wh: modified z-score threshold
    wh_m: int = 8                       # wh: neighbour half-width for the replacement
    wh_max_width: Optional[int] = 2     # wh: widest spike of the pair rule; None = published rule
    wh1_threshold: float = 3.5          # wh2 pass 1
    wh1_m: int = 8
    wh1_max_width: Optional[int] = 2    # wh2 pass 1: as wh_max_width
    resid_ref_window: int = 7           # wh2 pass 2: Savitzky-Golay reference
    resid_ref_polyorder: int = 3
    resid_threshold: float = 5.0
    resid_m: int = 2
    resid_replace: str = "ref"          # "ref" | "mean"
    protect: tuple = ()                 # axis ranges (lo, hi) never modified (wh/wh2)


@dataclass(frozen=True)
class SmoothParams:
    """``smooth`` / ``smooth2nd`` (:func:`~.smooth.smooth_savgol`)."""
    window: int = 7
    polyorder: int = 3
    mode: str = "interp"
    strength: int = 2


@dataclass(frozen=True)
class NormalizeParams:
    """``normalize`` (:func:`~.normalize.normalize`)."""
    mode: str = "instance"              # "instance" | "global" | "peak"
    round_decimals: Optional[int] = 3
    peak_window: Optional[tuple] = None  # (xmin, xmax), mode "peak" only
    peak_refine: int = 2


@dataclass(frozen=True)
class BaselineParams:
    """``baseline`` (:func:`~.baseline.subtract_baseline`)."""
    mode: str = "none"                  # "minline" | "peakstrip" | "anchors"; must be set when listed
    minline_window: int = 500
    minline_step: int = 150
    ps_window_min: int = 121
    ps_window_max: int = 1111
    ps_polyorder: int = 0
    protect_peaks: Optional[tuple] = None
    protect_half_width: float = 15.0
    clip_negative: bool = False
    anchors: tuple = ()
    anchor_degree: Optional[int] = 1
    anchor_reference: Optional[tuple] = None


@dataclass(frozen=True)
class InterpParams:
    """``interp`` (:func:`~.interpolate.interpolate`)."""
    step: float = 2.0
    round_decimals: Optional[int] = None


def _default_post_smooth() -> dict:
    return {1: SmoothBlock(True, 2, 13, 3), 2: SmoothBlock(True, 2, 13, 3)}


@dataclass(frozen=True)
class DerivativeParams:
    """``deriv``: derivative variants (:func:`~.derivative.savgol_derivative`).

    ``post_smooth`` maps an order to its post-smoothing; orders not listed
    use ``post_smooth_default``.
    """
    orders: tuple = (1, 2)
    window: int = 11
    polyorder: int = 3
    use_axis_delta: bool = True
    delta_override: float = 1.0
    pre_smooth: SmoothBlock = SmoothBlock(False, 0, 11, 3)
    post_smooth: Mapping[int, SmoothBlock] = field(default_factory=_default_post_smooth)
    post_smooth_default: SmoothBlock = SmoothBlock(False, 0, 9, 2)

    def post_for(self, order: int) -> SmoothBlock:
        return self.post_smooth.get(int(order), self.post_smooth_default)


@dataclass(frozen=True)
class SplitParams:
    """``split``: train/valid/test assignment (:func:`~.augmentation.split_assignments`)."""
    enable: bool = False
    train: float = 0.6
    valid: float = 0.2
    test: float = 0.2
    file_tags: Mapping[str, str] = field(default_factory=lambda: {"train": "trainset", "test": "testset",
                                                                 "valid": "valset"})


@dataclass(frozen=True)
class AugmentationParams:
    """``augmentation``: mixup and noise per split (:func:`~.augmentation.mixup`)."""
    splits: tuple = ("train", "valid", "test")
    max_new_per_split: Mapping[str, int] = field(default_factory=lambda: {"train": 300, "test": 100,
                                                                         "valid": 100, "all": 500})
    mixup_min_sources: int = 2
    mixup_max_sources: int = 4
    mixup_alpha_range: tuple = (0.35, 0.65)
    noise_mode: str = "white"           # "white" | "lowpass"
    lowpass_sigma_points: float = 4.0
    lowpass_kernel_radius: Optional[int] = None
    lowpass_pad_mode: str = "reflect"
    noise_std: float = 0.005


@dataclass(frozen=True)
class QAParams:
    """Quality-assurance images written by :func:`~.runner.run_preprocess`."""
    enable: bool = True
    trace_enable: bool = True           # one image per traced spectrum with every stage
    trace_mode: str = "random"          # "random" | "all"
    trace_n: int = 9
    trace_seed: int = 42
    trace_idxs: Optional[tuple] = None  # explicit rows (override trace_mode)
    aug_mixup_n: int = 9
    tsne_enable: bool = True
    tsne_stage_keys: Optional[tuple] = None
    tsne_perplexity: tuple = (5, 10, 20)  # several values: one image per value
    tsne_max_iter: int = 1000
    tsne_random_state: int = 123
    tsne_n_jobs: int = 1
    tsne_thread_limit: int = 1
    tsne_max_cols: int = 5
    tsne_point_size: float = 7.0
    tsne_alpha: float = 0.62
    dpi: int = 150


@dataclass(frozen=True)
class OutputParams:
    """What :func:`~.runner.run_preprocess` writes."""
    allow_existing_run_dir: bool = False
    save_processed_npz: bool = True
    include_source_variant: bool = False   # also save the unprocessed input as variant "source"
    save_noaug_reference_npz: bool = False  # also save the same pipeline without augmentation


@dataclass(frozen=True)
class PreprocessConfig:
    """Complete preprocessing configuration.

    ``stage_order`` lists the stages to run (a subset of :data:`STAGES`, no
    repeats). ``split`` must come first when ``split.enable`` is True and must
    be absent otherwise.
    """
    stage_order: tuple
    roi: RoiParams = RoiParams()
    despike: DespikeParams = DespikeParams()
    smooth: SmoothParams = SmoothParams()
    smooth2nd: SmoothParams = SmoothParams(window=5, polyorder=3, mode="interp", strength=1)
    normalize: NormalizeParams = NormalizeParams()
    baseline: BaselineParams = BaselineParams()
    interp: InterpParams = InterpParams()
    derivative: DerivativeParams = DerivativeParams()
    split: SplitParams = SplitParams()
    augmentation: AugmentationParams = AugmentationParams()
    qa: QAParams = QAParams()
    output: OutputParams = OutputParams()
    random_seed: int = 123

    def to_dict(self) -> dict:
        """Plain-data snapshot (for the run's ``run_config_snapshot.json``)."""
        return asdict(self)


def validate_config(cfg: PreprocessConfig) -> None:
    """Raise ``ValueError`` for an inconsistent configuration.

    Checks the stage order (known stages, no repeats, ``split`` first and only
    when enabled), that ``despike.mode`` and ``baseline.mode`` are set exactly
    when their stage is listed, that derivative orders imply the ``deriv``
    stage, the split ratios, the normalization and augmentation settings and
    the output options.
    """
    order = tuple(cfg.stage_order)
    unknown = [s for s in order if s not in STAGES]
    if unknown:
        raise ValueError(f"unknown stage(s) {unknown}; allowed: {STAGES}")
    if len(set(order)) != len(order):
        raise ValueError(f"stage_order contains a repeated stage: {order}")
    if "split" in order and order[0] != "split":
        raise ValueError("'split' must be the first stage")
    if cfg.split.enable and "split" not in order:
        raise ValueError("split.enable=True requires 'split' as the first stage")
    if not cfg.split.enable and "split" in order:
        raise ValueError("split.enable=False requires removing 'split' from stage_order")
    for name in ("train", "valid", "test"):
        if getattr(cfg.split, name) < 0:
            raise ValueError("split ratios must be non-negative")
    if abs(cfg.split.train + cfg.split.valid + cfg.split.test - 1.0) > 1e-9:
        raise ValueError("split ratios train + valid + test must sum to 1")
    if set(cfg.split.file_tags) != {"train", "valid", "test"} or not all(cfg.split.file_tags.values()):
        raise ValueError("split.file_tags needs non-empty tags for 'train', 'valid' and 'test'")

    if "despike" in order and cfg.despike.mode not in DESPIKE_MODES:
        raise ValueError(f"despike.mode must be one of {DESPIKE_MODES} when 'despike' is listed; "
                         f"got {cfg.despike.mode!r}")
    if cfg.despike.resid_replace not in ("ref", "mean"):
        raise ValueError("despike.resid_replace must be 'ref' or 'mean'")
    if "baseline" in order and cfg.baseline.mode not in BASELINE_MODES:
        raise ValueError(f"baseline.mode must be one of {BASELINE_MODES} when 'baseline' is listed; "
                         f"got {cfg.baseline.mode!r}")
    if "baseline" not in order and cfg.baseline.mode != "none":
        raise ValueError(f"baseline.mode={cfg.baseline.mode!r} is set but 'baseline' is not in stage_order")
    if cfg.normalize.mode not in NORMALIZE_MODES:
        raise ValueError(f"normalize.mode must be one of {NORMALIZE_MODES}")
    if "normalize" in order and cfg.normalize.mode == "peak" and cfg.normalize.peak_window is None:
        raise ValueError("normalize.mode='peak' requires normalize.peak_window")

    orders = tuple(cfg.derivative.orders)
    if any(int(o) != o or o < 1 for o in orders) or len(set(orders)) != len(orders):
        raise ValueError(f"derivative.orders must be distinct positive integers; got {orders}")
    if orders and "deriv" not in order:
        raise ValueError("derivative.orders is non-empty but 'deriv' is not in stage_order")
    if "deriv" in order and not orders:
        raise ValueError("'deriv' is listed but derivative.orders is empty")

    aug = cfg.augmentation
    for s in aug.splits:
        if s not in ("train", "valid", "test", "all"):
            raise ValueError(f"augmentation.splits entry {s!r} must be train, valid, test or all")
    if aug.noise_mode not in ("white", "lowpass"):
        raise ValueError("augmentation.noise_mode must be 'white' or 'lowpass'")

    out = cfg.output
    if not (out.save_processed_npz or out.save_noaug_reference_npz):
        raise ValueError("output must save at least one NPZ family")
    if out.save_noaug_reference_npz and "augmentation" not in order:
        raise ValueError("output.save_noaug_reference_npz=True requires 'augmentation' in stage_order")
    if out.include_source_variant and not out.save_noaug_reference_npz:
        raise ValueError("output.include_source_variant=True requires output.save_noaug_reference_npz=True")
    if cfg.qa.trace_mode not in ("random", "all"):
        raise ValueError("qa.trace_mode must be 'random' or 'all'")
