"""Stage-ordered preprocessing of spectrum maps.

Every stage is a function of a :class:`~hyper_spectral_process.io.StandardSpectrumMap`
in its own module; :func:`run_pipeline` applies a :class:`PreprocessConfig`
and :func:`run_preprocess` processes whole runs of files with QA output.
"""
from .augmentation import gaussian_kernel, lowpass_noise, mixup, split_assignments, split_counts
from .baseline import (anchor_baseline, fill_masked, minline_baseline, peakstrip_baseline, protect_mask,
                       strip_windows, subtract_baseline)
from .config import (STAGES, AugmentationParams, BaselineParams, DerivativeParams, DespikeParams, InterpParams,
                     NormalizeParams, OutputParams, PreprocessConfig, QAParams, RoiParams, SmoothParams, SplitParams,
                     validate_config)
from .derivative import DerivativeSteps, SmoothBlock, axis_delta, savgol_derivative
from .despike import (despike_residual, despike_simple, despike_two_pass, despike_whitaker_hayes, residual_reference,
                      whitaker_hayes_mask)
from .interpolate import interpolate, interpolation_axis
from .normalize import normalize, peak_height
from .pipeline import PipelineResult, run_pipeline
from .roi import crop_roi
from .runner import InputFile, read_input_table, run_preprocess
from .smooth import smooth_savgol

__all__ = [
    "crop_roi", "despike_simple", "whitaker_hayes_mask", "despike_whitaker_hayes", "residual_reference",
    "despike_residual", "despike_two_pass", "smooth_savgol", "normalize", "peak_height", "minline_baseline",
    "peakstrip_baseline", "strip_windows", "protect_mask", "fill_masked", "anchor_baseline", "subtract_baseline",
    "interpolation_axis", "interpolate", "SmoothBlock", "DerivativeSteps", "axis_delta", "savgol_derivative",
    "split_counts", "split_assignments", "gaussian_kernel", "lowpass_noise", "mixup",
    "STAGES", "PreprocessConfig", "RoiParams", "DespikeParams", "SmoothParams", "NormalizeParams", "BaselineParams",
    "InterpParams", "DerivativeParams", "SplitParams", "AugmentationParams", "QAParams", "OutputParams",
    "validate_config", "PipelineResult", "run_pipeline", "InputFile", "read_input_table", "run_preprocess",
]
