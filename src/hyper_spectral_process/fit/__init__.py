"""Batched multi-peak fitting with a staged Adam optimisation and a soft spring prior.

Typical use::

    cfg = FitConfig(peaks=(PeakSpec(...), ...), stages=(StageSpec("coarse", 15000, 5e-3),))
    fit = fit_map(ssm, cfg, MapFitOptions(devices=("cuda:0",)), initial_guess=InitialGuessOptions())
    fit.result.save("fit.npz")
"""
from .engine import BatchFitter, fit_batches, resolve_devices
from .fit_map import MapFit, fit_map
from .initial_guess import InitialGuess, InitialGuessOptions, average_rows, bic, fit_initial_guess, format_peaks_block
from .lineshapes import (AREA_C_GAUSS, AREA_C_LORENTZ, fano, gaussian, lorentzian, peak_area, peak_fwhm, pseudo_voigt,
                         pseudo_voigt_tch, tch_fwhm_eta)
from .model import PeakModel, PeakTable
from .objective import data_term, prior_term, spring
from .outputs import save_fit_outputs
from .prepare import MapFitOptions, PreparedMap, config_to_energy, peak_to_energy, peak_to_wavelength, prepare_map
from .result import FitSpectrumMap, fit_metrics, write_metrics_csv, write_params_csv
from .specs import PEAK_TYPES, FitConfig, PeakSpec, StageSpec, canonical_peak_type
from .variants import VariantSelection, composite_fit, fit_variants, input_peaks_of, select_variants, slot_map

__all__ = [
    "PEAK_TYPES", "PeakSpec", "StageSpec", "FitConfig", "canonical_peak_type",
    "gaussian", "lorentzian", "pseudo_voigt", "pseudo_voigt_tch", "tch_fwhm_eta", "fano", "peak_area", "peak_fwhm",
    "AREA_C_GAUSS", "AREA_C_LORENTZ", "PeakTable", "PeakModel", "spring", "data_term", "prior_term",
    "BatchFitter", "fit_batches", "resolve_devices", "MapFitOptions", "PreparedMap", "prepare_map", "peak_to_energy",
    "peak_to_wavelength", "config_to_energy", "InitialGuessOptions", "InitialGuess", "average_rows",
    "fit_initial_guess", "format_peaks_block", "bic", "MapFit", "fit_map", "FitSpectrumMap", "fit_metrics",
    "write_params_csv", "write_metrics_csv", "save_fit_outputs", "VariantSelection", "fit_variants",
    "select_variants", "composite_fit", "slot_map", "input_peaks_of",
]
