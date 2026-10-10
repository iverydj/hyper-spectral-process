"""Worked example: turn measured Raman and PL maps of monolayer MoS2 into parameter maps.

Data: ``examples/data/mos2_wrinkled`` (monolayer MoS2 on a wrinkled
substrate, 88 x 81 points, 532 nm excitation; see its README). Both maps are
analysed in two stages, so that no baseline term competes with the bands:

1. Background extraction from band-free anchor ranges
   (``subtract_baseline(mode="anchors")``). Raman: a straight line through
   300-345 and 478-505 cm^-1 (the off-flake spectra are straight there). PL:
   the mean off-flake spectrum (substrate emission) scaled to every spectrum
   in an anchor at 2.17-2.21 eV, above the B exciton and below the Raman lines.
2. Band fit without a baseline term (``fit_baseline=False``), after scaling
   every spectrum by the height of its main band. Raman: weak broad band near
   376 cm^-1 (unassigned), Gaussian E2g, Lorentzian A1g. PL: the axis is
   converted to photon energy with the lambda^2 Jacobian; the models A0 + B,
   A0 + A- and A0 + A- + B (Gaussian A0, Lorentzian A-, Gaussian B) are
   compared pixel by pixel with the BIC.

Outputs: two figures, ``summary.json`` and the FitSpectrumMap files in
``AppConfig.out_dir``. Use ``device="cuda:0"`` on a GPU.
"""
import json
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402

from hyper_spectral_process.fit import (FitConfig, InitialGuessOptions, MapFitOptions, PeakSpec, StageSpec,  # noqa: E402
                                        composite_fit, fit_map, fit_variants, input_peaks_of, select_variants)
from hyper_spectral_process.io import (StandardSpectrumMap, raman_shift_to_wavelength, raster_indices,  # noqa: E402
                                       to_image, wavelength_to_energy)
from hyper_spectral_process.plot import plot_fit, plot_map, plot_variant_map  # noqa: E402
from hyper_spectral_process.preprocess import (anchor_baseline, despike_whitaker_hayes, peak_height,  # noqa: E402
                                               subtract_baseline)

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AppConfig:
    data_dir: Path = HERE / "data" / "mos2_wrinkled"
    out_dir: Path = Path("example_output") / "05_mos2"
    device: str = "cpu"
    batch_size: int = 512
    excitation_nm: float = 532.0
    candidate_sd: float = 5.0                  # fit pixels whose band height exceeds the median by this many robust SD
    raman_anchors: tuple = ((300.0, 345.0), (478.0, 505.0))   # cm^-1
    raman_window: tuple = (350.0, 418.0)       # cm^-1, band fit window
    a1g_window: tuple = (398.0, 410.0)         # cm^-1, candidate selection and scaling
    max_separation_sd: float = 4.0             # monolayer: A1g - E2g separation below median + this many robust SD
    pl_anchor_ev: tuple = (2.17, 2.21)
    pl_window_ev: tuple = (1.70, 2.12)
    a_band_ev: tuple = (1.80, 1.92)            # A-exciton band, candidate selection and scaling
    offflake_margin_px: int = 4                # off-flake: this many pixels away from every candidate
    reference_smooth_window: int = 31          # samples, smoothing of the mean off-flake PL spectrum
    interpolated_column: int = 747             # detector column interpolated by the export (zero weight in PL)


CONFIG = AppConfig()


def spec(ptype, pos, width, height, name, pos_limit, width_limit, reg):
    return PeakSpec(ptype, pos=pos, width=width, height=height, pos_limit=pos_limit, pos_limit_reg=reg,
                    width_limit=width_limit, width_limit_reg=reg, height_limit=0.5, height_limit_reg=0.0, name=name)


# Raman (cm^-1, heights relative to A1g); springs only on the weak broad band
RAMAN_PEAKS = (spec("gaussian", 376.0, 10.0, 0.08, "X376", 6.0, 10.0, 1e-4),
               spec("gaussian", 383.5, 4.0, 0.4, "E2g", 3.0, 3.0, 0.0),
               spec("lorentzian", 403.5, 5.0, 1.0, "A1g", 3.0, 3.0, 0.0))
RAMAN_STAGES = (StageSpec("coarse", 6000, 5e-3), StageSpec("fine", 4000, 1e-3))
# PL (eV, heights relative to the A band)
A0 = spec("gaussian", 1.87, 0.10, 0.6, "A0", 0.02, 0.04, 1e-4)
AM = spec("lorentzian", 1.83, 0.06, 0.4, "A-", 0.02, 0.04, 1e-4)
B = spec("gaussian", 2.00, 0.15, 0.2, "B", 0.05, 0.08, 1e-4)
PL_VARIANTS = (("A0 + B", (A0, B)), ("A0 + A-", (A0, AM)), ("A0 + A- + B", (A0, AM, B)))
PL_STAGES = (StageSpec("coarse", 6000, 2e-3), StageSpec("fine", 4000, 2e-4))


def robust_above(values, n_sd):
    med = np.median(values)
    return values > med + n_sd * 1.4826 * np.median(np.abs(values - med))


def band_height(ssm, window):
    inside = (ssm.axis >= window[0]) & (ssm.axis <= window[1])
    return ssm.spectra[:, inside].max(axis=1)


def scaled_by_peak(ssm, window):
    """Divide every spectrum by the height of its band inside ``window`` (no minimum shift)."""
    scale = np.array([peak_height(ssm.axis, y, window, refine=2) for y in ssm.spectra])
    return ssm.replace(spectra=ssm.spectra / scale[:, None])


def full(values, rows, n):
    out = np.full(n, np.nan)
    out[rows] = values
    return out


def band_peak_and_fwhm(comp, step=2e-4):
    """Position and FWHM of the maximum of the summed fitted bands, on a fine grid."""
    grid = np.arange(comp.axis[0], comp.axis[-1], step)
    curves = np.where(comp.active_mask[:, :, None], comp.components(axis=grid), 0.0).sum(axis=1)
    k = np.argmax(curves, axis=1)
    half = curves >= 0.5 * curves[np.arange(curves.shape[0]), k][:, None]
    first, last = np.argmax(half, axis=1), grid.size - 1 - np.argmax(half[:, ::-1], axis=1)
    return grid[k], grid[last] - grid[first]


def raman_analysis(cfg):
    raw = despike_whitaker_hayes(StandardSpectrumMap.load(cfg.data_dir / "raman_map.npz"), 7.0, 3)
    sub, background = subtract_baseline(raw, "anchors", anchors=cfg.raman_anchors, anchor_degree=1)   # stage 1
    cand = np.flatnonzero(robust_above(band_height(sub, cfg.a1g_window), cfg.candidate_sd))
    data = scaled_by_peak(sub.select(cand), cfg.a1g_window)
    fit_cfg = FitConfig(RAMAN_PEAKS, RAMAN_STAGES, fit_baseline=False, x_fit_range=cfg.raman_window)  # stage 2
    opts = MapFitOptions(normalize_01=False, skip_low_signal=False, devices=(cfg.device,), batch_size=cfg.batch_size)
    r = fit_map(data, fit_cfg, opts, initial_guess=InitialGuessOptions("all")).result
    r.save(cfg.out_dir / "raman__FitSpectrumMap_v4.npz")
    e2g, a1g = r.peak_index("E2g"), r.peak_index("A1g")
    sep = r.pos[:, a1g] - r.pos[:, e2g]
    mono = ~robust_above(sep, cfg.max_separation_sd)          # leaves out a small few-layer island
    fwhm, area = r.peak_fwhms(), r.peak_areas()
    n = raw.n_spectra
    maps = {"E$_{2g}$ position (cm$^{-1}$)": r.pos[:, e2g], "A$_{1g}$ position (cm$^{-1}$)": r.pos[:, a1g],
            "A$_{1g}$ - E$_{2g}$ separation (cm$^{-1}$)": sep, "E$_{2g}$ FWHM (cm$^{-1}$)": fwhm[:, e2g],
            "A$_{1g}$ FWHM (cm$^{-1}$)": fwhm[:, a1g], "area ratio A$_{1g}$ / E$_{2g}$": area[:, a1g] / area[:, e2g]}
    maps = {k: full(np.where(mono, v, np.nan), cand, n) for k, v in maps.items()}
    stats = {"candidates_fitted": int(cand.size), "monolayer_pixels": int(mono.sum()),
             **{f"median {k}": float(np.nanmedian(v)) for k, v in maps.items()}}
    example = int(np.argmin(np.where(mono, np.abs(sep - np.median(sep[mono])), np.inf)))   # a typical pixel
    return {"raw": raw, "background": background, "cand": cand, "fit": r, "maps": maps, "stats": stats,
            "example": example}


def pl_analysis(cfg):
    raw = despike_whitaker_hayes(StandardSpectrumMap.load(cfg.data_dir / "pl_map.npz"), 7.0, 3)
    lam = raman_shift_to_wavelength(raw, cfg.excitation_nm, jacobian=False)   # counts per detector pixel
    step = np.diff(lam.axis)
    if np.ptp(step) / step.mean() > 0.01:
        raise RuntimeError("detector pixels are not uniform in wavelength; the lambda^2 Jacobian would not apply")
    e = wavelength_to_energy(lam, "eV", jacobian=True)                       # density per unit energy
    line = anchor_baseline(e.spectra, e.axis, [(1.63, 1.66), (2.05, 2.10)], degree=1)
    cand_mask = robust_above(band_height(e.replace(spectra=e.spectra - line), cfg.a_band_ev), cfg.candidate_sd)
    _, _, ix, iy = raster_indices(e.xy)
    near = ndimage.binary_dilation(to_image(cand_mask.astype(float), e.xy)[0] > 0, iterations=cfg.offflake_margin_px)
    offflake = ~near[iy, ix]
    reference = savgol_filter(e.spectra[offflake].mean(axis=0), cfg.reference_smooth_window, 3)
    cand = np.flatnonzero(cand_mask)
    sub, background = subtract_baseline(e.select(cand), "anchors", anchors=(cfg.pl_anchor_ev,), anchor_degree=None,
                                        anchor_reference=reference)                     # stage 1
    data = scaled_by_peak(sub, cfg.a_band_ev)
    e_bad = 1239.841984 / lam.axis[cfg.interpolated_column]
    base_cfg = FitConfig((A0,), PL_STAGES, fit_baseline=False, x_fit_range=cfg.pl_window_ev,
                         downweight_x_ranges=((e_bad - 1e-4, e_bad + 1e-4),), downweight_factor=0.0)
    opts = MapFitOptions(normalize_01=False, skip_low_signal=False, devices=(cfg.device,), batch_size=cfg.batch_size)
    fits = fit_variants(data, PL_VARIANTS, base_cfg, opts, initial_guess=InitialGuessOptions("all"))   # stage 2
    results = {k: f.result for k, f in fits.items()}
    sel = select_variants(results)
    comp = composite_fit(results, sel, input_peaks_of(fits))
    comp.save(cfg.out_dir / "pl__FitSpectrumMap_v4_composite.npz")
    peak, width = band_peak_and_fwhm(comp)
    area = np.nan_to_num(comp.area)
    i0, im = comp.peak_index("A0"), comp.peak_index("A-")
    trion_weight = area[:, im] / (area[:, i0] + area[:, im])
    n = raw.n_spectra
    has_trion = comp.active_mask[:, im]
    maps = {"PL peak energy (eV)": peak, "PL FWHM (meV)": 1e3 * width, "A$^0$ centre (eV)": comp.pos[:, i0],
            "A$^-$ (trion) centre (eV)": np.where(has_trion, comp.pos[:, im], np.nan),
            "trion weight A$^-$/(A$^0$ + A$^-$)": trion_weight}
    maps = {k: full(v, cand, n) for k, v in maps.items()}
    stats = {"candidates_fitted": int(cand.size), "offflake_pixels": int(offflake.sum()),
             "bic_choice_fraction": {nm: float(np.mean(sel.variant_id == v)) for v, nm in enumerate(sel.names)},
             "median PL peak energy (eV)": float(np.median(peak)), "median PL FWHM (meV)": float(1e3 * np.median(width)),
             "median A0 centre (eV)": float(np.median(comp.pos[:, i0])),
             "median A- centre (eV), where selected": float(np.median(comp.pos[has_trion, im])),
             "median trion weight, where selected": float(np.median(trion_weight[has_trion]))}
    example = int(np.argmin(np.abs(peak - np.median(peak)) + 10.0 * ~has_trion))
    return {"energy_map": e, "background": background, "cand": cand, "comp": comp, "sel": sel, "maps": maps,
            "stats": stats, "example": example, "xy": raw.xy}


def figure(maps, xy, panels, title, path):
    """``panels``: callables drawing into the first axes; the maps fill the rest."""
    n = len(panels) + len(maps)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.0 * nrows), squeeze=False)
    for ax, draw in zip(axes.flat, panels):
        draw(ax)
    for ax, (label, values) in zip(list(axes.flat)[len(panels):], maps.items()):
        plot_map(ax, xy, values, cmap="viridis", percentiles=(2, 98))
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("x ($\\mu$m)")
        ax.set_ylabel("y ($\\mu$m)")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def select_full(sel, rows, n):
    """The selection on the full raster (pixels not fitted are invalid)."""
    vid = np.full(n, -1)
    vid[rows] = sel.variant_id
    return replace(sel, variant_id=vid)


def main(cfg: AppConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    ra = raman_analysis(cfg)
    pa = pl_analysis(cfg)
    (cfg.out_dir / "summary.json").write_text(json.dumps({"raman": ra["stats"], "pl": pa["stats"]}, indent=2))
    print(json.dumps({"raman": ra["stats"], "pl": pa["stats"]}, indent=2))

    k_r = ra["example"]
    row_r = ra["cand"][k_r]

    def raman_background(ax):
        x = ra["raw"].axis
        ax.plot(x, ra["raw"].spectra[row_r], color="0.4", lw=0.7, label="despiked data")
        ax.plot(x, ra["background"][row_r], color="C3", lw=1.2, label="background (stage 1)")
        for lo, hi in cfg.raman_anchors:
            ax.axvspan(lo, hi, color="C2", alpha=0.18, lw=0)
        ax.set_xlim(280, 530)
        ax.set_title("stage 1: straight background through the anchors (green)", fontsize=9)
        ax.set_xlabel("Raman shift (cm$^{-1}$)")
        ax.legend(fontsize=7, frameon=False)

    def raman_fit(ax):
        plot_fit(ax, ra["fit"], k_r, title="stage 2: band fit, no baseline term")
        ax.set_xlabel("Raman shift (cm$^{-1}$)")
        ax.set_ylabel("(data - background) / A$_{1g}$ height")

    figure(ra["maps"], ra["raw"].xy, [raman_background, raman_fit],
           "MoS$_2$ Raman map: background, band fit and monolayer parameter maps", cfg.out_dir / "mos2_raman.png")

    k_p = pa["example"]
    row_p = pa["cand"][k_p]

    def pl_background(ax):
        em = pa["energy_map"]
        ax.plot(em.axis, em.spectra[row_p], color="0.4", lw=0.7, label="despiked data (per eV)")
        ax.plot(em.axis, pa["background"][k_p], color="C3", lw=1.2, label="scaled substrate emission")
        ax.axvspan(*cfg.pl_anchor_ev, color="C2", alpha=0.18, lw=0)
        ax.set_xlim(1.6, 2.25)
        ax.set_title("stage 1: off-flake reference scaled in the anchor (green)", fontsize=9)
        ax.set_xlabel("photon energy (eV)")
        ax.legend(fontsize=7, frameon=False)

    def pl_fit(ax):
        label = {"A0": "A$^0$", "A-": "A$^-$ (trion)", "B": "B"}
        plot_fit(ax, pa["comp"], k_p, title="stage 2: band fit (BIC-selected model)",
                 names=[label[n] for n in pa["comp"].peak_names])
        ax.set_xlabel("photon energy (eV)")
        ax.set_ylabel("(data - background) / A-band height")

    def pl_variants(ax):
        plot_variant_map(ax, select_full(pa["sel"], pa["cand"], len(pa["xy"])), pa["xy"])

    figure(pa["maps"], pa["xy"], [pl_background, pl_fit, pl_variants],
           "MoS$_2$ PL map: background, band fit, model choice and band maps", cfg.out_dir / "mos2_pl.png")


if __name__ == "__main__":
    main(CONFIG)
