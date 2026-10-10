"""Figure data_format.png: layout of StandardSpectrumMap and FitSpectrumMap."""
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _common import BLUE, GREEN, ORANGE, PURPLE, RED, box, save


@dataclass(frozen=True)
class AppConfig:
    name: str = "data_format.png"
    dpi: int = 150


def main(cfg: AppConfig) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax in (a, b):
        ax.set_axis_off()
    # --- StandardSpectrumMap
    a.set_title("StandardSpectrumMap (input of every stage)", fontsize=11)
    a.add_patch(Rectangle((0.22, 0.18), 0.5, 0.55, fc=BLUE, alpha=0.15, ec=BLUE, transform=a.transAxes))
    a.text(0.47, 0.455, "spectra\n(N, M) float64\nrow i = spectrum at xy[i]\ncolumn j = intensity at axis[j]",
           ha="center", va="center", transform=a.transAxes)
    a.add_patch(Rectangle((0.22, 0.76), 0.5, 0.07, fc=ORANGE, alpha=0.25, ec=ORANGE, transform=a.transAxes))
    a.text(0.47, 0.795, "axis (M,): strictly increasing", ha="center", va="center", transform=a.transAxes)
    a.add_patch(Rectangle((0.06, 0.18), 0.13, 0.55, fc=GREEN, alpha=0.2, ec=GREEN, transform=a.transAxes))
    a.text(0.125, 0.455, "xy\n(N, 2)\nstage\nx, y", ha="center", va="center", transform=a.transAxes)
    a.text(0.47, 0.88, 'unit: "cm^-1", "nm", "eV", ...', ha="center", transform=a.transAxes, color="0.3")
    a.text(0.76, 0.6, "NPZ keys\n  spectra\n  xy\n  axis\n  unit (optional)", va="center", transform=a.transAxes,
           family="monospace", fontsize=9)
    a.text(0.02, 0.04, "Readers sort the axis and average columns that share an axis value; every value must be\n"
           "finite. Coordinates are kept as exported (no origin or orientation is assumed).",
           transform=a.transAxes, fontsize=8, color="0.25")
    # --- FitSpectrumMap
    b.set_title("FitSpectrumMap (output of a fit, schema FitSpectrumMap.v4)", fontsize=11)
    rows = [("axis (L,), unit", "fit window, in the fit space", ORANGE),
            ("xy (N, 2), spectra_original (N, L)", "the data that were fitted", GREEN),
            ("peak_types (P,), metadata_json", "peak names, stage plan, options", PURPLE),
            ("params_pos / _width / _height / _eta (N, P)", "one column per peak", BLUE),
            ("params_width_g, params_q (N, P)", "TCH Gaussian FWHM, Fano asymmetry", BLUE),
            ("params_base (N,), recon (N, L)", "constant baseline, reconstruction", BLUE),
            ("params_area (N, P), loss_final (N,)", "analytic area, final objective", BLUE),
            ("valid_mask (N,)", "False: not fitted (parameters NaN)", RED),
            ("composite_active_mask (N, P)", "composite results only", RED)]
    for k, (key, what, color) in enumerate(rows):
        y = 0.86 - k * 0.095
        box(b, 0.02, y, 0.52, 0.075, key, color, fontsize=8.5)
        b.text(0.57, y + 0.037, what, va="center", transform=b.transAxes, fontsize=8.5)
    fig.tight_layout()
    save(fig, cfg.name, cfg.dpi)


if __name__ == "__main__":
    main(AppConfig())
