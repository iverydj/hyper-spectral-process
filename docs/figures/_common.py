"""Shared helpers of the documentation figure scripts (run each ``make_*.py`` from any directory)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent
GREY, BLUE, ORANGE, GREEN, RED, PURPLE = "0.45", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "legend.fontsize": 8})


def gauss(x, c, w, h=1.0):
    return h * np.exp(-4.0 * np.log(2.0) * ((x - c) / w) ** 2)


def lorentz(x, c, w, h=1.0):
    return h / (1.0 + 4.0 * ((x - c) / w) ** 2)


def save(fig, name: str, dpi: int) -> Path:
    path = FIG_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    return path


def box(ax, x, y, w, h, text, color, fontsize=9, alpha=0.18):
    """A rounded box with centred text in axes coordinates."""
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.01", fc=color, ec=color,
                                alpha=alpha, lw=1.2, transform=ax.transAxes))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, transform=ax.transAxes)


def arrow(ax, x0, y0, x1, y1, color="0.3"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2))
