"""Publication figure style for the shot-quality manuscript.

This house style uses compact Helvetica/Arial typography, dark titles,
restrained grey gridlines, colour-blind-aware blue/vermillion/green accents,
and multi-format exports suitable for journal submission.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

NAV = "#1c3557"
BLU = "#2c5f8a"
LBL = "#d4e8f7"
GRN = "#2e7d4f"
AMB = "#c8920a"
RED = "#b83232"
LRD = "#f5e2e2"
ORG = "#e07020"
VERMILLION = "#d55e00"
SKY = "#56b4e9"
PURPLE = "#cc79a7"

#: Fixed categorical order. Assign in sequence; never recycle.
CATEGORICAL = [BLU, VERMILLION, GRN, AMB, PURPLE, SKY]

#: Marker shapes provide the secondary encoding that hue alone must not carry.
MARKERS = ["o", "s", "^", "D", "v", "P"]

#: Single-hue sequential ramp for magnitude (shot density, probability).
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "shotq_seq",
    ["#f7fbff", LBL, "#9ecae1", BLU, NAV],
)

#: Two-pole diverging ramp with a neutral grey midpoint, for z-scored centroids.
DIVERGING = LinearSegmentedColormap.from_list(
    "shotq_div",
    [BLU, "#8bbbd8", "#f3f3f3", "#eaa06a", VERMILLION],
)

INK = "#262626"
INK_MUTED = "#555555"
GRID = "#ebebeb"
SPINE = "#c4c4c4"
SURFACE = "#ffffff"

# Column widths in inches.
W_SINGLE = 90 / 25.4
W_DOUBLE = 190 / 25.4


def apply() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.14,
            "savefig.facecolor": SURFACE,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.titlelocation": "center",
            "axes.labelsize": 8,
            "axes.labelpad": 9,
            "axes.titlepad": 10,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.major.pad": 4.5,
            "ytick.major.pad": 4.5,
            "legend.fontsize": 7.5,
            "axes.edgecolor": SPINE,
            "axes.linewidth": 0.7,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "grid.alpha": 1.0,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "lines.markersize": 4,
            "patch.linewidth": 0.6,
        }
    )


def save(fig, name: str, figdir=None) -> None:
    """Write a figure as SVG/PDF vectors plus PNG preview."""
    from common import FIGURES

    figdir = figdir or FIGURES
    for ext in ("svg", "pdf", "png"):
        fig.savefig(figdir / f"{name}.{ext}")
    print(f"  [fig  ] outputs/figures/{name}.svg / .pdf / .png")
    plt.close(fig)
