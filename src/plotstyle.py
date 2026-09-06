"""Publication figure style for the shot-quality manuscript.

Figures target Science and Medicine in Football: single-column width 90 mm,
double-column 190 mm, 300 dpi (600 dpi for line art), sans-serif labels at
7-9 pt, and colour that survives both greyscale printing and colour-vision
deficiency.

The categorical order below is fixed and never cycled. It was checked with the
six-check palette validator (light surface): lightness band PASS, chroma floor
PASS, CVD adjacent separation PASS (worst pair dE 9.6 deutan), normal-vision
floor PASS (worst pair dE 20.0). Colour is always paired with a second encoding
- a direct label, a legend entry, or a marker shape - so identity never rests on
hue alone.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

#: Fixed categorical order (validated). Assign in sequence; never recycle.
CATEGORICAL = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#E69F00",  # orange
    "#CC79A7",  # purple
    "#56B4E9",  # sky
]

#: Marker shapes provide the secondary encoding that hue alone must not carry.
MARKERS = ["o", "s", "^", "D", "v", "P"]

#: Single-hue sequential ramp for magnitude (shot density, probability).
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "shotq_seq",
    ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
)

#: Two-pole diverging ramp with a neutral grey midpoint, for z-scored centroids.
DIVERGING = LinearSegmentedColormap.from_list(
    "shotq_div",
    ["#0072B2", "#7fb8d9", "#f0f0f0", "#eaa06a", "#D55E00"],
)

INK = "#1a1a1a"
INK_MUTED = "#6b6b6b"
GRID = "#dcdcdc"
SURFACE = "#ffffff"

# Column widths in inches.
W_SINGLE = 90 / 25.4
W_DOUBLE = 190 / 25.4


def apply() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.facecolor": SURFACE,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.edgecolor": INK_MUTED,
            "axes.linewidth": 0.7,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 1.0,
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
    """Write a figure as both PDF (vector, for submission) and PNG (preview)."""
    from common import FIGURES

    figdir = figdir or FIGURES
    for ext in ("pdf", "png"):
        fig.savefig(figdir / f"{name}.{ext}")
    print(f"  [fig  ] outputs/figures/{name}.pdf / .png")
    plt.close(fig)
