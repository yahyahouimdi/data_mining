"""Shared matplotlib style for the report figures."""

from __future__ import annotations

import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Categorical slots in fixed order: class 0 / first series, class 1 / second series.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SEIZURE = "#d03b3b"
DIVERGING = ["#eb6834", "#f0efec", "#2a78d6"]


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": SECONDARY, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": SECONDARY, "ytick.labelcolor": SECONDARY,
        "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "regular", "legend.frameon": False,
        "axes.prop_cycle": plt.cycler(color=SERIES),
    })
