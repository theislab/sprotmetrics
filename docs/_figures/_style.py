"""Shared styling for the metric figures.

Each figure explains one *mechanism*, not one metric: several metrics share an
idea (positivity by Otsu split, say) and share the picture that explains it.

Figures are committed as SVG under ``docs/_static/metrics`` so the docs build
needs no plotting stack. Regenerate with::

    python docs/_figures/make_figures.py
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = pathlib.Path(__file__).resolve().parent.parent / "_static" / "metrics"

# Palette shared with the lab's diagrams; see _draw for the full set.
from _draw import GREEN, GREY, INK, MAGENTA, PINK, TEAL  # noqa: E402

GOOD = TEAL
BAD = PINK
NEUTRAL = GREY
ACCENT = GREEN
HIGHLIGHT = MAGENTA

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 110,
        # Keeps panel titles from running into the neighbouring axes.
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.w_pad": 0.08,
    }
)


def save(fig, name: str) -> None:
    """Write one figure as SVG, trimmed."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("  wrote", name + ".svg")


def panel(ax, title: str, *, hide_ticks: bool = False) -> None:
    """Title a panel, optionally dropping the ticks for image-like axes."""
    ax.set_title(title, loc="left")
    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)


def headline(fig, text: str) -> None:
    """Figure-level title, left-aligned and placed by the constrained layout.

    Passing an explicit ``y`` here fights the layout engine and lands the
    headline on top of the first panel title, so it is left to position itself.
    """
    fig.suptitle(text, x=0.01, ha="left")
