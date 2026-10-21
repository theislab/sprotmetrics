"""Figures for positivity, local background and the signal integral."""

from __future__ import annotations

import numpy as np
from _draw import (
    BLUE,
    GREY,
    INK,
    MAGENTA,
    PALE_GREY,
    PINK,
    TEAL,
    cell,
    field,
    label,
)
from _style import headline, panel, plt, save
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath


def _ring(ax, path, scale, centre, *, color):
    """The band between a cell outline and a grown copy of it."""
    inner = np.asarray(path.vertices)
    outer = (inner - centre) * scale + centre
    verts = np.vstack([outer, outer[:1], inner[::-1], inner[-1:]])
    codes = list(path.codes) + [MplPath.CLOSEPOLY] + [MplPath.LINETO] * len(inner) + [MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=0.9, zorder=1))


def otsu_positivity():
    """Splitting one marker into negative and positive cells."""
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))

    ax = axes[0]
    field(ax, (0, 10, 0, 7))
    places = [(x, y) for y in (1.7, 4.0, 5.9) for x in (1.8, 4.1, 6.4, 8.6)]
    positive = {0, 3, 5, 8, 10}
    for i, (x, y) in enumerate(places):
        cell(ax, x, y, 0.85, fill=MAGENTA if i in positive else PALE_GREY, seed=7 + i * 5, angle=i * 0.5)
    label(ax, 5.0, -0.5, "positive cells in colour", size=8)
    panel(ax, "One marker, cell by cell")

    ax2 = axes[1]
    neg = rng.lognormal(1.2, 0.45, 900)
    pos = rng.lognormal(3.4, 0.35, 330)
    ax2.hist(np.log1p(np.concatenate([neg, pos])), bins=42, color=GREY, alpha=0.9)
    ax2.axvline(np.log1p(12.0), color=MAGENTA, lw=2)
    label(ax2, np.log1p(12.0) + 0.1, 95, "Otsu cut", size=8, ha="left", color=MAGENTA)
    ax2.set_xlabel("log1p(intensity)")
    ax2.set_ylabel("cells")
    panel(ax2, "The cut comes from the data")
    headline(fig, "Positivity is a data-driven split, not a fixed threshold")
    save(fig, "otsu-positivity")


def mask_and_ring():
    """Where the background comparison comes from."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    field(ax, (0, 10, 0, 7))
    for i, (x, y) in enumerate([(2.4, 4.8), (5.4, 5.1), (8.0, 4.4), (2.6, 1.9), (5.6, 1.8), (8.2, 1.7)]):
        path = cell(ax, x, y, 0.95, fill=TEAL, seed=21 + i * 4, angle=i * 0.6)
        _ring(ax, path, 1.5, np.array([x, y]), color=PINK)
    label(ax, 2.4, 6.5, "mask", size=8, color=INK)
    label(ax, 5.4, 6.9, "ring just outside it", size=8, color=MAGENTA)
    panel(ax, "Cells, and the band around them")

    axes[1].bar(["inside", "ring"], [172, 11], color=[TEAL, PINK], width=0.5, edgecolor=INK, lw=0.6)
    axes[1].set_ylabel("mean intensity")
    panel(axes[1], "Their ratio is the metric")
    headline(fig, "Background is sampled next to the cells, not from empty slide")
    save(fig, "mask-and-ring")


def rank_matched_curves():
    """The signal integral, as an area between two sorted curves."""
    rng = np.random.default_rng(3)
    background = np.sort(rng.lognormal(2.6, 0.35, 200))
    strong = np.sort(rng.lognormal(3.2, 0.55, 200))
    weak = np.sort(rng.lognormal(2.74, 0.4, 200))
    x = np.linspace(0, 100, 200)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), sharey=True)
    for ax, signal, title in (
        (axes[0], strong, "Marker well above background"),
        (axes[1], weak, "Marker barely above it"),
    ):
        ax.plot(x, signal, color=BLUE, lw=2, label="cells")
        ax.plot(x, background, color=MAGENTA, lw=2, ls="--", label="background")
        ax.fill_between(x, background, signal, where=signal >= background, color=TEAL, alpha=0.55)
        ax.set_xlabel("cells, sorted by intensity (%)")
        ax.set_ylim(0, 90)
        panel(ax, title)
    axes[0].set_ylabel("mean intensity")
    axes[0].legend(loc="upper left", fontsize=8)
    headline(fig, "rel_sni is the shaded area, scaled by how bright the cells are")
    save(fig, "rank-matched-curves")


def intracell_coverage():
    """Bright and extensive against bright and sparse."""
    rng = np.random.default_rng(4)
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.8), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for ax, punctate, title in (
        (axes[0], False, "Real staining"),
        (axes[1], True, "Punctate artefact"),
    ):
        field(ax, (0, 6, 0, 6))
        path = cell(ax, 3, 3, 2.3, fill=PALE_GREY if punctate else MAGENTA, seed=5, angle=0.3, nucleus=False)
        if punctate:
            for _ in range(22):
                px, py = rng.uniform(1.4, 4.6, 2)
                if (px - 3) ** 2 + (py - 3) ** 2 < 2.0**2:
                    ax.add_patch(plt.Circle((px, py), 0.18, color=MAGENTA, zorder=4))
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor=INK, lw=1.1, zorder=5))
        panel(ax, title)

    axes[2].bar(["real", "punctate"], [0.93, 0.17], color=[MAGENTA, PALE_GREY], width=0.5, edgecolor=INK, lw=0.6)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("fraction of own pixels positive")
    panel(axes[2], "Coverage separates them")
    headline(fig, "Both are bright on average; only one is stained throughout")
    save(fig, "intracell-coverage")


def signal_drop():
    """Pixels that were positive, aggregated down to nothing."""
    rng = np.random.default_rng(9)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    field(ax, (0, 7, 0, 6))
    path = cell(ax, 3.5, 3.2, 2.4, fill=PALE_GREY, seed=13, angle=0.2, nucleus=False)
    for _ in range(9):
        px, py = rng.uniform(1.8, 5.2, 2)
        if (px - 3.5) ** 2 + (py - 3.2) ** 2 < 1.9**2:
            ax.add_patch(plt.Circle((px, py), 0.2, color=MAGENTA, zorder=4))
    ax.add_patch(PathPatch(path, facecolor="none", edgecolor=INK, lw=1.1, zorder=5))
    label(ax, 3.5, 0.5, "a few bright pixels in a large mask", size=8)
    panel(ax, "What the pixels show")

    axes[1].bar(
        ["mean over\nthe mask", "positive\npixels"],
        [0.0, 9],
        color=[PALE_GREY, MAGENTA],
        width=0.5,
        edgecolor=INK,
        lw=0.6,
    )
    axes[1].set_ylabel("value after aggregation")
    label(axes[1], 0, 1.2, "rounds to zero", size=8, color=INK)
    panel(axes[1], "What the cell table records")
    headline(fig, "signal_drop_on_aggregation: signal that existed, then did not")
    save(fig, "signal-drop")


def nuclear_enrichment():
    """Where inside the cell a marker sits."""
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.8), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for ax, nuclear, title in (
        (axes[0], True, "Nuclear marker"),
        (axes[1], False, "Membrane marker"),
    ):
        field(ax, (0, 6, 0, 6))
        # Cell body, then the stained compartment, then the nucleus on top so it
        # stays visible in both panels.
        body = cell(ax, 3, 3, 2.3, fill=PALE_GREY, seed=17, angle=0.25, nucleus=False)
        if nuclear:
            cell(ax, 3, 3, 1.25, fill=MAGENTA, seed=18, nucleus=False, zorder=3)
        else:
            cell(ax, 3, 3, 2.3, fill=MAGENTA, seed=17, angle=0.25, nucleus=False, zorder=3)
            cell(ax, 3, 3, 1.72, fill=PALE_GREY, seed=17, angle=0.25, nucleus=False, zorder=4)
        cell(ax, 3, 3, 0.72, fill=INK, seed=18, nucleus=False, zorder=6)
        ax.add_patch(PathPatch(body, facecolor="none", edgecolor=INK, lw=1.2, zorder=7))
        panel(ax, title)
    label(axes[0], 3, 0.45, "stain over the nucleus", size=7.5)
    label(axes[1], 3, 0.45, "stain at the rim", size=7.5)

    axes[2].bar(["nuclear", "membrane"], [3.1, 0.4], color=[MAGENTA, BLUE], width=0.5, edgecolor=INK, lw=0.6)
    axes[2].axhline(1.0, color=INK, lw=0.8, ls="--")
    axes[2].set_ylabel("nucleus / cytoplasm")
    panel(axes[2], "Above 1 is nuclear")
    headline(fig, "nuclear_enrichment: which compartment holds the signal")
    save(fig, "nuclear-enrichment")


FIGURES = [
    otsu_positivity,
    mask_and_ring,
    rank_matched_curves,
    intracell_coverage,
    signal_drop,
    nuclear_enrichment,
]
