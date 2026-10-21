"""Figures for the metrics that judge where cell boundaries were drawn."""

from __future__ import annotations

import numpy as np
from _draw import (
    BLUE,
    GREEN,
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


def _outline(ax, path: MplPath, scale: float, centre, *, color, lw=1.6, ls="-"):
    """Draw a scaled copy of a cell outline, as a mask boundary."""
    verts = (np.asarray(path.vertices) - centre) * scale + centre
    ax.add_patch(PathPatch(MplPath(verts, path.codes), facecolor="none", edgecolor=color, lw=lw, ls=ls, zorder=6))


def spillover():
    """A marker positive in one cell should be absent from its neighbours."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    field(ax, (0, 10, 0, 9))

    places = [(2.4, 5.4), (5.2, 5.7), (7.8, 5.0), (2.0, 2.2), (4.9, 2.4), (7.9, 2.0)]
    roles = ["positive", "touching", "far", "touching", "far", "far"]
    fills = {"positive": MAGENTA, "touching": PINK, "far": PALE_GREY}
    for i, ((x, y), role) in enumerate(zip(places, roles, strict=True)):
        cell(ax, x, y, 1.15, fill=fills[role], seed=11 + i * 5, angle=i * 0.8, nucleus_offset=(0.1, -0.1))
    label(ax, 2.4, 7.2, "positive", size=8, weight="bold")
    label(ax, 5.2, 8.2, "negative, touching it", size=8)
    label(ax, 7.8, 6.9, "negative, far", size=8)
    panel(ax, "One positive cell among its neighbours")

    ax2 = axes[1]
    ax2.bar(["touching", "far"], [41.0, 38.0], color=[PINK, PALE_GREY], width=0.5, edgecolor=INK, lw=0.6)
    ax2.set_ylim(0, 56)
    ax2.set_ylabel("mean intensity of negative cells")
    label(ax2, 0.5, 50, "ratio = 1.08\n(1.0 would be no bleed)", size=8)
    panel(ax2, "Does signal cross the boundary?")
    headline(fig, "spillover_ratio: an over-expanded mask collects its neighbour's signal")
    save(fig, "spillover")


def exclusive_coexpression():
    """Markers that no single cell should carry at once."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for ax, merged, title in (
        (axes[0], False, "Cells resolved correctly"),
        (axes[1], True, "One mask spanning two cells"),
    ):
        field(ax, (0, 10, 0, 7))
        spots = [(2.2, 4.8), (5.0, 5.2), (7.8, 4.6), (2.4, 1.9), (7.6, 1.8)]
        for i, (x, y) in enumerate(spots):
            cell(ax, x, y, 1.1, fill=TEAL if i % 2 else GREEN, seed=31 + i * 4, angle=i * 0.9)
        if merged:
            # A single object covering an immune and an epithelial cell: it is
            # positive for both markers, which no real cell should be.
            cell(ax, 5.0, 2.0, 1.75, fill=TEAL, seed=77, angle=0.4, squash=0.72, nucleus=False, zorder=2)
            cell(ax, 5.55, 2.0, 1.1, fill=GREEN, seed=78, angle=0.2, alpha=0.75, nucleus=False, zorder=3)
            cell(ax, 4.45, 2.05, 0.42, fill=INK, seed=79, nucleus=False, zorder=4)
            cell(ax, 5.7, 1.95, 0.42, fill=INK, seed=80, nucleus=False, zorder=4)
            label(ax, 5.0, 0.3, "positive for both", size=8, weight="bold")
        else:
            cell(ax, 5.0, 2.0, 1.1, fill=GREEN, seed=41, angle=0.3)
        panel(ax, title)
    headline(
        fig,
        "mutually_exclusive_coexpression_ratio: cells positive for two incompatible markers",
    )
    save(fig, "exclusive-coexpression")


def nuclear_containment():
    """Growing a mask captures more signal, and more of the neighbour's."""
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.9), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    places = [(2.3, 4.7), (5.4, 5.0), (8.0, 4.4), (2.6, 1.8), (5.6, 1.7), (8.2, 1.6)]
    for ax, scale, title, colour in (
        (axes[0], 1.18, "Conservative masks", BLUE),
        (axes[1], 1.95, "Over-expanded masks", MAGENTA),
    ):
        field(ax, (0, 10, 0, 7))
        for i, (x, y) in enumerate(places):
            path = cell(ax, x, y, 1.0, fill=TEAL if i % 2 else GREEN, seed=51 + i * 6, angle=i * 0.6)
            _outline(ax, path, scale, np.array([x, y]), color=colour, ls="--")
        panel(ax, title)

    axes[2].bar(["conservative", "expanded"], [0.72, 0.97], color=[BLUE, MAGENTA], width=0.5, edgecolor=INK, lw=0.6)
    axes[2].set_ylim(0, 1.1)
    axes[2].set_ylabel("nuclear signal inside a mask")
    panel(axes[2], "Bigger always scores higher")
    headline(
        fig,
        "nuclear_signal_containment rises with mask size, so it needs spillover_ratio beside it",
    )
    save(fig, "nuclear-containment")


def object_matching():
    """Agreement against reference masks, when you have them."""
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.9), gridspec_kw={"width_ratios": [1, 1, 0.85]})
    places = [(2.4, 4.7), (5.6, 5.0), (8.0, 4.3), (2.6, 1.8), (5.8, 1.7)]
    for ax, scale, title, colour in (
        (axes[0], 1.08, "Predicted ≈ reference", BLUE),
        (axes[1], 1.9, "Predicted ≫ reference", MAGENTA),
    ):
        field(ax, (0, 10, 0, 7))
        for i, (x, y) in enumerate(places):
            path = cell(ax, x, y, 1.0, fill=TEAL if i % 2 else GREEN, seed=61 + i * 3, angle=i * 0.5)
            _outline(ax, path, scale, np.array([x, y]), color=colour)
        panel(ax, title)
    label(axes[0], 5.0, 0.3, "outline = what the segmentation claims", size=7.5)

    x = np.arange(2)
    axes[2].bar(x - 0.18, [0.96, 0.36], width=0.34, color=BLUE, edgecolor=INK, lw=0.6, label="precision")
    axes[2].bar(x + 0.18, [0.98, 0.99], width=0.34, color=GREEN, edgecolor=INK, lw=0.6, label="recall")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(["good", "expanded"])
    axes[2].set_ylim(0, 1.18)
    axes[2].legend(fontsize=7.5, loc="upper center", ncols=2)
    panel(axes[2], "Precision falls; recall does not")
    headline(fig, "Against reference masks, the two failure modes separate cleanly")
    save(fig, "object-matching")


def replicate_concordance():
    """The same nuclei, imaged twice."""
    rng = np.random.default_rng(6)
    truth = rng.lognormal(3.0, 0.5, 260)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True, sharey=True)
    for ax, noise, title, colour in (
        (axes[0], 0.05, "Masks follow their nuclei", TEAL),
        (axes[1], 0.45, "Masks stray off them", PINK),
    ):
        second = np.abs(truth * rng.normal(1.0, noise, truth.size))
        ax.scatter(np.log1p(truth), np.log1p(second), s=14, color=colour, alpha=0.85, linewidths=0.3, edgecolors=INK)
        ax.plot([1.5, 5.2], [1.5, 5.2], color=INK, lw=0.8, ls="--")
        ax.set_xlabel("log1p(DNA, cycle 1)")
        panel(ax, title)
    axes[0].set_ylabel("log1p(DNA, cycle 2)")
    headline(fig, "replicate_correlation: a restained nucleus should read the same twice")
    save(fig, "replicate-concordance")


def tile_cv():
    """Illumination that varies across the slide."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    places = [(x, y) for y in (1.6, 3.6, 5.6) for x in (1.8, 4.0, 6.2, 8.4)]
    for ax, shade, title in (
        (axes[0], False, "Evenly illuminated"),
        (axes[1], True, "A gradient across the field"),
    ):
        field(ax, (0, 10, 0, 7.4))
        for i, (x, y) in enumerate(places):
            alpha = 1.0 if not shade else float(np.clip(0.3 + x / 11.0, 0.25, 1.0))
            cell(ax, x, y, 0.85, fill=TEAL, seed=91 + i * 3, angle=i * 0.4, alpha=alpha)
        for gx in (3.0, 5.2, 7.4):
            ax.plot([gx, gx], [0.2, 7.2], color=GREY, lw=0.7, ls=":", zorder=5)
        panel(ax, title)
    label(axes[1], 5.0, -0.25, "tiles are compared with each other", size=7.5)
    headline(fig, "tile_cv: does brightness depend on where a cell sits?")
    save(fig, "tile-cv")


FIGURES = [
    spillover,
    exclusive_coexpression,
    nuclear_containment,
    object_matching,
    replicate_concordance,
    tile_cv,
]
