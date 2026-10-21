"""Figures for marker distribution, normalization and downstream structure."""

from __future__ import annotations

import numpy as np
from _draw import BLUE, GREEN, INK, PALE_GREY, PINK, TEAL, cell, field
from _style import headline, panel, plt, save


def marker_coverage():
    """How many cells carry a marker at all."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    places = [(x, y) for y in (1.6, 3.7, 5.8) for x in (1.8, 4.0, 6.2, 8.4)]
    for ax, fraction, title in (
        (axes[0], 0.83, "Widely detected marker"),
        (axes[1], 0.25, "Barely detected marker"),
    ):
        field(ax, (0, 10, 0, 7.4))
        n_on = round(fraction * len(places))
        for i, (x, y) in enumerate(places):
            cell(ax, x, y, 0.82, fill=BLUE if i < n_on else PALE_GREY, seed=101 + i * 3, angle=i * 0.45)
        panel(ax, f"{title} ({fraction:.0%})")
    headline(fig, "marker_coverage: the fraction of cells above the detection threshold")
    save(fig, "marker-coverage")


def area_dependence():
    """Expression that tracks cell size rather than biology."""
    rng = np.random.default_rng(12)
    area = rng.lognormal(5.0, 0.45, 260)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharex=True)
    for ax, scales, title, colour in (
        (axes[0], False, "Mean aggregation", TEAL),
        (axes[1], True, "Sum aggregation", PINK),
    ):
        value = area * rng.normal(1.0, 0.25, area.size) if scales else rng.lognormal(3.0, 0.4, area.size)
        ax.scatter(area, value, s=14, color=colour, alpha=0.85, linewidths=0.3, edgecolors=INK)
        ax.set_xlabel("cell area (px)")
        panel(ax, f"{title}  (R² {'0.91' if scales else '0.02'})")
    axes[0].set_ylabel("marker intensity")
    headline(
        fig,
        "variance_explained_by_covariate: is a marker reporting biology, or cell size?",
    )
    save(fig, "area-dependence")


def variance_evenness():
    """Whether a few markers own all the variance."""
    even = np.array([0.14, 0.13, 0.13, 0.12, 0.12, 0.12, 0.12, 0.12])
    skewed = np.array([0.62, 0.21, 0.07, 0.04, 0.02, 0.02, 0.01, 0.01])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    for ax, shares, title, colour, value in (
        (axes[0], even, "Variance spread across the panel", TEAL, 0.99),
        (axes[1], skewed, "Two markers dominate", PINK, 0.58),
    ):
        ax.bar(range(len(shares)), shares, color=colour, edgecolor=INK, lw=0.6, width=0.7)
        ax.set_xticks(range(len(shares)))
        ax.set_xticklabels([f"m{i + 1}" for i in range(len(shares))], fontsize=7)
        ax.set_xlabel("marker")
        panel(ax, f"{title}  (evenness {value})")
    axes[0].set_ylabel("share of total variance")
    headline(fig, "marker_variance_evenness: does any embedding see the whole panel?")
    save(fig, "variance-evenness")


def normalization_spread():
    """What a per-cell normalization is for."""
    rng = np.random.default_rng(14)
    raw = rng.lognormal(3.2, 0.75, 400)
    normalized = np.arcsinh(raw / np.median(raw)) * np.median(raw) / 2.2
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for ax, data, title, colour, cv in (
        (axes[0], raw, "Before", PINK, 0.82),
        (axes[1], normalized, "After", TEAL, 0.31),
    ):
        ax.hist(data, bins=40, color=colour, edgecolor=INK, lw=0.4)
        ax.set_xlabel("per-cell total intensity")
        panel(ax, f"{title}  (CV {cv})")
    axes[0].set_ylabel("cells")
    headline(
        fig,
        "total_cv_ratio: the spread in per-cell totals should shrink, without flattening the biology",
    )
    save(fig, "normalization-spread")


def cluster_separability():
    """Whether cell types survive preprocessing."""
    rng = np.random.default_rng(15)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True, sharey=True)
    for ax, spread, title, score in (
        (axes[0], 0.55, "Types stay separable", 0.61),
        (axes[1], 1.9, "Types blur together", 0.07),
    ):
        for centre, colour in (((-2, 0), TEAL), ((2, 0.6), PINK), ((0, 2.6), GREEN)):
            pts = rng.normal(centre, spread, size=(70, 2))
            ax.scatter(pts[:, 0], pts[:, 1], s=13, color=colour, alpha=0.85, linewidths=0.3, edgecolors=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        panel(ax, f"{title}  (silhouette {score})")
    headline(fig, "silhouette_cell_type: do known populations still sit apart?")
    save(fig, "cluster-separability")


def marker_gini():
    """Concentration: a lineage marker against a housekeeper."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    places = [(x, y) for y in (1.6, 3.7, 5.8) for x in (1.8, 4.0, 6.2, 8.4)]
    rng = np.random.default_rng(16)
    for ax, concentrated, title, value in (
        (axes[0], False, "Housekeeper: everywhere", 0.12),
        (axes[1], True, "Lineage marker: a few cells", 0.74),
    ):
        field(ax, (0, 10, 0, 7.4))
        for i, (x, y) in enumerate(places):
            if concentrated:
                alpha = 1.0 if i in (1, 5, 9) else 0.18
            else:
                alpha = float(rng.uniform(0.7, 1.0))
            cell(ax, x, y, 0.82, fill=GREEN, seed=131 + i * 3, angle=i * 0.4, alpha=alpha)
        panel(ax, f"{title}  (gini {value})")
    headline(fig, "marker_gini: descriptive, not directional — both of these are correct")
    save(fig, "marker-gini")


FIGURES = [
    marker_coverage,
    area_dependence,
    variance_evenness,
    normalization_spread,
    cluster_separability,
    marker_gini,
]
