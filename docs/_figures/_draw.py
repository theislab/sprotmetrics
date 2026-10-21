"""A small drawing toolkit for the metric figures.

The figures are illustrations rather than plots: flat pastel cells with dark
nuclei on a cream field, matching the house style used in the lab's diagrams.
Quantitative panels sit alongside them in the same palette, so one figure can
show a mechanism and its consequence together.
"""

from __future__ import annotations

import numpy as np
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path

# Taken from the reference diagram so the figures sit alongside it.
CREAM = "#fdf9f0"
INK = "#231f20"
TEAL = "#8ed3cc"
GREEN = "#c3d64b"
PINK = "#f3a7b8"
LILAC = "#e9cce3"
YELLOW = "#fcf6b9"
BLUE = "#3dadde"
MAGENTA = "#dc218f"
GREY = "#bababa"
PALE_GREY = "#e4e4e4"

#: Lineage colours, used consistently across the figures.
CELL_FILLS = (TEAL, GREEN, PINK)


def _blob_path(
    x: float,
    y: float,
    radius: float,
    *,
    seed: int,
    squash: float = 1.0,
    angle: float = 0.0,
    wobble: float = 0.22,
    n: int = 9,
) -> Path:
    """A closed, smooth, slightly irregular outline: a cell rather than a circle."""
    rng = np.random.default_rng(seed)
    thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radii = radius * (1 + rng.uniform(-wobble, wobble, n))
    # One lobe drawn out, which is what gives these shapes their teardrop look.
    radii[rng.integers(n)] *= 1.0 + wobble

    pts = np.column_stack([radii * np.cos(thetas), radii * np.sin(thetas) * squash])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    pts = pts @ rot.T + np.array([x, y])

    # Catmull-Rom through the points, converted to cubic Bezier segments, so the
    # outline closes smoothly instead of showing corners.
    verts, codes = [tuple(pts[0])], [Path.MOVETO]
    for i in range(len(pts)):
        p0, p1 = pts[i - 1], pts[i]
        p2, p3 = pts[(i + 1) % len(pts)], pts[(i + 2) % len(pts)]
        verts += [tuple(p1 + (p2 - p0) / 6), tuple(p2 - (p3 - p1) / 6), tuple(p2)]
        codes += [Path.CURVE4, Path.CURVE4, Path.CURVE4]
    return Path(verts, codes)


def cell(
    ax,
    x,
    y,
    radius=1.0,
    *,
    fill=TEAL,
    seed=0,
    angle=0.0,
    squash=1.0,
    nucleus=True,
    nucleus_scale=0.42,
    nucleus_offset=(0.0, 0.0),
    alpha=1.0,
    edge=None,
    lw=0.0,
    zorder=2,
):
    """Draw one cell, and return its outline path."""
    path = _blob_path(x, y, radius, seed=seed, squash=squash, angle=angle)
    ax.add_patch(PathPatch(path, facecolor=fill, edgecolor=edge or "none", lw=lw, alpha=alpha, zorder=zorder))
    if nucleus:
        nx, ny = x + nucleus_offset[0] * radius, y + nucleus_offset[1] * radius
        ax.add_patch(
            PathPatch(
                _blob_path(nx, ny, radius * nucleus_scale, seed=seed + 101, wobble=0.12),
                facecolor=INK,
                edgecolor="none",
                zorder=zorder + 1,
            )
        )
    return path


def field(ax, extent=(0, 10, 0, 10), *, facecolor=CREAM):
    """A cream tissue panel with no axes."""
    x0, x1, y0, y1 = extent
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            boxstyle="round,pad=0,rounding_size=0.25",
            facecolor=facecolor,
            edgecolor="none",
            zorder=0,
        )
    )
    ax.set_xlim(x0 - 0.15, x1 + 0.15)
    ax.set_ylim(y0 - 0.15, y1 + 0.15)
    ax.set_aspect("equal")
    ax.axis("off")


def scatter_cells(ax, positions, *, fills=None, radius=0.9, seed=0, size_jitter=0.16, **kwargs):
    """Draw a set of cells at `positions`, cycling through `fills`.

    Sizes and nucleus placement vary a little from cell to cell, because a field
    of identical discs reads as a diagram of circles rather than of tissue.
    """
    fills = fills or CELL_FILLS
    rng = np.random.default_rng(seed)
    paths = []
    for i, (x, y) in enumerate(positions):
        fill = fills[i % len(fills)] if isinstance(fills, tuple | list) else fills
        paths.append(
            cell(
                ax,
                x,
                y,
                radius * (1 + rng.uniform(-size_jitter, size_jitter)),
                fill=fill,
                seed=seed + i * 7,
                angle=(seed + i) * 0.7,
                nucleus_offset=kwargs.pop("nucleus_offset", None) or tuple(rng.uniform(-0.18, 0.18, 2)),
                **kwargs,
            )
        )
    return paths


def label(ax, x, y, text, *, size=8.5, color=INK, ha="center", va="center", weight="normal"):
    """Place a label in the illustration's own coordinates."""
    ax.text(x, y, text, fontsize=size, color=color, ha=ha, va=va, weight=weight, zorder=10)


def hexgrid(n_rows, n_cols, *, spacing=2.1, jitter=0.25, seed=0):
    """Positions for a plausible tissue packing: offset rows, slightly jittered."""
    rng = np.random.default_rng(seed)
    out = []
    for r in range(n_rows):
        for c in range(n_cols):
            x = c * spacing + (r % 2) * spacing / 2
            y = r * spacing * 0.88
            out.append((x + rng.uniform(-jitter, jitter), y + rng.uniform(-jitter, jitter)))
    return out
