"""Regenerate every metric figure.

Run from the repository root::

    python docs/_figures/make_figures.py

Needs matplotlib and scipy, which are development dependencies rather than
documentation ones: the SVGs are committed, so building the docs does not plot.
"""

from __future__ import annotations

import importlib
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

#: Every module contributing figures, in the order they appear in the docs.
MODULES = ("fig_segmentation", "fig_signal", "fig_distribution")


def main() -> None:
    """Build every registered figure."""
    for name in MODULES:
        if not (HERE / f"{name}.py").exists():
            continue
        module = importlib.import_module(name)
        print(name)
        for figure in module.FIGURES:
            figure()


if __name__ == "__main__":
    main()
