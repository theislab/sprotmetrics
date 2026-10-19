#!/usr/bin/env python
"""Prove the extraction is a no-op, by re-deriving the ISP benchmark's own numbers.

The benchmark's metric components were moved into this package. The only gate
that can show the move changed nothing is a bit-for-bit comparison against
metric files the benchmark itself produced, on the same inputs.

Run it against a directory holding exemplar-001 pipeline outputs::

    python scripts/check_against_benchmark.py --runs /path/to/data/golden/*

Each run directory needs ``cell_by_protein_norm.h5ad``, ``cell_by_protein_qc.h5ad``,
``corrected_image.npy``, ``segmentation_labels.npy``, ``corrected_markers.json``
and the ``metrics_*.h5ad`` files to compare against.

Equality is exact, not approximate: the same inputs through the same arithmetic
give identical floats, and a tolerance would hide exactly the reordering and
dtype mistakes a port introduces. Two caveats learned the hard way:

* Run this under the *same* numpy as produced the reference files. The
  covariate family uses SVD and lstsq, and different BLAS builds disagree in the
  last bit -- which is drift in the environment, not a bug in the port.
* ``--tolerance`` exists for deliberate changes only, and prints what it hid.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np

NUCLEAR_MARKERS = "DNA_6,DNA_7,DNA_8"
EXCLUSIVE_PAIRS = "ECAD:CD45,ECAD:CD11B,SMA:CD45"


def families(run: Path):
    """Each benchmark family, as (name, callable) with the workflow's own parameters."""
    from sprotmetrics import (
        basic_statistics,
        covariate_dependence,
        marker_distribution,
        normalization_diagnostics,
        segmentation_quality,
        signal_background,
    )

    table = lambda: ad.read_h5ad(run / "cell_by_protein_norm.h5ad")

    def pixels():
        markers = json.loads((run / "corrected_markers.json").read_text())
        return (
            np.load(run / "corrected_image.npy"),
            np.load(run / "segmentation_labels.npy"),
            markers["markers"] if isinstance(markers, dict) else markers,
            ad.read_h5ad(run / "cell_by_protein_qc.h5ad"),
        )

    return {
        "basic": lambda: basic_statistics(table(), threshold=0.1, qc_variants=True),
        "marker_distribution": lambda: marker_distribution(
            table(),
            layer="counts",
            zero_threshold=0.0,
            coverage_threshold=0.1,
            quantiles=(0.01, 0.99),
            qc_variants=True,
        ),
        "normalization_diagnostics": lambda: normalization_diagnostics(
            table(),
            raw_layer="counts",
            normalized_layer="normalized",
            max_cells=5000,
            random_state=0,
            qc_variants=True,
        ),
        "segmentation_quality": lambda: segmentation_quality(
            table(),
            layer="counts",
            nuclear_markers=NUCLEAR_MARKERS,
            exclusive_pairs=EXCLUSIVE_PAIRS,
            n_neighbors=6,
            max_cells=5000,
            random_state=0,
            qc_variants=True,
        ),
        "technical_covariates": lambda: covariate_dependence(
            table(),
            covariate="area",
            layer="normalized",
            n_pcs=10,
            max_cells=5000,
            random_state=0,
            qc_variants=True,
        ),
        "signal_background": lambda: (
            lambda image, labels, markers, qc: signal_background(
                image,
                labels,
                markers,
                table=qc,
                dilation_iterations=3,
                norm_percentile=95.0,
                random_state=0,
                qc_variants=True,
            )
        )(*pixels()),
    }


def reference(path: Path) -> tuple[list[str], list[float]]:
    """Metric ids and values from a benchmark-produced metrics h5ad."""
    adata = ad.read_h5ad(path)
    return (
        [str(i) for i in adata.uns["metric_ids"]],
        [float(v) for v in adata.uns["metric_values"]],
    )


def main() -> int:
    """Compare every family in every run, and return non-zero on any mismatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="relative tolerance; use only for deliberate changes, which it reports",
    )
    args = parser.parse_args()

    checked = failures = tolerated = 0
    for run in args.runs:
        for name, compute in families(run).items():
            path = run / f"metrics_{name}.h5ad"
            if not path.exists():
                print(f"{run.name:18s} {name:28s} SKIP (no reference file)")
                continue
            want_ids, want_values = reference(path)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                got = compute()
            if list(got) != want_ids:
                missing = set(want_ids) - set(got)
                extra = set(got) - set(want_ids)
                detail = f"missing={sorted(missing)} extra={sorted(extra)}" or "order differs"
                print(f"{run.name:18s} {name:28s} ID MISMATCH {detail}")
                failures += 1
                continue
            diffs = []
            for key, want, have in zip(want_ids, want_values, got.values(), strict=True):
                checked += 1
                if (want != want and have != have) or want == have:
                    continue
                if args.tolerance and np.isclose(want, have, rtol=args.tolerance, equal_nan=True):
                    tolerated += 1
                    print(f"  tolerated {key}: {want!r} -> {have!r}")
                    continue
                diffs.append((key, want, have))
            if diffs:
                failures += 1
                print(f"{run.name:18s} {name:28s} {len(diffs)} DIFFER")
                for key, want, have in diffs[:5]:
                    print(f"    {key}: reference={want!r} got={have!r}")
            else:
                print(f"{run.name:18s} {name:28s} identical ({len(want_ids)} ids)")

    print(f"\n{checked} metric values compared, {failures} family mismatches", end="")
    print(f", {tolerated} within tolerance" if tolerated else "")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
