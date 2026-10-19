# sprotmetrics

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/theislab/sprotmetrics/test.yaml?branch=main
[badge-docs]: https://app.readthedocs.org/projects/sprotmetrics/badge/

Quality metrics for imaging-based spatial proteomics: segmentation,
signal-to-background, normalization, marker distribution and technical-covariate
diagnostics.

Multiplexed imaging pipelines make a long series of choices — how to correct the
image, how to segment, how to aggregate pixels into cells, what to filter, how to
normalize — and most of the numbers usually reported describe a run rather than
judge it. Median cell area of 76 against 255 does not say which segmentation was
better. These metrics are chosen and labelled so that the ones with a defensible
direction can be told apart from the ones without.

## Getting started

```python
import anndata as ad
import sprotmetrics as spm

adata = ad.read_h5ad("cell_by_protein.h5ad")

# One panel-level number per metric, including QC-passing variants.
spm.segmentation_quality(
    adata,
    nuclear_markers=["DNA_6", "DNA_7", "DNA_8"],
    exclusive_pairs=[("ECAD", "CD45")],
    qc_variants=True,
)
# {'dapi_positive_ratio': 0.535, 'replicate_correlation': 0.837, ...}

# Or the per-marker detail behind a summary.
spm.marker_table(adata)["coverage"].sort_values().head()
```

Pixel-level metrics see what the cell table cannot — signal that never made it
inside a mask. They share one pass over the image:

```python
import numpy as np

stats = spm.cell_pixel_stats(
    np.load("corrected_image.npy"),      # (channels, y, x)
    np.load("segmentation_labels.npy"),  # (y, x)
    markers=["DNA_6", "CD45", "ECAD"],
)
spm.rel_sni(stats)                       # per marker
spm.nuclear_signal_containment(stats, ["DNA_6"])
```

## Which metrics can rank a method?

Most metrics describe a distribution rather than judging it, and treating a
descriptive one as a score is how a benchmark ends up ranking on noise. The
catalogue records the difference, so you can ask for it:

```python
spm.list_metrics(directional_only=True)[["label", "direction", "strength"]]
spm.list_metrics(task=spm.Task.SEGMENTATION)
spm.list_metrics(include_deprecated=True).query("superseded_by != ''")
```

`strength` flags metrics that are directional but weak evidence on their own.
`nuclear_signal_containment` rises whenever masks grow, so it must be read
against `spillover_ratio`, which falls when they grow too far; either alone is
gameable.

## QC-passing variants

Every metric can be reported twice: over all segmented objects, and over those
that passed QC, with a `_qc_pass` suffix.

```python
spm.basic_statistics(adata, qc_variants=True)
```

This is what makes a QC step measurable at all. QC components typically annotate
rather than filter, so the all-cell metrics are identical whichever QC method
ran — the `_qc_pass` variants are where the choice becomes visible.

## Installation

Python 3.12 or newer.

```bash
pip install sprotmetrics
pip install "sprotmetrics[spatialdata]"   # for the sprotmetrics.sd wrappers
```

## Provenance

The metrics were extracted from the ISP preprocessing benchmark
(`task_isp_preprocessing`), itself derived from the OpenProblems
`task_ist_preprocessing`, and from the scverse hackathon `sprot` prototypes. The
extraction is verified: `scripts/check_against_benchmark.py` re-derives the
benchmark's own metric files and requires them to match bit for bit. See the `NOTICE` file in the repository for attribution and licensing of the
adapted code.

[tests]: https://github.com/theislab/sprotmetrics/actions/workflows/test.yaml
[documentation]: https://sprotmetrics.readthedocs.io
