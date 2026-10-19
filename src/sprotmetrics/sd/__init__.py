"""SpatialData wrappers around the array and AnnData metrics.

Deliberately thin: every function here resolves element names, materialises the
arrays and delegates. It adds no metrics and no parameters of its own beyond the
element keys, so there is one implementation of each metric, not two.

``spatialdata`` is an optional dependency (``pip install sprotmetrics[spatialdata]``)
and is imported only inside this subpackage -- the ISP benchmark's components
pin ``zarr<3.0.0`` and would not install it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from sprotmetrics import _signal
from sprotmetrics._pixelstats import CellPixelStats
from sprotmetrics._pixelstats import cell_pixel_stats as _cell_pixel_stats
from sprotmetrics._registry import list_metrics
from sprotmetrics.sd._resolve import channel_names, materialise, resolve_element, resolve_table

if TYPE_CHECKING:
    from spatialdata import SpatialData

__all__ = ["cell_pixel_stats", "compute_all", "segmentation_agreement", "signal_background"]


def cell_pixel_stats(
    sdata: SpatialData,
    image_key: str | None = None,
    labels_key: str | None = None,
    *,
    markers: list[str] | None = None,
    **kwargs: Any,
) -> CellPixelStats:
    """:func:`sprotmetrics.cell_pixel_stats` on a SpatialData image/labels pair.

    Channel names come from the image's ``c`` coordinate unless given.
    """
    image = resolve_element(sdata, "images", image_key)
    labels = resolve_element(sdata, "labels", labels_key)
    return _cell_pixel_stats(
        materialise(image, label="image"),
        materialise(labels, label="labels"),
        markers if markers is not None else channel_names(image),
        **kwargs,
    )


def signal_background(
    sdata: SpatialData,
    image_key: str | None = None,
    labels_key: str | None = None,
    *,
    table_key: str | None = None,
    markers: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """:func:`sprotmetrics.signal_background` on a SpatialData object.

    `table_key` supplies the QC column for the ``_qc_pass`` variants.
    """
    image = resolve_element(sdata, "images", image_key)
    labels = resolve_element(sdata, "labels", labels_key)
    return _signal.signal_background(
        materialise(image, label="image"),
        materialise(labels, label="labels"),
        markers if markers is not None else channel_names(image),
        table=resolve_table(sdata, table_key),
        **kwargs,
    )


def segmentation_agreement(
    sdata: SpatialData,
    predicted_key: str,
    reference_key: str,
    **kwargs: Any,
) -> dict[str, float]:
    """:func:`sprotmetrics.segmentation_agreement` between two labels elements."""
    from sprotmetrics._reference import segmentation_agreement as _agreement

    return _agreement(
        materialise(resolve_element(sdata, "labels", predicted_key), label="predicted"),
        materialise(resolve_element(sdata, "labels", reference_key), label="reference"),
        **kwargs,
    )


def compute_all(
    sdata: SpatialData,
    *,
    image_key: str | None = None,
    labels_key: str | None = None,
    table_key: str | None = None,
    qc_variants: bool | str = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """Every family whose inputs this object provides, as a tidy frame.

    Columns are ``metric_id, value, family, direction, maximize``, joined from
    :func:`sprotmetrics.list_metrics` so a caller can sort by what is rankable
    without looking anything up.
    """
    import sprotmetrics as spm

    values: dict[str, float] = {}
    table = resolve_table(sdata, table_key)
    if table is not None:
        for family in (
            spm.basic_statistics,
            spm.marker_distribution,
            spm.normalization_diagnostics,
            spm.covariate_dependence,
            spm.segmentation_quality,
        ):
            try:
                values.update(family(table, qc_variants=qc_variants))
            except (KeyError, ValueError):
                # A family whose inputs this object does not carry is skipped,
                # rather than failing the whole sweep.
                continue
    if getattr(sdata, "images", None) and getattr(sdata, "labels", None):
        values.update(
            signal_background(
                sdata,
                image_key,
                labels_key,
                table_key=table_key,
                qc_variants=qc_variants,
                **kwargs,
            )
        )

    frame = pd.DataFrame({"metric_id": list(values), "value": list(values.values())})
    catalogue = list_metrics(include_qc_variants=True)[["family", "direction", "maximize"]]
    return frame.join(catalogue, on="metric_id").set_index("metric_id")
