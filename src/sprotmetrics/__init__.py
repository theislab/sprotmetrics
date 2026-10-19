"""Quality metrics for imaging-based spatial proteomics."""

from importlib.metadata import version

from sprotmetrics._registry import (
    Direction,
    Inputs,
    MetricInfo,
    Reference,
    Status,
    Strength,
    Task,
    compute,
    get_metric,
    list_metrics,
    to_openproblems_info,
)

__all__ = [
    "Direction",
    "Inputs",
    "MetricInfo",
    "Reference",
    "Status",
    "Strength",
    "Task",
    "compute",
    "get_metric",
    "list_metrics",
    "to_openproblems_info",
]

__version__ = version("sprotmetrics")
