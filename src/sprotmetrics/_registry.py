"""The metric catalogue, as code.

Direction -- whether a larger value is better -- is what lets a benchmark *rank*
preprocessing methods rather than merely describe them. In the ISP benchmark it
lived only in a hand-maintained CSV, so nothing could read it at runtime and the
Viash configs omitted the ``info.metrics[]`` block the OpenProblems schema wants.

Here it is a field on :class:`MetricInfo`, attached to the function that computes
the quantity by the :func:`metric` decorator. One decorator expands into one entry
per emitted metric id: ``aggregations=("median", "min")`` on ``rel_sni`` registers
both ``median_rel_sni`` and ``min_rel_sni``, so 49 ids need roughly 26 functions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, TypeVar

import numpy as np
import pandas as pd

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
    "metric",
    "register_placeholder",
    "to_openproblems_info",
]

#: Suffix marking the variant computed over QC-passing cells only.
QC_SUFFIX = "qc_pass"


class Task(StrEnum):
    """The preprocessing stage a metric scores. Deliberately many-valued."""

    IMAGE_CORRECTION = "image correction"
    SEGMENTATION = "segmentation"
    AGGREGATION = "aggregation"
    QC = "QC"
    NORMALIZATION = "normalization"
    MARKER_CHARACTERISATION = "marker characterisation"
    DATASET_STATISTICS = "dataset statistics"
    ALL_AXES = "all axes"


class Reference(StrEnum):
    """What a metric needs beyond the sample itself."""

    NONE = "none"
    PANEL_KNOWLEDGE = "panel knowledge"
    REFERENCE_MASKS = "reference segmentation masks"
    CELL_TYPE_LABELS = "cell-type labels"
    EXTERNAL_ATLAS = "external atlas"
    REFERENCE_SCRNA = "reference scRNA-seq"


class Inputs(StrEnum):
    """The shape of data a metric consumes; this determines its signature."""

    MATRIX = "matrix"
    MATRIX_OBS = "matrix + obs"
    MATRIX_TWO_LAYERS = "matrix, two layers"
    IMAGE_LABELS = "image + labels"
    LABELS_PAIR = "labels + reference labels"


class Direction(StrEnum):
    """Whether, and how confidently, the metric ranks methods.

    ``NONE`` and ``AMBIGUOUS`` are not decoration. A metric saturated near 0.995,
    or one whose value could equally reflect biology or noise, must not silently
    become a ranking criterion -- so :meth:`MetricInfo.to_openproblems` refuses to
    emit a ``maximize`` flag for them.
    """

    HIGHER = "higher is better"
    LOWER = "lower is better"
    NONE = "descriptive"
    AMBIGUOUS = "ambiguous"


class Strength(StrEnum):
    """How much weight a directional metric can bear."""

    STRONG = "strong"
    WEAK = "weak"


class Status(StrEnum):
    """Lifecycle of a metric id."""

    STABLE = "stable"
    EXPERIMENTAL = "experimental"
    DEPRECATED = "deprecated"


@dataclass(frozen=True, slots=True)
class MetricInfo:
    """Everything the benchmark, the docs and the OpenProblems schema need about one id."""

    metric_id: str
    label: str
    summary: str
    description: str
    family: str
    module: str
    func: Callable[..., Any] | None
    aggregation: str | None
    suffix: str | None
    task: tuple[Task, ...]
    reference: Reference
    inputs: Inputs
    min: float
    max: float
    direction: Direction
    strength: Strength = Strength.STRONG
    status: Status = Status.STABLE
    qc_variant: bool = True
    references: tuple[str, ...] = ()
    links: Mapping[str, str] = field(default_factory=dict)
    origin: str = ""
    superseded_by: str | None = None
    notes: str = ""

    @property
    def maximize(self) -> bool | None:
        """``True``/``False`` for directional metrics, ``None`` where ranking is unsafe."""
        return {Direction.HIGHER: True, Direction.LOWER: False}.get(self.direction)

    def to_openproblems(self) -> dict[str, Any]:
        """One entry of ``info.metrics[]`` as ``common/schemas/task_metric.yaml`` wants it."""
        if self.maximize is None:
            raise ValueError(
                f"{self.metric_id} is {self.direction.value}; the OpenProblems schema has no "
                "way to express that, and emitting maximize=true would let the scoring stage "
                "rank methods on it. Exclude it, or give it a direction."
            )
        out: dict[str, Any] = {
            "name": self.metric_id,
            "label": self.label,
            "summary": self.summary,
            "description": self.description,
            "min": "-.inf" if self.min == -np.inf else self.min,
            "max": "+.inf" if self.max == np.inf else self.max,
            "maximize": self.maximize,
        }
        if self.references:
            out["references"] = {"doi": list(self.references)}
        if self.links:
            out["links"] = dict(self.links)
        return out


_REGISTRY: dict[str, MetricInfo] = {}
F = TypeVar("F", bound=Callable[..., Any])


def _compose_id(name: str, aggregation: str | None, suffix: str | None) -> str:
    base = f"{aggregation}_{name}" if aggregation else name
    return f"{base}_{suffix}" if suffix else base


def _docstring_parts(func: Callable[..., Any]) -> tuple[str, str]:
    doc = (func.__doc__ or "").strip()
    if not doc:
        return "", ""
    first = doc.split("\n", 1)[0].strip()
    return first, doc


def metric(
    *,
    name: str,
    label: str,
    task: tuple[Task, ...] | list[Task],
    reference: Reference,
    inputs: Inputs,
    min: float,
    max: float,
    direction: Direction,
    family: str,
    aggregations: tuple[str | None, ...] = (None,),
    suffixes: tuple[str | None, ...] = (None,),
    summary: str | None = None,
    description: str | None = None,
    strength: Strength = Strength.STRONG,
    status: Status = Status.STABLE,
    qc_variant: bool = True,
    references: tuple[str, ...] = (),
    links: Mapping[str, str] | None = None,
    origin: str = "",
    notes: str = "",
) -> Callable[[F], F]:
    """Register the metric ids a quantity function produces.

    One id per ``(aggregation, suffix)`` pair. The function's docstring supplies
    `summary` (first line) and `description` (all of it) unless they are given
    explicitly, so the prose that reaches the OpenProblems config, the rendered
    docs and the catalogue is the same prose, written once.
    """

    def decorate(func: F) -> F:
        auto_summary, auto_description = _docstring_parts(func)
        for aggregation in aggregations:
            for suffix in suffixes:
                metric_id = _compose_id(name, aggregation, suffix)
                if metric_id in _REGISTRY:
                    raise ValueError(
                        f"{metric_id} is already registered by "
                        f"{_REGISTRY[metric_id].module}.{_REGISTRY[metric_id].family}"
                    )
                _REGISTRY[metric_id] = MetricInfo(
                    metric_id=metric_id,
                    label=label,
                    summary=summary or auto_summary,
                    description=description or auto_description,
                    family=family,
                    module=func.__module__,
                    func=func,
                    aggregation=aggregation,
                    suffix=suffix,
                    task=tuple(task),
                    reference=reference,
                    inputs=inputs,
                    min=min,
                    max=max,
                    direction=direction,
                    strength=strength,
                    status=status,
                    qc_variant=qc_variant,
                    references=tuple(references),
                    links=dict(links or {}),
                    origin=origin,
                    notes=notes,
                )
        return func

    return decorate


def register_placeholder(
    *,
    metric_id: str,
    label: str,
    summary: str,
    family: str,
    task: tuple[Task, ...] | list[Task],
    reference: Reference,
    inputs: Inputs,
    direction: Direction = Direction.NONE,
    superseded_by: str | None = None,
    notes: str = "",
    origin: str = "",
    status: Status = Status.DEPRECATED,
) -> None:
    """Record a metric that exists in the literature but that this package will not compute.

    Keeping these queryable is the point: "why is there no SNR metric?" is then
    answered by ``list_metrics(include_deprecated=True)`` rather than by reading a
    spreadsheet. ``snr_per_protein``, for instance, is exactly ``1 / cv``.
    """
    _REGISTRY[metric_id] = MetricInfo(
        metric_id=metric_id,
        label=label,
        summary=summary,
        description=summary,
        family=family,
        module="",
        func=None,
        aggregation=None,
        suffix=None,
        task=tuple(task),
        reference=reference,
        inputs=inputs,
        min=float("nan"),
        max=float("nan"),
        direction=direction,
        status=status,
        qc_variant=False,
        superseded_by=superseded_by,
        notes=notes,
        origin=origin,
    )


def get_metric(metric_id: str) -> MetricInfo:
    """Look up one metric, resolving a ``_qc_pass`` id to its base entry."""
    if metric_id in _REGISTRY:
        return _REGISTRY[metric_id]
    if metric_id.endswith(f"_{QC_SUFFIX}"):
        base = _REGISTRY.get(metric_id[: -len(QC_SUFFIX) - 1])
        if base is not None and base.qc_variant:
            return replace(base, metric_id=metric_id)
    raise KeyError(f"unknown metric id: {metric_id!r}")


def _as_tuple(value: object) -> tuple[Any, ...] | None:
    if value is None:
        return None
    return tuple(value) if isinstance(value, list | tuple | set) else (value,)


def list_metrics(
    *,
    family: str | list[str] | None = None,
    task: Task | list[Task] | None = None,
    reference: Reference | list[Reference] | None = None,
    inputs: Inputs | list[Inputs] | None = None,
    status: Status | list[Status] | None = None,
    directional_only: bool = False,
    include_deprecated: bool = False,
    include_qc_variants: bool = False,
) -> pd.DataFrame:
    """The catalogue as a DataFrame, indexed by ``metric_id``.

    This is the source the benchmark's ``docs/metrics_catalogue.csv`` should be
    generated from, rather than maintained alongside.
    """
    rows = []
    for info in _REGISTRY.values():
        if not include_deprecated and info.status is Status.DEPRECATED:
            continue
        if family is not None and info.family not in _as_tuple(family):
            continue
        if task is not None and not set(info.task) & set(_as_tuple(task)):
            continue
        if reference is not None and info.reference not in _as_tuple(reference):
            continue
        if inputs is not None and info.inputs not in _as_tuple(inputs):
            continue
        if status is not None and info.status not in _as_tuple(status):
            continue
        if directional_only and info.maximize is None:
            continue
        variants = [(info.metric_id, None)]
        if include_qc_variants and info.qc_variant:
            variants.append((f"{info.metric_id}_{QC_SUFFIX}", QC_SUFFIX))
        for mid, variant in variants:
            rows.append(
                {
                    "metric_id": mid,
                    "label": info.label,
                    "summary": info.summary,
                    "family": info.family,
                    "module": info.module,
                    "function": info.func.__name__ if info.func else "",
                    "aggregation": info.aggregation or "",
                    "suffix": info.suffix or "",
                    "variant": variant or "",
                    "task": ", ".join(t.value for t in info.task),
                    "reference": info.reference.value,
                    "inputs": info.inputs.value,
                    "min": info.min,
                    "max": info.max,
                    "direction": info.direction.value,
                    "maximize": info.maximize,
                    "strength": info.strength.value,
                    "status": info.status.value,
                    "qc_variant": info.qc_variant,
                    "references": ", ".join(info.references),
                    "origin": info.origin,
                    "superseded_by": info.superseded_by or "",
                    "notes": info.notes,
                }
            )
    return pd.DataFrame(rows).set_index("metric_id") if rows else pd.DataFrame()


def compute(metric_id: str, *args: Any, **kwargs: Any) -> float:
    """Compute one metric id, applying its registered panel aggregation.

    ``compute("median_marker_coverage", adata)`` is the median over
    ``marker_coverage(adata)``; the ``_qc_pass`` form adds the QC subset.
    """
    from sprotmetrics._utils import DEFAULT_QC_COLUMN, aggregate

    info = get_metric(metric_id)
    if info.func is None:
        hint = f"; see {info.superseded_by}" if info.superseded_by else ""
        raise KeyError(f"{metric_id} is {info.status.value} and has no implementation{hint}")
    if metric_id.endswith(f"_{QC_SUFFIX}"):
        kwargs.setdefault("subset", DEFAULT_QC_COLUMN)
    return aggregate(np.atleast_1d(info.func(*args, **kwargs)), info.aggregation)


def to_openproblems_info(
    *,
    family: str | None = None,
    metric_ids: list[str] | None = None,
    include_qc_variants: bool = True,
) -> dict[str, Any]:
    """``{"metrics": [...]}`` for a Viash component's ``info:`` block.

    Descriptive and ambiguous metrics are dropped: the schema has no way to say
    "no direction", so including them would hand the scoring stage a ranking
    criterion the metric cannot support. They are still computed and reported --
    they just do not appear here.
    """
    if metric_ids is None:
        frame = list_metrics(family=family, include_qc_variants=include_qc_variants)
        metric_ids = list(frame.index)
    entries = []
    for mid in metric_ids:
        info = get_metric(mid)
        if info.maximize is None:
            continue
        entries.append(info.to_openproblems())
    return {"metrics": entries}
