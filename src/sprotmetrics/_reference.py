"""Segmentation scored against reference masks.

The only metrics here that need a second dataset. They are unusable on
exemplar-001, whose public download carries no masks -- which is exactly why the
reference-free metrics elsewhere in this package exist. They are included
because the comparison is standard, and because a user who *does* have masks
should not have to reimplement object matching.

Adapted from ``compare_segmentation_to_gt.py`` in the ISP benchmark, with the
CSV writing and overlay plotting dropped: a metrics package should return
numbers and let the caller decide how to store or draw them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sprotmetrics._registry import Direction, Inputs, Reference, Task, metric

__all__ = [
    "foreground_agreement",
    "label_areas",
    "match_objects",
    "object_f1",
    "object_overlaps",
    "segmentation_agreement",
    "split_merge_stats",
]

FAMILY = "segmentation_reference"


def _as_labels(x: np.ndarray) -> np.ndarray:
    arr = np.squeeze(np.asarray(x))
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D label mask, got shape {arr.shape}")
    return arr.astype(np.int64, copy=False)


def label_areas(labels: np.ndarray) -> pd.Series:
    """Pixel area of each non-zero label."""
    labels = _as_labels(labels)
    ids, counts = np.unique(labels, return_counts=True)
    keep = ids > 0
    return pd.Series(counts[keep], index=ids[keep].astype(int), name="area")


def foreground_agreement(predicted: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Pixel-level precision, recall, Dice and IoU of foreground against reference.

    Ignores how pixels are grouped into objects: a single mask covering two cells
    scores perfectly here. Read it with :func:`object_f1`, which does not.
    """
    pred_fg = _as_labels(predicted) > 0
    ref_fg = _as_labels(reference) > 0
    tp = int(np.logical_and(pred_fg, ref_fg).sum())
    fp = int(np.logical_and(pred_fg, ~ref_fg).sum())
    fn = int(np.logical_and(~pred_fg, ref_fg).sum())
    tn = int(np.logical_and(~pred_fg, ~ref_fg).sum())

    def ratio(num: float, den: float) -> float:
        return float(num / den) if den > 0 else float("nan")

    return {
        "pixel_tp": tp,
        "pixel_fp": fp,
        "pixel_fn": fn,
        "pixel_tn": tn,
        "foreground_precision": ratio(tp, tp + fp),
        "foreground_recall": ratio(tp, tp + fn),
        "foreground_dice": ratio(2 * tp, 2 * tp + fp + fn),
        "foreground_iou": ratio(tp, tp + fp + fn),
    }


def object_overlaps(predicted: np.ndarray, reference: np.ndarray) -> pd.DataFrame:
    """Every overlapping label pair, with intersection, union, IoU and overlap fractions."""
    predicted, reference = _as_labels(predicted), _as_labels(reference)
    both = (predicted > 0) & (reference > 0)
    columns = [
        "predicted_label",
        "reference_label",
        "intersection",
        "predicted_area",
        "reference_area",
        "union",
        "iou",
        "overlap_frac_predicted",
        "overlap_frac_reference",
    ]
    if both.sum() == 0:
        return pd.DataFrame(columns=columns)

    table = (
        pd.DataFrame(
            {
                "predicted_label": predicted[both].astype(int),
                "reference_label": reference[both].astype(int),
            }
        )
        .groupby(["predicted_label", "reference_label"])
        .size()
        .reset_index(name="intersection")
    )
    pred_area, ref_area = label_areas(predicted), label_areas(reference)
    table["predicted_area"] = table["predicted_label"].map(pred_area).astype(float)
    table["reference_area"] = table["reference_label"].map(ref_area).astype(float)
    table["union"] = table["predicted_area"] + table["reference_area"] - table["intersection"]
    table["iou"] = table["intersection"] / table["union"]
    table["overlap_frac_predicted"] = table["intersection"] / table["predicted_area"]
    table["overlap_frac_reference"] = table["intersection"] / table["reference_area"]
    return table[columns]


def match_objects(overlaps: pd.DataFrame, *, iou_threshold: float = 0.5) -> pd.DataFrame:
    """Greedy one-to-one matching by descending IoU, above `iou_threshold`."""
    if overlaps.empty:
        return overlaps.copy()
    used_pred: set[int] = set()
    used_ref: set[int] = set()
    rows = []
    for _, row in overlaps.sort_values("iou", ascending=False).iterrows():
        if row["iou"] < iou_threshold:
            break
        pred, ref = int(row["predicted_label"]), int(row["reference_label"])
        if pred in used_pred or ref in used_ref:
            continue
        used_pred.add(pred)
        used_ref.add(ref)
        rows.append(row)
    return pd.DataFrame(rows, columns=overlaps.columns)


@metric(
    name="object_f1",
    label="Object-level F1 against reference masks",
    family=FAMILY,
    task=(Task.SEGMENTATION,),
    reference=Reference.REFERENCE_MASKS,
    inputs=Inputs.LABELS_PAIR,
    min=0,
    max=1,
    direction=Direction.HIGHER,
    suffixes=("at_iou_10", "at_iou_25", "at_iou_50"),
    qc_variant=False,
    origin="compare_segmentation_to_gt.py in task_isp_preprocessing",
)
def object_f1(
    predicted: np.ndarray,
    reference: np.ndarray,
    *,
    thresholds: tuple[float, ...] = (0.1, 0.25, 0.5),
) -> np.ndarray:
    """F1 of greedily matched objects, at each IoU threshold.

    Unlike :func:`foreground_agreement` this sees splits and merges: a mask
    covering two reference cells matches at most one of them.
    """
    return np.asarray(detection_table(predicted, reference, thresholds=thresholds)["f1"].to_numpy(), dtype=float)


def detection_table(
    predicted: np.ndarray,
    reference: np.ndarray,
    *,
    thresholds: tuple[float, ...] = (0.1, 0.25, 0.5),
) -> pd.DataFrame:
    """True positives, false positives, precision, recall and F1 per IoU threshold."""
    predicted, reference = _as_labels(predicted), _as_labels(reference)
    overlaps = object_overlaps(predicted, reference)
    n_pred = int((np.unique(predicted) > 0).sum())
    n_ref = int((np.unique(reference) > 0).sum())

    rows = []
    for threshold in thresholds:
        tp = len(match_objects(overlaps, iou_threshold=threshold))
        fp, fn = n_pred - tp, n_ref - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        if np.isfinite(precision) and np.isfinite(recall):
            # Zero precision and zero recall means nothing matched, which is a
            # score of 0, not an undefined one. The script this came from
            # returned NaN here while returning 0.0 for the no-overlap case --
            # the same outcome reported two different ways, which makes an
            # over-expanded segmentation look unmeasurable rather than bad.
            total = precision + recall
            f1 = 2 * precision * recall / total if total > 0 else 0.0
        else:
            f1 = float("nan")
        rows.append(
            {
                "iou_threshold": threshold,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
    return pd.DataFrame(rows)


def split_merge_stats(
    predicted: np.ndarray, reference: np.ndarray, *, min_overlap_fraction: float = 0.1
) -> dict[str, float]:
    """How often one predicted mask covers several reference cells, and the reverse.

    Splits and merges are the two segmentation failures that object F1 reports
    only as a lower score; this says which of them happened.
    """
    overlaps = object_overlaps(predicted, reference)
    n_pred = int((np.unique(_as_labels(predicted)) > 0).sum())
    n_ref = int((np.unique(_as_labels(reference)) > 0).sum())
    if overlaps.empty:
        return {
            "merged_predicted_objects": 0,
            "split_reference_objects": 0,
            "merge_rate": float("nan"),
            "split_rate": float("nan"),
        }
    real = overlaps[
        (overlaps["overlap_frac_predicted"] >= min_overlap_fraction)
        | (overlaps["overlap_frac_reference"] >= min_overlap_fraction)
    ]
    merged = int((real.groupby("predicted_label")["reference_label"].nunique() > 1).sum())
    split = int((real.groupby("reference_label")["predicted_label"].nunique() > 1).sum())
    return {
        "merged_predicted_objects": merged,
        "split_reference_objects": split,
        "merge_rate": float(merged / n_pred) if n_pred else float("nan"),
        "split_rate": float(split / n_ref) if n_ref else float("nan"),
    }


def segmentation_agreement(
    predicted: np.ndarray,
    reference: np.ndarray,
    *,
    thresholds: tuple[float, ...] = (0.1, 0.25, 0.5),
    min_overlap_fraction: float = 0.1,
) -> dict[str, float]:
    """Every reference-based segmentation score, reported together."""
    out: dict[str, float] = dict(foreground_agreement(predicted, reference))
    table = detection_table(predicted, reference, thresholds=thresholds)
    for _, row in table.iterrows():
        tag = f"at_iou_{int(row['iou_threshold'] * 100):02d}"
        for column in ("precision", "recall", "f1"):
            out[f"object_{column}_{tag}"] = float(row[column])
    out.update(split_merge_stats(predicted, reference, min_overlap_fraction=min_overlap_fraction))
    return out
