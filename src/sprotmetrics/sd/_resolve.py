"""Turning SpatialData element names into the arrays the metrics want."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from spatialdata import SpatialData

#: Pixels above which materialising a lazy array is worth warning about.
PIXEL_BUDGET = 500_000_000


def resolve_element(sdata: SpatialData, kind: str, key: str | None):
    """One element of `kind`, named or inferred when there is exactly one.

    With several and no name the error lists them, rather than silently picking
    whichever the dictionary happened to order first.
    """
    elements = getattr(sdata, kind)
    if key is not None:
        if key not in elements:
            raise KeyError(f"{key!r} is not in sdata.{kind} (have: {sorted(elements)})")
        return elements[key]
    names = sorted(elements)
    if len(names) == 1:
        return elements[names[0]]
    if not names:
        raise KeyError(f"sdata.{kind} is empty")
    raise ValueError(f"sdata.{kind} holds {len(names)} elements ({names}); name the one you mean")


def materialise(element: Any, *, label: str = "element") -> np.ndarray:
    """Dense ndarray from a SpatialData element, computing a lazy one if needed.

    Multiscale elements resolve to their full-resolution level. This is the one
    place a caller can accidentally pull a whole mosaic into memory, so it says
    so rather than simply filling the machine.
    """
    import warnings

    data = element
    if hasattr(data, "scale0"):  # multiscale: take full resolution
        data = next(iter(data["scale0"].values()))
    values = getattr(data, "data", data)
    size = int(np.prod(getattr(values, "shape", ()) or (0,)))
    if hasattr(values, "compute"):
        if size > PIXEL_BUDGET:
            warnings.warn(
                f"materialising {label} pulls {size:,} pixels into memory; "
                "crop it first if that is more than you meant",
                UserWarning,
                stacklevel=3,
            )
        values = values.compute()
    return np.asarray(values)


def channel_names(element: Any) -> list[str] | None:
    """Channel names from an image element's ``c`` coordinate, if it has them."""
    data = element
    if hasattr(data, "scale0"):
        data = next(iter(data["scale0"].values()))
    coords = getattr(data, "coords", None)
    if coords is not None and "c" in coords:
        return [str(c) for c in np.asarray(coords["c"].values)]
    return None


def resolve_table(sdata: SpatialData, key: str | None):
    """One table, named or inferred; ``None`` when the object carries none."""
    tables = getattr(sdata, "tables", {})
    if not tables:
        return None
    return resolve_element(sdata, "tables", key)
