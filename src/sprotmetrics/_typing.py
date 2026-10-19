"""Argument types shared by every metric function."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from anndata import AnnData
    from scipy.sparse import spmatrix

# A cell-by-marker matrix, however the caller happens to hold it. PEP 695 aliases
# are evaluated lazily, so the names above need not exist at runtime.
type MatrixLike = AnnData | np.ndarray | spmatrix | pd.DataFrame

# A per-cell vector. A ``str`` names a column of ``.obs`` and is only valid when
# the matrix is an AnnData; anything else is passed through after a length check.
type VectorLike = np.ndarray | pd.Series | Sequence[float] | str

# A per-cell selection: a boolean array, an integer index, an ``.obs`` column
# name, or None for "every cell".
type Mask = np.ndarray | pd.Series | Sequence[bool] | Sequence[int] | str | None

__all__ = ["Any", "Mapping", "MatrixLike", "Mask", "Sequence", "VectorLike"]
