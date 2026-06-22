"""Shared feature handling, including automatic categorical (string) encoding.

scikit-learn estimators need a numeric ``X``. To let callers pass raw string
columns, ``fit`` wraps the estimator in a ``Pipeline`` whose first step
one-hot-encodes the string columns (and passes numeric/boolean columns through);
``predict`` then replays the same encoding because the whole pipeline is the
stored model. Numeric and boolean features are coerced to floats; only
string/large-string columns are treated as categorical.

The ``categorical`` mask (one bool per feature, in feature order) is recorded in
the model metadata so prediction rebuilds ``X`` with the same column dtypes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa


def is_categorical(arrow_type: pa.DataType) -> bool:
    """Whether an Arrow column type is a (string) categorical feature."""
    return pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type)


def categorical_mask(field_types: list[pa.DataType]) -> list[bool]:
    """One ``is_categorical`` flag per column, in order."""
    return [is_categorical(t) for t in field_types]


def rows_from_table(table: pa.Table, feature_names: list[str]) -> list[list[Any]]:
    """Extract raw per-row feature values (strings kept as strings) from a table."""
    cols = [table.column(n).to_pylist() for n in feature_names]
    if not cols:
        return [[] for _ in range(table.num_rows)]
    return [list(r) for r in zip(*cols, strict=True)]


def build_x(rows: list[list[Any]], cat_mask: list[bool]) -> np.ndarray:
    """Build a feature matrix from raw rows.

    Returns a float matrix when there are no categorical columns, otherwise an
    object array where categorical cells are strings and numeric cells floats
    (so a downstream ``ColumnTransformer`` can one-hot the former and pass the
    latter through).
    """
    n_cols = len(cat_mask)
    if not rows:
        return np.empty((0, n_cols), dtype=object if any(cat_mask) else float)
    if not any(cat_mask):
        return np.asarray(rows, dtype=float)
    x = np.empty((len(rows), n_cols), dtype=object)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            if cat_mask[j]:
                x[i, j] = "" if v is None else str(v)
            else:
                x[i, j] = float("nan") if v is None else float(v)
    return x


def wrap_estimator(estimator: Any, cat_mask: list[bool]) -> Any:
    """Wrap an estimator in a one-hot Pipeline when any feature is categorical.

    The pipeline one-hot-encodes the categorical columns (ignoring categories
    unseen at fit time) and passes the numeric columns through, then runs the
    estimator. With no categorical columns the estimator is returned unchanged.
    """
    if not any(cat_mask):
        return estimator
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    cat_idx = [i for i, c in enumerate(cat_mask) if c]
    num_idx = [i for i, c in enumerate(cat_mask) if not c]
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_idx),
            ("num", "passthrough", num_idx),
        ]
    )
    return Pipeline([("pre", pre), ("est", estimator)])


# When the estimator is wrapped in a pipeline, hyperparameter names must be
# prefixed to reach the estimator step (e.g. ``n_estimators`` -> ``est__n_estimators``).
PIPELINE_PARAM_PREFIX = "est__"


def prefix_grid(grid: dict[str, Any], wrapped: bool) -> dict[str, Any]:
    """Prefix grid-search parameter names for a wrapped (pipeline) estimator."""
    if not wrapped:
        return grid
    return {f"{PIPELINE_PARAM_PREFIX}{k}": v for k, v in grid.items()}
