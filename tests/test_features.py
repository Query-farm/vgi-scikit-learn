"""Unit tests for shared categorical feature handling (vgi_sklearn.features)."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from vgi_sklearn.features import (
    build_x,
    categorical_mask,
    is_categorical,
    prefix_grid,
    rows_from_table,
    wrap_estimator,
)


def test_categorical_mask_only_strings() -> None:
    types = [pa.string(), pa.float64(), pa.int64(), pa.bool_(), pa.large_string()]
    assert categorical_mask(types) == [True, False, False, False, True]
    assert is_categorical(pa.string())
    assert not is_categorical(pa.float64())


def test_rows_from_table_preserves_strings() -> None:
    t = pa.table({"a": ["x", "y"], "b": [1.0, 2.0]})
    assert rows_from_table(t, ["a", "b"]) == [["x", 1.0], ["y", 2.0]]


def test_rows_from_table_no_features() -> None:
    t = pa.table({"id": [1, 2, 3]})
    assert rows_from_table(t, []) == [[], [], []]


class TestBuildX:
    def test_all_numeric_is_float_matrix(self) -> None:
        x = build_x([[1.0, 2.0], [3.0, 4.0]], [False, False])
        assert x.dtype == float
        assert np.array_equal(x, np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_mixed_is_object_with_strings(self) -> None:
        x = build_x([["NYC", 1.0], ["LA", 2.0]], [True, False])
        assert x.dtype == object
        assert x[0, 0] == "NYC"
        assert x[1, 1] == 2.0

    def test_none_handling(self) -> None:
        x = build_x([[None, None]], [True, False])
        assert x[0, 0] == ""
        assert np.isnan(x[0, 1])

    def test_empty_rows(self) -> None:
        assert build_x([], [True, False]).shape == (0, 2)


class TestWrapEstimator:
    def test_no_categoricals_returns_estimator_unchanged(self) -> None:
        est = DecisionTreeClassifier()
        assert wrap_estimator(est, [False, False]) is est

    def test_categorical_wraps_in_pipeline(self) -> None:
        est = DecisionTreeClassifier()
        wrapped = wrap_estimator(est, [True, False])
        assert isinstance(wrapped, Pipeline)
        # the pipeline fits on object data with a string column and predicts
        x = build_x([["a", 1.0], ["b", 2.0], ["a", 1.5], ["b", 2.5]], [True, False])
        wrapped.fit(x, [0, 1, 0, 1])
        assert list(wrapped.predict(build_x([["a", 1.0]], [True, False]))) == [0]
        # unseen category is tolerated (handle_unknown='ignore')
        assert wrapped.predict(build_x([["z", 9.0]], [True, False])).shape == (1,)


def test_prefix_grid() -> None:
    grid = {"max_depth": [1, 2], "n_estimators": [10]}
    assert prefix_grid(grid, False) == grid
    assert prefix_grid(grid, True) == {"est__max_depth": [1, 2], "est__n_estimators": [10]}
