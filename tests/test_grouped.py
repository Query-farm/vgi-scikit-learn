"""Unit tests for the per-group feature/target helpers.

The full aggregate+scalar lifecycle is covered by
test/sql/sklearn_grouped.test.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from vgi_sklearn.grouped import _matrix_for, _parse, _struct_rows


def _struct(rows: list[dict]) -> pa.Array:
    return pa.array(rows, type=pa.struct([pa.field("a", pa.float64()), pa.field("b", pa.float64())]))


class TestStructRows:
    def test_names_and_rows(self) -> None:
        names, rows = _struct_rows(_struct([{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}]))
        assert names == ["a", "b"]
        assert rows == [[1.0, 2.0], [3.0, 4.0]]

    def test_non_struct_errors(self) -> None:
        with pytest.raises(ValueError, match="must be a STRUCT"):
            _struct_rows(pa.array([1.0, 2.0], type=pa.float64()))

    def test_non_numeric_field_errors(self) -> None:
        s = pa.array([{"a": "x", "b": 1.0}], type=pa.struct([pa.field("a", pa.string()), pa.field("b", pa.float64())]))
        with pytest.raises(ValueError, match="not numeric"):
            _struct_rows(s)


class TestMatrixFor:
    def test_aligns_by_name_not_position(self) -> None:
        # model trained on [a, b]; input struct has them in the other order
        s = pa.array(
            [{"b": 20.0, "a": 10.0}], type=pa.struct([pa.field("b", pa.float64()), pa.field("a", pa.float64())])
        )
        m = _matrix_for(["a", "b"], s)
        assert np.array_equal(m, np.array([[10.0, 20.0]]))

    def test_extra_columns_ignored(self) -> None:
        s = pa.array(
            [{"a": 1.0, "b": 2.0, "c": 9.0}],
            type=pa.struct([pa.field("a", pa.float64()), pa.field("b", pa.float64()), pa.field("c", pa.float64())]),
        )
        assert np.array_equal(_matrix_for(["a", "b"], s), np.array([[1.0, 2.0]]))

    def test_missing_feature_errors(self) -> None:
        with pytest.raises(ValueError, match="missing model column"):
            _matrix_for(["a", "b", "z"], _struct([{"a": 1.0, "b": 2.0}]))


def test_parse() -> None:
    assert _parse("") == {}
    assert _parse("  ") == {}
    assert _parse('{"n_estimators": 50}') == {"n_estimators": 50}
