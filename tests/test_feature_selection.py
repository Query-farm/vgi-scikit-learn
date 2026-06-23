"""Unit tests for feature_selection helpers.

The full buffering lifecycle is covered by test/sql/sklearn_feature_selection.test.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from vgi_sklearn.feature_selection import (
    _SCORE_FUNCS,
    _features_excluding,
    _score_func,
)


def test_features_excluding_drops_target_and_id() -> None:
    schema = pa.schema([pa.field(n, pa.float64()) for n in ["id", "a", "b", "y"]])
    assert _features_excluding(schema, "y", "id") == ["a", "b"]


def test_features_excluding_ignores_empty() -> None:
    schema = pa.schema([pa.field(n, pa.float64()) for n in ["a", "b"]])
    assert _features_excluding(schema, "", "") == ["a", "b"]


class TestScoreFuncs:
    def test_classification_scorers_use_int_targets(self) -> None:
        assert _SCORE_FUNCS["f_classif"] is True
        assert _SCORE_FUNCS["chi2"] is True
        assert _SCORE_FUNCS["mutual_info_classif"] is True

    def test_regression_scorers_use_float_targets(self) -> None:
        assert _SCORE_FUNCS["f_regression"] is False
        assert _SCORE_FUNCS["mutual_info_regression"] is False

    def test_every_name_resolves_to_a_callable(self) -> None:
        for name in _SCORE_FUNCS:
            assert callable(_score_func(name))

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(KeyError):
            _score_func("nope")
