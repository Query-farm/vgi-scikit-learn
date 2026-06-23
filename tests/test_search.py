"""Unit tests for grid_search's union type + param-grid translation.

grid_search needs union-typed arguments (a newer vgi-python); when the installed
vgi-python lacks ``TaggedUnion`` the module can't import, so the whole suite is
skipped. End-to-end coverage lives in test/sql/sklearn_grid_search.test.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

try:
    from vgi_sklearn.search import _GRID_UNION, _grid_size, _json_safe, _param_grid
    from vgi_sklearn.typed_models import _HPARAMS
except ImportError:  # pragma: no cover - depends on the installed vgi-python
    pytest.skip("vgi-python without union-tag support", allow_module_level=True)


class TestGridUnionType:
    def test_member_per_estimator(self) -> None:
        names = [_GRID_UNION.field(i).name for i in range(_GRID_UNION.num_fields)]
        assert set(names) == set(_HPARAMS)

    def test_members_are_structs_of_lists(self) -> None:
        # random_forest_classifier should expose n_estimators as a list<int>
        idx = next(i for i in range(_GRID_UNION.num_fields) if _GRID_UNION.field(i).name == "random_forest_classifier")
        member = _GRID_UNION.field(idx).type
        ne = member.field(member.get_field_index("n_estimators")).type
        assert pa.types.is_list(ne)
        assert pa.types.is_integer(ne.value_type)


class TestParamGrid:
    def test_only_listed_params_searched(self) -> None:
        grid = _param_grid("random_forest_classifier", {"n_estimators": [100, 300], "max_depth": None})
        assert grid == {"n_estimators": [100, 300]}  # max_depth (None) omitted -> estimator default

    def test_none_if_translation_per_element(self) -> None:
        # max_depth 0 means "unlimited" -> None, applied to each grid value
        grid = _param_grid("random_forest_classifier", {"max_depth": [0, 5, 10]})
        assert grid["max_depth"] == [None, 5, 10]

    def test_wrap_tuple_translation(self) -> None:
        # mlp hidden_units -> hidden_layer_sizes tuples
        grid = _param_grid("mlp_classifier", {"hidden_units": [50, 100]})
        assert grid["hidden_layer_sizes"] == [(50,), (100,)]

    def test_empty_value_is_empty_grid(self) -> None:
        assert _param_grid("ridge", None) == {}


def test_json_safe_tuples() -> None:
    assert _json_safe((50,)) == [50]
    assert _json_safe(5) == 5
    assert _json_safe(None) is None


class TestGridSize:
    def test_product_of_list_lengths(self) -> None:
        assert _grid_size({"a": [1, 2, 3], "b": [10, 20]}) == 6

    def test_empty_grid_is_one(self) -> None:
        assert _grid_size({}) == 1  # randomized_search then caps n_iter at 1
