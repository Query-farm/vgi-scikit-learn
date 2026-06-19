"""Tests for the dataset table functions."""

from __future__ import annotations

from sklearn import datasets as skd

from vgi_sklearn.datasets import IrisFunction
from vgi_sklearn.schema_utils import dedupe_names, snake_case

from .harness import invoke_table_function


class TestSnakeCase:
    def test_sklearn_feature_label(self) -> None:
        assert snake_case("sepal length (cm)") == "sepal_length_cm"

    def test_leading_digit(self) -> None:
        assert snake_case("3d coordinate") == "f_3d_coordinate"

    def test_empty(self) -> None:
        assert snake_case("  ") == "feature"

    def test_dedupe(self) -> None:
        assert dedupe_names(["a", "a", "b", "a"]) == ["a", "a_2", "b", "a_3"]


class TestIris:
    def test_shape_and_schema(self) -> None:
        table = invoke_table_function(IrisFunction)
        assert table.num_rows == 150
        assert table.column_names == [
            "sample_id",
            "sepal_length_cm",
            "sepal_width_cm",
            "petal_length_cm",
            "petal_width_cm",
            "target",
            "target_name",
        ]

    def test_values_match_sklearn(self) -> None:
        bunch = skd.load_iris()
        table = invoke_table_function(IrisFunction).to_pydict()
        assert table["sample_id"][:3] == [0, 1, 2]
        assert table["sepal_length_cm"][0] == bunch.data[0, 0]
        assert table["target"][0] == int(bunch.target[0])
        assert table["target_name"][0] == bunch.target_names[bunch.target[0]]

    def test_three_species(self) -> None:
        table = invoke_table_function(IrisFunction).to_pydict()
        assert set(table["target_name"]) == {"setosa", "versicolor", "virginica"}
        assert set(table["target"]) == {0, 1, 2}
