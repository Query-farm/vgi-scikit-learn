"""Tests for the dataset table functions."""

from __future__ import annotations

import pyarrow as pa
from sklearn import datasets as skd

from vgi_sklearn.datasets import (
    BreastCancerFunction,
    DiabetesFunction,
    DigitsFunction,
    IrisFunction,
    MakeBlobsFunction,
    MakeCirclesFunction,
    MakeClassificationFunction,
    MakeMoonsFunction,
    MakeRegressionFunction,
    WineFunction,
)
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


class TestToyDatasets:
    def test_wine(self) -> None:
        table = invoke_table_function(WineFunction)
        assert table.num_rows == 178
        assert "target_name" in table.column_names

    def test_digits(self) -> None:
        table = invoke_table_function(DigitsFunction)
        assert table.num_rows == 1797
        # 64 pixel features + sample_id + target + target_name
        assert table.num_columns == 67

    def test_breast_cancer(self) -> None:
        table = invoke_table_function(BreastCancerFunction)
        assert table.num_rows == 569
        assert set(table.column("target").to_pylist()) == {0, 1}

    def test_diabetes_is_regression(self) -> None:
        table = invoke_table_function(DiabetesFunction)
        assert table.num_rows == 442
        # Regression: float target, no target_name column
        assert "target_name" not in table.column_names
        assert table.schema.field("target").type == pa.float64()


class TestGenerators:
    def test_make_classification_shape(self) -> None:
        table = invoke_table_function(
            MakeClassificationFunction,
            named={
                "n_samples": pa.scalar(60),
                "n_features": pa.scalar(5),
                "n_informative": pa.scalar(3),
                "n_classes": pa.scalar(3),
            },
        )
        assert table.num_rows == 60
        assert table.column_names == ["sample_id", "feature_0", "feature_1", "feature_2", "feature_3", "feature_4", "target"]
        assert set(table.column("target").to_pylist()) == {0, 1, 2}

    def test_make_classification_is_reproducible(self) -> None:
        kw = {"named": {"n_samples": pa.scalar(40), "n_features": pa.scalar(4), "random_state": pa.scalar(7)}}
        a = invoke_table_function(MakeClassificationFunction, **kw).to_pydict()
        b = invoke_table_function(MakeClassificationFunction, **kw).to_pydict()
        assert a == b

    def test_make_regression_target_is_float(self) -> None:
        table = invoke_table_function(
            MakeRegressionFunction,
            named={"n_samples": pa.scalar(30), "n_features": pa.scalar(3), "noise": pa.scalar(2.0)},
        )
        assert table.num_rows == 30
        assert table.schema.field("target").type == pa.float64()

    def test_make_blobs_cluster_column(self) -> None:
        table = invoke_table_function(
            MakeBlobsFunction,
            named={"n_samples": pa.scalar(90), "n_features": pa.scalar(2), "centers": pa.scalar(3)},
        )
        assert table.num_rows == 90
        assert "cluster" in table.column_names
        assert set(table.column("cluster").to_pylist()) == {0, 1, 2}

    def test_make_moons(self) -> None:
        table = invoke_table_function(MakeMoonsFunction, named={"n_samples": pa.scalar(50)})
        assert table.num_rows == 50
        assert table.column_names == ["sample_id", "feature_0", "feature_1", "target"]
        assert set(table.column("target").to_pylist()) == {0, 1}

    def test_make_circles(self) -> None:
        table = invoke_table_function(MakeCirclesFunction, named={"n_samples": pa.scalar(50)})
        assert table.num_rows == 50
        assert set(table.column("target").to_pylist()) == {0, 1}


class TestFetchers:
    def test_california_housing(self) -> None:
        # Downloads on first use; skip cleanly when offline / not cached.
        try:
            from vgi_sklearn.datasets import CaliforniaHousingFunction

            table = invoke_table_function(CaliforniaHousingFunction)
        except Exception as exc:  # noqa: BLE001 - network/cache dependent
            import pytest

            pytest.skip(f"california_housing unavailable: {exc}")
        assert table.num_rows == 20640
        assert table.schema.field("target").type == pa.float64()
        assert "MedInc" in [c.lower() for c in table.column_names] or "medinc" in table.column_names
