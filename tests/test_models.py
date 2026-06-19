"""Unit tests for the model registry and estimator catalog.

The full fit -> predict -> list -> drop lifecycle is covered end-to-end by
test/sql/sklearn_models.test; here we test the storage backend and helpers.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from vgi_sklearn.models import _parse_params, build_estimator
from vgi_sklearn.registry import (
    LocalDiskStore,
    ModelMetadata,
    ModelNameError,
    ModelNotFoundError,
    validate_name,
)


def _fitted() -> LogisticRegression:
    x = np.array([[0.0], [1.0], [0.0], [1.0]])
    y = np.array([0, 1, 0, 1])
    return LogisticRegression().fit(x, y)


def _meta(name: str = "m") -> ModelMetadata:
    return ModelMetadata(
        name=name,
        estimator="logistic_regression",
        task="classification",
        target="y",
        feature_names=["a"],
        classes=[0, 1],
        n_samples=4,
        n_features=1,
        train_score=1.0,
        sklearn_version="x",
        created_at="now",
    )


class TestLocalDiskStore:
    def test_roundtrip(self, tmp_path) -> None:
        store = LocalDiskStore(tmp_path)
        store.save(_fitted(), _meta())
        assert store.exists("m")
        est, meta = store.load("m")
        assert meta.name == "m"
        assert meta.feature_names == ["a"]
        assert meta.classes == [0, 1]
        assert int(est.predict([[1.0]])[0]) == 1

    def test_list(self, tmp_path) -> None:
        store = LocalDiskStore(tmp_path)
        store.save(_fitted(), _meta("a"))
        store.save(_fitted(), _meta("b"))
        assert sorted(m.name for m in store.list()) == ["a", "b"]

    def test_delete(self, tmp_path) -> None:
        store = LocalDiskStore(tmp_path)
        store.save(_fitted(), _meta())
        assert store.delete("m") is True
        assert store.delete("m") is False
        assert not store.exists("m")

    def test_load_missing_raises(self, tmp_path) -> None:
        with pytest.raises(ModelNotFoundError):
            LocalDiskStore(tmp_path).load("nope")


class TestValidateName:
    def test_accepts_reasonable(self) -> None:
        assert validate_name("iris_rf-1.2") == "iris_rf-1.2"

    @pytest.mark.parametrize("bad", ["", "../etc", "a/b", ".hidden", "with space"])
    def test_rejects_unsafe(self, bad: str) -> None:
        with pytest.raises(ModelNameError):
            validate_name(bad)


class TestEstimatorCatalog:
    def test_build_with_params(self) -> None:
        task, est = build_estimator("random_forest_classifier", {"n_estimators": 7})
        assert task == "classification"
        assert est.n_estimators == 7

    def test_regression_task(self) -> None:
        task, est = build_estimator("ridge", {})
        assert task == "regression"

    def test_unknown_estimator(self) -> None:
        with pytest.raises(ValueError, match="unknown estimator"):
            build_estimator("does_not_exist", {})


class TestParseParams:
    def test_empty(self) -> None:
        assert _parse_params("") == {}
        assert _parse_params("   ") == {}

    def test_json_object(self) -> None:
        assert _parse_params('{"n_estimators": 200, "max_depth": 4}') == {"n_estimators": 200, "max_depth": 4}

    def test_non_object_rejected(self) -> None:
        with pytest.raises(ValueError, match="JSON object"):
            _parse_params("[1, 2, 3]")
