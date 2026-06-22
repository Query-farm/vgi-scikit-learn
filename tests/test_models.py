"""Unit tests for the model registry and estimator catalog.

The full fit -> predict -> list -> drop lifecycle is covered end-to-end by
test/sql/sklearn_models.test; here we test the storage backend and helpers.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from vgi_sklearn.models import _parse_params, build_estimator, estimator_param_names
from vgi_sklearn.registry import (
    LocalDiskStore,
    ModelMetadata,
    ModelNameError,
    ModelNotFoundError,
    pack_model,
    unpack_meta,
    unpack_model,
    validate_name,
)
from vgi_sklearn.typed_models import _HPARAMS, _estimator_kwargs


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

    def test_unknown_hyperparameter_lists_valid(self) -> None:
        with pytest.raises(ValueError, match="unknown hyperparameter"):
            build_estimator("ridge", {"nonsense": 1})

    def test_estimator_param_names(self) -> None:
        names = estimator_param_names("ridge")
        assert "alpha" in names and "fit_intercept" in names


class TestModelBlob:
    def test_pack_unpack_roundtrip(self) -> None:
        from sklearn.linear_model import LogisticRegression

        x = np.array([[0.0], [1.0], [0.0], [1.0]])
        est = LogisticRegression().fit(x, np.array([0, 1, 0, 1]))
        meta = ModelMetadata(
            name="b",
            estimator="logistic_regression",
            task="classification",
            target="y",
            feature_names=["a"],
            classes=[0, 1],
            n_features=1,
        )
        blob = pack_model(est, meta)
        # metadata-only read is cheap and correct
        assert unpack_meta(blob).feature_names == ["a"]
        # full read restores a working estimator
        est2, meta2 = unpack_model(blob)
        assert meta2.classes == [0, 1]
        assert int(est2.predict([[1.0]])[0]) == 1

    def test_bad_blob_raises(self) -> None:
        with pytest.raises(ValueError, match="valid sklearn model BLOB"):
            unpack_meta(b"\x00\x00")


class TestTypedFitSpec:
    def test_every_estimator_has_a_spec(self) -> None:
        from vgi_sklearn.models import _ESTIMATORS

        assert set(_HPARAMS) == set(_ESTIMATORS)

    def test_typed_params_are_valid_for_estimator(self) -> None:
        # Every exposed hyperparameter (translated to its sklearn kwarg) must be
        # a real param of that estimator.
        for name, spec in _HPARAMS.items():
            valid = set(estimator_param_names(name))
            for hp in spec:
                assert (hp.kwarg or hp.name) in valid, f"{name}.{hp.name} -> {hp.kwarg or hp.name} not valid"

    def test_kwargs_translation(self) -> None:
        from types import SimpleNamespace

        # max_depth 0 -> None; mlp hidden_units -> hidden_layer_sizes tuple
        rf = _estimator_kwargs(
            _HPARAMS["random_forest_classifier"],
            SimpleNamespace(n_estimators=10, max_depth=0, min_samples_split=2, max_features="sqrt", random_state=0),
        )
        assert rf["max_depth"] is None and rf["n_estimators"] == 10
        mlp = _estimator_kwargs(
            _HPARAMS["mlp_classifier"],
            SimpleNamespace(hidden_units=20, alpha=0.0001, max_iter=50, learning_rate_init=0.001, random_state=0),
        )
        assert mlp["hidden_layer_sizes"] == (20,)


class TestParseParams:
    def test_empty(self) -> None:
        assert _parse_params("") == {}
        assert _parse_params("   ") == {}

    def test_json_object(self) -> None:
        assert _parse_params('{"n_estimators": 200, "max_depth": 4}') == {"n_estimators": 200, "max_depth": 4}

    def test_non_object_rejected(self) -> None:
        with pytest.raises(ValueError, match="JSON object"):
            _parse_params("[1, 2, 3]")
