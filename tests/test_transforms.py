"""Unit tests for the transform compute logic and output schema.

The full buffering lifecycle (sink -> combine -> finalize over storage) is
covered by the SQL integration tests in test/sql/sklearn_transforms.test; here
we check the scikit-learn computation and the bind-time schema directly.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.datasets import load_iris

from vgi_sklearn.transforms import (
    IsolationForestFn,
    KMeansFn,
    MinMaxScalerFn,
    PcaFn,
    SimpleImputerFn,
    StandardScalerFn,
    TruncatedSvdFn,
)

X = load_iris().data
FEATS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def _stack(out: dict[str, list[float]], cols: list[str]) -> np.ndarray:
    return np.array([out[c] for c in cols]).T


class TestScalers:
    def test_standard_scaler_zero_mean_unit_var(self) -> None:
        out = StandardScalerFn.transform(X, FEATS, SimpleNamespace())
        arr = _stack(out, FEATS)
        assert np.allclose(arr.mean(axis=0), 0.0, atol=1e-9)
        assert np.allclose(arr.std(axis=0), 1.0, atol=1e-6)

    def test_standard_scaler_fields_mirror_features(self) -> None:
        fields = StandardScalerFn.output_fields(FEATS, SimpleNamespace())
        assert [f.name for f in fields] == FEATS

    def test_minmax_in_unit_range(self) -> None:
        out = MinMaxScalerFn.transform(X, FEATS, SimpleNamespace())
        arr = _stack(out, FEATS)
        assert arr.min() >= -1e-9
        assert arr.max() <= 1.0 + 1e-9


class TestDecomposition:
    def test_pca_shapes(self) -> None:
        args = SimpleNamespace(n_components=2)
        out = PcaFn.transform(X, FEATS, args)
        assert set(out) == {"component_1", "component_2"}
        assert len(out["component_1"]) == len(X)
        assert [f.name for f in PcaFn.output_fields(FEATS, args)] == ["component_1", "component_2"]

    def test_pca_caps_components_at_n_features(self) -> None:
        fields = PcaFn.output_fields(FEATS, SimpleNamespace(n_components=99))
        assert len(fields) == len(FEATS)

    def test_truncated_svd(self) -> None:
        args = SimpleNamespace(n_components=2)
        out = TruncatedSvdFn.transform(X, FEATS, args)
        assert set(out) == {"component_1", "component_2"}
        assert [f.name for f in TruncatedSvdFn.output_fields(FEATS, args)] == ["component_1", "component_2"]


class TestClustering:
    def test_kmeans_labels(self) -> None:
        args = SimpleNamespace(n_clusters=3, random_state=0)
        out = KMeansFn.transform(X, FEATS, args)
        assert len(out["cluster"]) == len(X)
        assert set(out["cluster"]) == {0, 1, 2}


class TestOutliers:
    def test_isolation_forest_flags_some(self) -> None:
        args = SimpleNamespace(contamination=0.1, random_state=0)
        out = IsolationForestFn.transform(X, FEATS, args)
        assert set(out["is_outlier"]) <= {0, 1}
        n_out = sum(out["is_outlier"])
        assert 0 < n_out < len(X)
        assert len(out["anomaly_score"]) == len(X)


class TestImputer:
    def test_mean_imputation_fills_nan(self) -> None:
        x = X.copy()
        x[0, 0] = np.nan
        out = SimpleImputerFn.transform(x, FEATS, SimpleNamespace(strategy="mean"))
        assert not np.isnan(out["sepal_length"][0])
        # imputed value equals the mean of the remaining column
        assert out["sepal_length"][0] == pytest.approx(float(np.mean(X[1:, 0])))
