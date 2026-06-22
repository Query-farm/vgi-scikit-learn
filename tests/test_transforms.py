"""Unit tests for the transform compute logic and output schema.

The full buffering lifecycle (sink -> combine -> finalize over storage) is
covered by the SQL integration tests in test/sql/sklearn_transforms.test; here
we check the scikit-learn computation and the bind-time schema directly.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
from sklearn.datasets import load_iris

from vgi_sklearn.transforms import (
    AgglomerativeFn,
    GaussianMixtureFn,
    IsolationForestFn,
    KBinsDiscretizerFn,
    KMeansFn,
    LocalOutlierFactorFn,
    MaxAbsScalerFn,
    MinMaxScalerFn,
    OneClassSvmFn,
    OneHotEncoderFn,
    OrdinalEncoderFn,
    PcaFn,
    PowerTransformerFn,
    SimpleImputerFn,
    StandardScalerFn,
    TruncatedSvdFn,
    TsneFn,
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


class TestEncoders:
    def _table(self) -> pa.Table:
        return pa.table({"id": [1, 2, 3, 4], "city": ["NYC", "LA", "NYC", None]})

    def test_ordinal_codes_and_missing(self) -> None:
        out = OrdinalEncoderFn.encode(self._table(), ["city"], SimpleNamespace(id="id"))
        assert out["id"] == [1, 2, 3, 4]
        # alphabetical: LA=0, NYC=1; missing -> -1
        assert out["city"] == [1, 0, 1, -1]

    def test_ordinal_output_schema(self) -> None:
        schema = OrdinalEncoderFn.output_schema(
            pa.schema([pa.field("id", pa.int64()), pa.field("city", pa.string())]), ["city"], SimpleNamespace(id="id")
        )
        assert schema.names == ["id", "city"]
        assert schema.field("city").type == pa.int64()

    def test_one_hot_active_cells_only(self) -> None:
        out = OneHotEncoderFn.encode(self._table(), ["city"], SimpleNamespace(id="id"))
        # the NULL row contributes no active cell
        assert out["id"] == [1, 2, 3]
        assert set(out["category"]) == {"NYC", "LA"}
        assert out["value"] == [1.0, 1.0, 1.0]

    def test_one_hot_output_schema(self) -> None:
        schema = OneHotEncoderFn.output_schema(
            pa.schema([pa.field("id", pa.int64()), pa.field("city", pa.string())]), ["city"], SimpleNamespace(id="id")
        )
        assert schema.names == ["id", "feature", "category", "value"]


class TestAddedClustering:
    def test_agglomerative_labels(self) -> None:
        out = AgglomerativeFn.transform(X, FEATS, SimpleNamespace(n_clusters=3, linkage="ward"))
        assert len(out["cluster"]) == len(X)
        assert set(out["cluster"]) == {0, 1, 2}

    def test_gaussian_mixture_labels(self) -> None:
        args = SimpleNamespace(n_components=3, covariance_type="full", random_state=0)
        out = GaussianMixtureFn.transform(X, FEATS, args)
        assert set(out["cluster"]) <= {0, 1, 2}


class TestAddedOutliers:
    def test_lof_flags_some(self) -> None:
        out = LocalOutlierFactorFn.transform(X, FEATS, SimpleNamespace(n_neighbors=20, contamination=0.1))
        assert set(out["is_outlier"]) <= {0, 1}
        assert 0 < sum(out["is_outlier"]) < len(X)
        assert len(out["anomaly_score"]) == len(X)

    def test_one_class_svm_runs(self) -> None:
        out = OneClassSvmFn.transform(X, FEATS, SimpleNamespace(nu=0.1, kernel="rbf", gamma="scale"))
        assert len(out["anomaly_score"]) == len(X)


class TestAddedManifold:
    def test_tsne_two_components(self) -> None:
        args = SimpleNamespace(n_components=2, perplexity=10.0, random_state=0)
        out = TsneFn.transform(X, FEATS, args)
        assert set(out) == {"component_1", "component_2"}
        assert len(out["component_1"]) == len(X)


class TestAddedScalers:
    def test_maxabs_in_unit_range(self) -> None:
        out = MaxAbsScalerFn.transform(X, FEATS, SimpleNamespace())
        z = _stack(out, FEATS)
        assert np.all(np.abs(z) <= 1.0 + 1e-9)

    def test_power_transformer_runs(self) -> None:
        out = PowerTransformerFn.transform(X, FEATS, SimpleNamespace(method="yeo-johnson"))
        assert len(out["sepal_length"]) == len(X)

    def test_kbins_integer_codes(self) -> None:
        out = KBinsDiscretizerFn.transform(X, FEATS, SimpleNamespace(n_bins=4, strategy="quantile"))
        codes = out["petal_length"]
        assert all(isinstance(c, int) for c in codes)
        assert set(codes) <= {0, 1, 2, 3}
        assert [f.name for f in KBinsDiscretizerFn.output_fields(FEATS, SimpleNamespace())] == FEATS
