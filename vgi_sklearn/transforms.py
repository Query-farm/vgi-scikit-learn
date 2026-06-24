"""Unsupervised scikit-learn transforms as DuckDB table-buffering functions.

These all need the *whole* input matrix before producing output (fit_transform),
so they use ``TableBufferingFunction``: every input batch is buffered to
execution-scoped storage during the sink phase, then in finalize the full
feature matrix is assembled, fit_transform/fit_predict is run once, and the
result is streamed out.

Conventions:
* The input relation IS the feature matrix X. Name an ``id`` column to carry it
  through to the output (so you can join predictions back); every other column
  is treated as a numeric feature.
* Output schema is declared at bind time from the arguments + input schema.

    SELECT * FROM sklearn.kmeans((SELECT id, x, y FROM points), id => 'id', n_clusters => 3);
    SELECT * FROM sklearn.pca((SELECT * FROM sklearn.iris()), id => 'sample_id', n_components => 2);
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import numpy as np
import pyarrow as pa
from vgi.arguments import Arg, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_buffering_function import TableBufferingParams
from vgi.table_function import BindParams
from vgi_rpc.rpc import OutputCollector

from .buffering import DrainState, SinkBuffer, input_schema_of, matrix
from .features import rows_from_table
from .schema_utils import columns_md_rows
from .schema_utils import field as sfield

_ID_NOTE = "If an `id` column is named, it is carried through unchanged as the first column."


def _scaler_md(value_type: str, value_desc: str) -> str:
    return columns_md_rows(
        [],
        note=(f"One {value_type} column per input feature (same name as the input), {value_desc} " + _ID_NOTE),
    )


_COMPONENTS_MD = columns_md_rows(
    [("component_<i>", "DOUBLE", "Projection onto component i (one per kept component).")],
    note=_ID_NOTE,
)
_CLUSTER_MD = columns_md_rows(
    [("cluster", "INTEGER", "Assigned cluster label (-1 = noise for density methods).")],
    note=_ID_NOTE,
)
_OUTLIER_MD = columns_md_rows(
    [
        ("anomaly_score", "DOUBLE", "Anomaly score; higher = more anomalous."),
        ("is_outlier", "INTEGER", "1 if flagged as an outlier, else 0."),
    ],
    note=_ID_NOTE,
)


@dataclass(slots=True, frozen=True)
class _BaseArgs:
    data: Annotated[TableInput, Arg(0, doc="Input feature table (each non-id column is a feature)")]
    id: Annotated[str, Arg("id", default="", doc="Optional column to carry through unchanged to the output")]


class _BufferingTransform[TArgs: _BaseArgs](SinkBuffer[TArgs, DrainState]):
    """Buffer the whole input, fit_transform once in finalize, stream out.

    Subclasses set ``FunctionArguments`` and implement ``output_fields`` (the
    non-id output columns) and ``transform`` (the scikit-learn computation).
    """

    @classmethod
    def feature_names(cls, input_schema: pa.Schema, id_col: str) -> list[str]:
        return [n for n in input_schema.names if n != id_col]

    @classmethod
    def output_fields(cls, feature_names: list[str], args: Any) -> list[pa.Field]:
        raise NotImplementedError

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        raise NotImplementedError

    @classmethod
    def build_output_schema(cls, input_schema: pa.Schema, args: Any) -> pa.Schema:
        feats = cls.feature_names(input_schema, args.id)
        fields: list[pa.Field] = []
        if args.id:
            fields.append(input_schema.field(args.id))
        fields.extend(cls.output_fields(feats, args))
        return pa.schema(fields)

    @classmethod
    def on_bind(cls, params: BindParams[TArgs]) -> BindResponse:
        assert params.bind_call.input_schema is not None
        return BindResponse(output_schema=cls.build_output_schema(params.bind_call.input_schema, params.args))

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[TArgs]) -> DrainState:
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[TArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        if state.done:
            out.finish()
            return
        state.done = True

        input_schema = input_schema_of(params)
        id_col = params.args.id
        feats = cls.feature_names(input_schema, id_col)

        table = cls.buffered_table(params, input_schema)
        if table is None:
            empty: dict[str, list[Any]] = {name: [] for name in params.output_schema.names}
            out.emit(pa.RecordBatch.from_pydict(empty, schema=params.output_schema))
            return

        x = matrix(table, feats)
        columns: dict[str, list[Any]] = {}
        if id_col:
            columns[id_col] = table.column(id_col).to_pylist()
        columns.update(cls.transform(x, feats, params.args))
        out.emit(pa.RecordBatch.from_pydict(columns, schema=params.output_schema))


def _ex(name: str, extra: str = "") -> list[FunctionExample]:
    args = f", {extra}" if extra else ""
    return [
        FunctionExample(
            sql=f"SELECT * FROM sklearn.{name}((SELECT * FROM sklearn.iris()), id => 'sample_id'{args})",
            description=f"Apply {name} to the iris features",
        )
    ]


# ===========================================================================
# Scalers (output mirrors the feature columns)
# ===========================================================================


def _scaler_fields(feature_names: list[str], _args: Any) -> list[pa.Field]:
    return [sfield(f, pa.float64(), f"Scaled value of {f}.") for f in feature_names]


class StandardScalerFn(_BufferingTransform[_BaseArgs]):
    """Standardize features to zero mean and unit variance."""

    FunctionArguments: ClassVar[type] = _BaseArgs

    class Meta:
        """VGI metadata for the standard_scaler function."""

        name = "standard_scaler"
        description = "Standardize features to zero mean and unit variance"
        categories = ["preprocessing", "scaling"]
        examples = _ex("standard_scaler")
        tags = {"vgi.columns_md": _scaler_md("DOUBLE", "the scaled value.")}

    output_fields = staticmethod(_scaler_fields)  # type: ignore[assignment]

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        """Run the scikit-learn computation and return the output columns."""
        from sklearn.preprocessing import StandardScaler

        z = StandardScaler().fit_transform(x)
        return {f: z[:, j].tolist() for j, f in enumerate(feature_names)}


class MinMaxScalerFn(_BufferingTransform[_BaseArgs]):
    """Scale features to the [0, 1] range."""

    FunctionArguments: ClassVar[type] = _BaseArgs

    class Meta:
        """VGI metadata for the minmax_scaler function."""

        name = "minmax_scaler"
        description = "Scale features to the [0, 1] range"
        categories = ["preprocessing", "scaling"]
        examples = _ex("minmax_scaler")
        tags = {"vgi.columns_md": _scaler_md("DOUBLE", "the scaled value.")}

    output_fields = staticmethod(_scaler_fields)  # type: ignore[assignment]

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        """Run the scikit-learn computation and return the output columns."""
        from sklearn.preprocessing import MinMaxScaler

        z = MinMaxScaler().fit_transform(x)
        return {f: z[:, j].tolist() for j, f in enumerate(feature_names)}


class RobustScalerFn(_BufferingTransform[_BaseArgs]):
    """Scale features using statistics robust to outliers (median/IQR)."""

    FunctionArguments: ClassVar[type] = _BaseArgs

    class Meta:
        """VGI metadata for the robust_scaler function."""

        name = "robust_scaler"
        description = "Scale features using statistics robust to outliers (median/IQR)"
        categories = ["preprocessing", "scaling"]
        examples = _ex("robust_scaler")
        tags = {"vgi.columns_md": _scaler_md("DOUBLE", "the scaled value.")}

    output_fields = staticmethod(_scaler_fields)  # type: ignore[assignment]

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        """Run the scikit-learn computation and return the output columns."""
        from sklearn.preprocessing import RobustScaler

        z = RobustScaler().fit_transform(x)
        return {f: z[:, j].tolist() for j, f in enumerate(feature_names)}


@dataclass(slots=True, frozen=True)
class NormalizerArgs(_BaseArgs):
    """Arguments for the normalizer function."""

    norm: Annotated[str, Arg("norm", default="l2", doc="Norm to use: 'l1', 'l2', or 'max'.")]


class NormalizerFn(_BufferingTransform[NormalizerArgs]):
    """Scale each sample (row) to unit norm."""

    FunctionArguments: ClassVar[type] = NormalizerArgs

    class Meta:
        """VGI metadata for the normalizer function."""

        name = "normalizer"
        description = "Scale each sample (row) to unit norm"
        categories = ["preprocessing", "scaling"]
        examples = _ex("normalizer", "norm => 'l2'")
        tags = {"vgi.columns_md": _scaler_md("DOUBLE", "the normalized value.")}

    output_fields = staticmethod(_scaler_fields)  # type: ignore[assignment]

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        """Run the scikit-learn computation and return the output columns."""
        from sklearn.preprocessing import Normalizer

        z = Normalizer(norm=args.norm).fit_transform(x)
        return {f: z[:, j].tolist() for j, f in enumerate(feature_names)}


@dataclass(slots=True, frozen=True)
class ImputerArgs(_BaseArgs):
    """Arguments for the imputer function."""

    strategy: Annotated[str, Arg("strategy", default="mean", doc="mean, median, most_frequent, or constant.")]


class SimpleImputerFn(_BufferingTransform[ImputerArgs]):
    """Fill missing (NULL/NaN) feature values using a column statistic."""

    FunctionArguments: ClassVar[type] = ImputerArgs

    class Meta:
        """VGI metadata for the simple_imputer function."""

        name = "simple_imputer"
        description = "Fill missing (NULL/NaN) feature values using a column statistic"
        categories = ["preprocessing", "imputation"]
        examples = _ex("simple_imputer", "strategy => 'median'")
        tags = {"vgi.columns_md": _scaler_md("DOUBLE", "the imputed value.")}

    output_fields = staticmethod(_scaler_fields)  # type: ignore[assignment]

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        """Run the scikit-learn computation and return the output columns."""
        from sklearn.impute import SimpleImputer

        z = SimpleImputer(strategy=args.strategy).fit_transform(x)
        return {f: z[:, j].tolist() for j, f in enumerate(feature_names)}


# ===========================================================================
# Dimensionality reduction (output = component_1..component_k)
# ===========================================================================


@dataclass(slots=True, frozen=True)
class ComponentsArgs(_BaseArgs):
    """Arguments for the components function."""

    n_components: Annotated[int, Arg("n_components", default=2, doc="Number of components to keep.")]


def _effective_components(n_components: int, n_features: int) -> int:
    return max(1, min(n_components, n_features))


def _component_fields(feature_names: list[str], args: Any) -> list[pa.Field]:
    k = _effective_components(args.n_components, len(feature_names))
    return [sfield(f"component_{i + 1}", pa.float64(), f"Projection onto component {i + 1}.") for i in range(k)]


class PcaFn(_BufferingTransform[ComponentsArgs]):
    """Principal component analysis (linear dimensionality reduction)."""

    FunctionArguments: ClassVar[type] = ComponentsArgs

    class Meta:
        """VGI metadata for the pca function."""

        name = "pca"
        description = "Principal component analysis (linear dimensionality reduction)"
        categories = ["decomposition", "dimensionality-reduction"]
        examples = _ex("pca", "n_components => 2")
        tags = {"vgi.columns_md": _COMPONENTS_MD}

    output_fields = staticmethod(_component_fields)

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        """Run the scikit-learn computation and return the output columns."""
        from sklearn.decomposition import PCA

        k = _effective_components(args.n_components, len(feature_names))
        comps = PCA(n_components=k).fit_transform(x)
        return {f"component_{i + 1}": comps[:, i].tolist() for i in range(k)}


class TruncatedSvdFn(_BufferingTransform[ComponentsArgs]):
    """Truncated SVD (LSA) dimensionality reduction."""

    FunctionArguments: ClassVar[type] = ComponentsArgs

    class Meta:
        """VGI metadata for the truncated_svd function."""

        name = "truncated_svd"
        description = "Truncated SVD (LSA) dimensionality reduction"
        categories = ["decomposition", "dimensionality-reduction"]
        examples = _ex("truncated_svd", "n_components => 2")
        tags = {"vgi.columns_md": _COMPONENTS_MD}

    @staticmethod
    def _svd_k(n_components: int, n_features: int) -> int:
        # TruncatedSVD requires n_components < n_features.
        return max(1, min(n_components, n_features - 1))

    @staticmethod
    def output_fields(feature_names: list[str], args: Any) -> list[pa.Field]:
        """Return the non-id output fields for the given features."""
        k = TruncatedSvdFn._svd_k(args.n_components, len(feature_names))
        return [sfield(f"component_{i + 1}", pa.float64(), f"SVD component {i + 1}.") for i in range(k)]

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        """Run the scikit-learn computation and return the output columns."""
        from sklearn.decomposition import TruncatedSVD

        k = cls._svd_k(args.n_components, len(feature_names))
        comps = TruncatedSVD(n_components=k, random_state=0).fit_transform(x)
        return {f"component_{i + 1}": comps[:, i].tolist() for i in range(k)}


# ===========================================================================
# Clustering (output = cluster label)
# ===========================================================================


def _cluster_fields(_feature_names: list[str], _args: Any) -> list[pa.Field]:
    return [sfield("cluster", pa.int32(), "Assigned cluster label (-1 = noise for density methods).", nullable=False)]


@dataclass(slots=True, frozen=True)
class KMeansArgs(_BaseArgs):
    """Arguments for the k_means function."""

    n_clusters: Annotated[int, Arg("n_clusters", default=8, doc="Number of clusters.")]
    random_state: Annotated[int, Arg("random_state", default=0, doc="Random seed.")]


class KMeansFn(_BufferingTransform[KMeansArgs]):
    """K-Means clustering; emits a cluster label per row."""

    FunctionArguments: ClassVar[type] = KMeansArgs

    class Meta:
        """VGI metadata for the kmeans function."""

        name = "kmeans"
        description = "K-Means clustering; emits a cluster label per row"
        categories = ["clustering"]
        examples = _ex("kmeans", "n_clusters => 3")
        tags = {"vgi.columns_md": _CLUSTER_MD}

    output_fields = staticmethod(_cluster_fields)  # type: ignore[assignment]

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        """Run the scikit-learn computation and return the output columns."""
        from sklearn.cluster import KMeans

        labels = KMeans(n_clusters=args.n_clusters, random_state=args.random_state, n_init=10).fit_predict(x)
        return {"cluster": [int(v) for v in labels]}


@dataclass(slots=True, frozen=True)
class DbscanArgs(_BaseArgs):
    """Arguments for the dbscan function."""

    eps: Annotated[float, Arg("eps", default=0.5, doc="Max neighbourhood distance.")]
    min_samples: Annotated[int, Arg("min_samples", default=5, doc="Min samples to form a dense region.")]


class DbscanFn(_BufferingTransform[DbscanArgs]):
    """DBSCAN density clustering; emits a cluster label per row (-1 = noise)."""

    FunctionArguments: ClassVar[type] = DbscanArgs

    class Meta:
        """VGI metadata for the dbscan function."""

        name = "dbscan"
        description = "DBSCAN density clustering; emits a cluster label per row (-1 = noise)"
        categories = ["clustering"]
        examples = _ex("dbscan", "eps => 0.5")
        tags = {"vgi.columns_md": _CLUSTER_MD}

    output_fields = staticmethod(_cluster_fields)  # type: ignore[assignment]

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        """Run the scikit-learn computation and return the output columns."""
        from sklearn.cluster import DBSCAN

        labels = DBSCAN(eps=args.eps, min_samples=args.min_samples).fit_predict(x)
        return {"cluster": [int(v) for v in labels]}


# ===========================================================================
# Outlier detection (output = anomaly_score + is_outlier)
# ===========================================================================


def _outlier_fields(_feature_names: list[str], _args: Any) -> list[pa.Field]:
    return [
        sfield("anomaly_score", pa.float64(), "Anomaly score; higher = more anomalous.", nullable=False),
        sfield("is_outlier", pa.int32(), "1 if flagged as an outlier, else 0.", nullable=False),
    ]


@dataclass(slots=True, frozen=True)
class IsolationForestArgs(_BaseArgs):
    """Arguments for the isolation_forest function."""

    contamination: Annotated[float, Arg("contamination", default=0.1, doc="Expected proportion of outliers (0-0.5).")]
    random_state: Annotated[int, Arg("random_state", default=0, doc="Random seed.")]


class IsolationForestFn(_BufferingTransform[IsolationForestArgs]):
    """Isolation Forest outlier detection; emits an anomaly score and flag per row."""

    FunctionArguments: ClassVar[type] = IsolationForestArgs

    class Meta:
        """VGI metadata for the isolation_forest function."""

        name = "isolation_forest"
        description = "Isolation Forest outlier detection; emits an anomaly score and flag per row"
        categories = ["outlier-detection"]
        examples = _ex("isolation_forest", "contamination => 0.1")
        tags = {"vgi.columns_md": _OUTLIER_MD}

    output_fields = staticmethod(_outlier_fields)  # type: ignore[assignment]

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        """Run the scikit-learn computation and return the output columns."""
        from sklearn.ensemble import IsolationForest

        model = IsolationForest(contamination=args.contamination, random_state=args.random_state)
        pred = model.fit_predict(x)
        score = -model.decision_function(x)  # flip so higher = more anomalous
        return {
            "anomaly_score": [float(v) for v in score],
            "is_outlier": [1 if v == -1 else 0 for v in pred],
        }


# ===========================================================================
# More scalers / discretizers (output mirrors the feature columns)
# ===========================================================================


class MaxAbsScalerFn(_BufferingTransform[_BaseArgs]):
    """Scale each feature by its maximum absolute value (to [-1, 1])."""

    FunctionArguments: ClassVar[type] = _BaseArgs

    class Meta:
        """VGI metadata for the maxabs_scaler function."""

        name = "maxabs_scaler"
        description = "Scale each feature by its maximum absolute value (to [-1, 1])"
        categories = ["preprocessing", "scaling"]
        examples = _ex("maxabs_scaler")
        tags = {"vgi.columns_md": _scaler_md("DOUBLE", "the scaled value.")}

    output_fields = staticmethod(_scaler_fields)  # type: ignore[assignment]

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        """Run the scikit-learn computation and return the output columns."""
        from sklearn.preprocessing import MaxAbsScaler

        z = MaxAbsScaler().fit_transform(x)
        return {f: z[:, j].tolist() for j, f in enumerate(feature_names)}


@dataclass(slots=True, frozen=True)
class PowerTransformerArgs(_BaseArgs):
    """Arguments for the power_transformer function."""

    method: Annotated[str, Arg("method", default="yeo-johnson", doc="'yeo-johnson' (any sign) or 'box-cox' (>0).")]


class PowerTransformerFn(_BufferingTransform[PowerTransformerArgs]):
    """Make features more Gaussian via a power transform (Yeo-Johnson / Box-Cox)."""

    FunctionArguments: ClassVar[type] = PowerTransformerArgs

    class Meta:
        """VGI metadata for the power_transformer function."""

        name = "power_transformer"
        description = "Make features more Gaussian via a power transform (Yeo-Johnson / Box-Cox)"
        categories = ["preprocessing", "scaling"]
        examples = _ex("power_transformer", "method => 'yeo-johnson'")
        tags = {"vgi.columns_md": _scaler_md("DOUBLE", "the transformed value.")}

    output_fields = staticmethod(_scaler_fields)  # type: ignore[assignment]

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        """Run the scikit-learn computation and return the output columns."""
        from sklearn.preprocessing import PowerTransformer

        z = PowerTransformer(method=args.method).fit_transform(x)
        return {f: z[:, j].tolist() for j, f in enumerate(feature_names)}


@dataclass(slots=True, frozen=True)
class QuantileTransformerArgs(_BaseArgs):
    """Arguments for the quantile_transformer function."""

    n_quantiles: Annotated[int, Arg("n_quantiles", default=1000, doc="Number of quantiles (capped at n_samples).")]
    output_distribution: Annotated[str, Arg("output_distribution", default="uniform", doc="'uniform' or 'normal'.")]


class QuantileTransformerFn(_BufferingTransform[QuantileTransformerArgs]):
    """Map features to a uniform or normal distribution via quantiles (robust to outliers)."""

    FunctionArguments: ClassVar[type] = QuantileTransformerArgs

    class Meta:
        """VGI metadata for the quantile_transformer function."""

        name = "quantile_transformer"
        description = "Map features to a uniform or normal distribution via quantiles (robust to outliers)"
        categories = ["preprocessing", "scaling"]
        examples = _ex("quantile_transformer", "output_distribution => 'normal'")
        tags = {"vgi.columns_md": _scaler_md("DOUBLE", "the transformed value.")}

    output_fields = staticmethod(_scaler_fields)  # type: ignore[assignment]

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        """Run the scikit-learn computation and return the output columns."""
        from sklearn.preprocessing import QuantileTransformer

        n_q = max(1, min(args.n_quantiles, x.shape[0]))
        z = QuantileTransformer(n_quantiles=n_q, output_distribution=args.output_distribution).fit_transform(x)
        return {f: z[:, j].tolist() for j, f in enumerate(feature_names)}


@dataclass(slots=True, frozen=True)
class BinarizerArgs(_BaseArgs):
    """Arguments for the binarizer function."""

    threshold: Annotated[float, Arg("threshold", default=0.0, doc="Values above this map to 1, else 0.")]


class BinarizerFn(_BufferingTransform[BinarizerArgs]):
    """Threshold features to 0/1."""

    FunctionArguments: ClassVar[type] = BinarizerArgs

    class Meta:
        """VGI metadata for the binarizer function."""

        name = "binarizer"
        description = "Threshold features to 0/1"
        categories = ["preprocessing"]
        examples = _ex("binarizer", "threshold => 0.0")
        tags = {"vgi.columns_md": _scaler_md("DOUBLE", "the 0/1 thresholded value.")}

    output_fields = staticmethod(_scaler_fields)  # type: ignore[assignment]

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        """Run the scikit-learn computation and return the output columns."""
        from sklearn.preprocessing import Binarizer

        z = Binarizer(threshold=args.threshold).fit_transform(x)
        return {f: z[:, j].tolist() for j, f in enumerate(feature_names)}


@dataclass(slots=True, frozen=True)
class KBinsDiscretizerArgs(_BaseArgs):
    """Arguments for the k_bins_discretizer function."""

    n_bins: Annotated[int, Arg("n_bins", default=5, doc="Number of bins per feature.")]
    strategy: Annotated[str, Arg("strategy", default="quantile", doc="'uniform', 'quantile', or 'kmeans'.")]


def _bin_fields(feature_names: list[str], _args: Any) -> list[pa.Field]:
    return [sfield(f, pa.int64(), f"Bin index for {f}.", nullable=False) for f in feature_names]


class KBinsDiscretizerFn(_BufferingTransform[KBinsDiscretizerArgs]):
    """Discretize continuous features into integer bins (one bin index column per feature)."""

    FunctionArguments: ClassVar[type] = KBinsDiscretizerArgs

    class Meta:
        """VGI metadata for the kbins_discretizer function."""

        name = "kbins_discretizer"
        description = "Discretize continuous features into integer bins (one bin index column per feature)"
        categories = ["preprocessing", "encoding"]
        examples = _ex("kbins_discretizer", "n_bins => 5")
        tags = {"vgi.columns_md": _scaler_md("BIGINT", "the bin index.")}

    output_fields = staticmethod(_bin_fields)  # type: ignore[assignment]

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        """Run the scikit-learn computation and return the output columns."""
        from sklearn.preprocessing import KBinsDiscretizer

        codes = KBinsDiscretizer(n_bins=args.n_bins, encode="ordinal", strategy=args.strategy).fit_transform(x)
        return {f: [int(v) for v in codes[:, j]] for j, f in enumerate(feature_names)}


# ===========================================================================
# More clustering (emit a cluster label per row, like kmeans/dbscan)
# ===========================================================================


@dataclass(slots=True, frozen=True)
class AgglomerativeArgs(_BaseArgs):
    """Arguments for the agglomerative function."""

    n_clusters: Annotated[int, Arg("n_clusters", default=2, doc="Number of clusters.")]
    linkage: Annotated[str, Arg("linkage", default="ward", doc="Linkage: ward, complete, average, single.")]


class AgglomerativeFn(_BufferingTransform[AgglomerativeArgs]):
    """Hierarchical (agglomerative) clustering; emits a cluster label per row."""

    FunctionArguments: ClassVar[type] = AgglomerativeArgs

    class Meta:
        """VGI metadata for the agglomerative_clustering function."""

        name = "agglomerative_clustering"
        description = "Hierarchical (agglomerative) clustering; emits a cluster label per row"
        categories = ["clustering"]
        examples = _ex("agglomerative_clustering", "n_clusters => 3")
        tags = {"vgi.columns_md": _CLUSTER_MD}

    output_fields = staticmethod(_cluster_fields)  # type: ignore[assignment]

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        """Run the scikit-learn computation and return the output columns."""
        from sklearn.cluster import AgglomerativeClustering

        labels = AgglomerativeClustering(n_clusters=args.n_clusters, linkage=args.linkage).fit_predict(x)
        return {"cluster": [int(v) for v in labels]}


@dataclass(slots=True, frozen=True)
class SpectralClusteringArgs(_BaseArgs):
    """Arguments for the spectral_clustering function."""

    n_clusters: Annotated[int, Arg("n_clusters", default=2, doc="Number of clusters.")]
    random_state: Annotated[int, Arg("random_state", default=0, doc="Random seed.")]


class SpectralClusteringFn(_BufferingTransform[SpectralClusteringArgs]):
    """Spectral clustering on the affinity graph; emits a cluster label per row."""

    FunctionArguments: ClassVar[type] = SpectralClusteringArgs

    class Meta:
        """VGI metadata for the spectral_clustering function."""

        name = "spectral_clustering"
        description = "Spectral clustering on the affinity graph; emits a cluster label per row"
        categories = ["clustering"]
        examples = _ex("spectral_clustering", "n_clusters => 3")
        tags = {"vgi.columns_md": _CLUSTER_MD}

    output_fields = staticmethod(_cluster_fields)  # type: ignore[assignment]

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        """Run the scikit-learn computation and return the output columns."""
        from sklearn.cluster import SpectralClustering

        labels = SpectralClustering(n_clusters=args.n_clusters, random_state=args.random_state).fit_predict(x)
        return {"cluster": [int(v) for v in labels]}


@dataclass(slots=True, frozen=True)
class MeanShiftArgs(_BaseArgs):
    """Arguments for the mean_shift function."""

    bandwidth: Annotated[float, Arg("bandwidth", default=0.0, doc="Kernel bandwidth; 0 = estimate automatically.")]


class MeanShiftFn(_BufferingTransform[MeanShiftArgs]):
    """Mean-shift clustering (auto-discovers the number of clusters)."""

    FunctionArguments: ClassVar[type] = MeanShiftArgs

    class Meta:
        """VGI metadata for the mean_shift function."""

        name = "mean_shift"
        description = "Mean-shift clustering (auto-discovers the number of clusters)"
        categories = ["clustering"]
        examples = _ex("mean_shift")
        tags = {"vgi.columns_md": _CLUSTER_MD}

    output_fields = staticmethod(_cluster_fields)  # type: ignore[assignment]

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        """Run the scikit-learn computation and return the output columns."""
        from sklearn.cluster import MeanShift

        labels = MeanShift(bandwidth=args.bandwidth or None).fit_predict(x)
        return {"cluster": [int(v) for v in labels]}


@dataclass(slots=True, frozen=True)
class BirchArgs(_BaseArgs):
    """Arguments for the birch function."""

    n_clusters: Annotated[int, Arg("n_clusters", default=3, doc="Number of clusters for the final step.")]
    threshold: Annotated[float, Arg("threshold", default=0.5, doc="Radius of a subcluster to absorb a sample.")]


class BirchFn(_BufferingTransform[BirchArgs]):
    """BIRCH clustering (memory-efficient for large datasets)."""

    FunctionArguments: ClassVar[type] = BirchArgs

    class Meta:
        """VGI metadata for the birch function."""

        name = "birch"
        description = "BIRCH clustering (memory-efficient for large datasets)"
        categories = ["clustering"]
        examples = _ex("birch", "n_clusters => 3")
        tags = {"vgi.columns_md": _CLUSTER_MD}

    output_fields = staticmethod(_cluster_fields)  # type: ignore[assignment]

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        """Run the scikit-learn computation and return the output columns."""
        from sklearn.cluster import Birch

        labels = Birch(n_clusters=args.n_clusters, threshold=args.threshold).fit_predict(x)
        return {"cluster": [int(v) for v in labels]}


@dataclass(slots=True, frozen=True)
class OpticsArgs(_BaseArgs):
    """Arguments for the optics function."""

    min_samples: Annotated[int, Arg("min_samples", default=5, doc="Min samples in a neighbourhood for a core point.")]


class OpticsFn(_BufferingTransform[OpticsArgs]):
    """OPTICS density clustering; emits a cluster label per row (-1 = noise)."""

    FunctionArguments: ClassVar[type] = OpticsArgs

    class Meta:
        """VGI metadata for the optics function."""

        name = "optics"
        description = "OPTICS density clustering; emits a cluster label per row (-1 = noise)"
        categories = ["clustering"]
        examples = _ex("optics", "min_samples => 5")
        tags = {"vgi.columns_md": _CLUSTER_MD}

    output_fields = staticmethod(_cluster_fields)  # type: ignore[assignment]

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        """Run the scikit-learn computation and return the output columns."""
        from sklearn.cluster import OPTICS

        labels = OPTICS(min_samples=args.min_samples).fit_predict(x)
        return {"cluster": [int(v) for v in labels]}


class MiniBatchKMeansFn(_BufferingTransform[KMeansArgs]):
    """Mini-batch K-Means clustering (faster on large datasets)."""

    FunctionArguments: ClassVar[type] = KMeansArgs

    class Meta:
        """VGI metadata for the minibatch_kmeans function."""

        name = "minibatch_kmeans"
        description = "Mini-batch K-Means clustering (faster on large datasets)"
        categories = ["clustering"]
        examples = _ex("minibatch_kmeans", "n_clusters => 3")
        tags = {"vgi.columns_md": _CLUSTER_MD}

    output_fields = staticmethod(_cluster_fields)  # type: ignore[assignment]

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        """Run the scikit-learn computation and return the output columns."""
        from sklearn.cluster import MiniBatchKMeans

        labels = MiniBatchKMeans(n_clusters=args.n_clusters, random_state=args.random_state, n_init=10).fit_predict(x)
        return {"cluster": [int(v) for v in labels]}


@dataclass(slots=True, frozen=True)
class GaussianMixtureArgs(_BaseArgs):
    """Arguments for the gaussian_mixture function."""

    n_components: Annotated[int, Arg("n_components", default=2, doc="Number of mixture components (clusters).")]
    covariance_type: Annotated[str, Arg("covariance_type", default="full", doc="full, tied, diag, or spherical.")]
    random_state: Annotated[int, Arg("random_state", default=0, doc="Random seed.")]


class GaussianMixtureFn(_BufferingTransform[GaussianMixtureArgs]):
    """Gaussian mixture model clustering; emits the most likely component per row."""

    FunctionArguments: ClassVar[type] = GaussianMixtureArgs

    class Meta:
        """VGI metadata for the gaussian_mixture function."""

        name = "gaussian_mixture"
        description = "Gaussian mixture model clustering; emits the most likely component per row"
        categories = ["clustering"]
        examples = _ex("gaussian_mixture", "n_components => 3")
        tags = {"vgi.columns_md": _CLUSTER_MD}

    output_fields = staticmethod(_cluster_fields)  # type: ignore[assignment]

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        """Run the scikit-learn computation and return the output columns."""
        from sklearn.mixture import GaussianMixture

        labels = GaussianMixture(
            n_components=args.n_components, covariance_type=args.covariance_type, random_state=args.random_state
        ).fit_predict(x)
        return {"cluster": [int(v) for v in labels]}


# ===========================================================================
# More outlier detection (anomaly_score + is_outlier, like isolation_forest)
# ===========================================================================


@dataclass(slots=True, frozen=True)
class LofArgs(_BaseArgs):
    """Arguments for the lof function."""

    n_neighbors: Annotated[int, Arg("n_neighbors", default=20, doc="Number of neighbours to use.")]
    contamination: Annotated[float, Arg("contamination", default=0.1, doc="Expected proportion of outliers (0-0.5).")]


class LocalOutlierFactorFn(_BufferingTransform[LofArgs]):
    """Local Outlier Factor; emits an anomaly score and flag per row."""

    FunctionArguments: ClassVar[type] = LofArgs

    class Meta:
        """VGI metadata for the local_outlier_factor function."""

        name = "local_outlier_factor"
        description = "Local Outlier Factor; emits an anomaly score and flag per row"
        categories = ["outlier-detection"]
        examples = _ex("local_outlier_factor", "n_neighbors => 20")
        tags = {"vgi.columns_md": _OUTLIER_MD}

    output_fields = staticmethod(_outlier_fields)  # type: ignore[assignment]

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        """Run the scikit-learn computation and return the output columns."""
        from sklearn.neighbors import LocalOutlierFactor

        model = LocalOutlierFactor(n_neighbors=args.n_neighbors, contamination=args.contamination)
        pred = model.fit_predict(x)
        score = -model.negative_outlier_factor_  # flip so higher = more anomalous
        return {
            "anomaly_score": [float(v) for v in score],
            "is_outlier": [1 if v == -1 else 0 for v in pred],
        }


@dataclass(slots=True, frozen=True)
class OneClassSvmArgs(_BaseArgs):
    """Arguments for the one_class_svm function."""

    nu: Annotated[float, Arg("nu", default=0.5, doc="Upper bound on the fraction of outliers (0-1).")]
    kernel: Annotated[str, Arg("kernel", default="rbf", doc="Kernel: rbf, linear, poly, sigmoid.")]
    gamma: Annotated[str, Arg("gamma", default="scale", doc="Kernel coefficient ('scale' or 'auto').")]


class OneClassSvmFn(_BufferingTransform[OneClassSvmArgs]):
    """One-Class SVM novelty/outlier detection; emits an anomaly score and flag per row."""

    FunctionArguments: ClassVar[type] = OneClassSvmArgs

    class Meta:
        """VGI metadata for the one_class_svm function."""

        name = "one_class_svm"
        description = "One-Class SVM novelty/outlier detection; emits an anomaly score and flag per row"
        categories = ["outlier-detection"]
        examples = _ex("one_class_svm", "nu => 0.1")
        tags = {"vgi.columns_md": _OUTLIER_MD}

    output_fields = staticmethod(_outlier_fields)  # type: ignore[assignment]

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        """Run the scikit-learn computation and return the output columns."""
        from sklearn.svm import OneClassSVM

        model = OneClassSVM(nu=args.nu, kernel=args.kernel, gamma=args.gamma)
        pred = model.fit_predict(x)
        score = -model.decision_function(x)  # flip so higher = more anomalous
        return {
            "anomaly_score": [float(v) for v in score],
            "is_outlier": [1 if v == -1 else 0 for v in pred],
        }


@dataclass(slots=True, frozen=True)
class EllipticEnvelopeArgs(_BaseArgs):
    """Arguments for the elliptic_envelope function."""

    contamination: Annotated[float, Arg("contamination", default=0.1, doc="Expected proportion of outliers (0-0.5).")]
    random_state: Annotated[int, Arg("random_state", default=0, doc="Random seed.")]


class EllipticEnvelopeFn(_BufferingTransform[EllipticEnvelopeArgs]):
    """Elliptic Envelope (Gaussian) outlier detection; emits an anomaly score and flag per row."""

    FunctionArguments: ClassVar[type] = EllipticEnvelopeArgs

    class Meta:
        """VGI metadata for the elliptic_envelope function."""

        name = "elliptic_envelope"
        description = "Elliptic Envelope (Gaussian) outlier detection; emits an anomaly score and flag per row"
        categories = ["outlier-detection"]
        examples = _ex("elliptic_envelope", "contamination => 0.1")
        tags = {"vgi.columns_md": _OUTLIER_MD}

    output_fields = staticmethod(_outlier_fields)  # type: ignore[assignment]

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        """Run the scikit-learn computation and return the output columns."""
        from sklearn.covariance import EllipticEnvelope

        model = EllipticEnvelope(contamination=args.contamination, random_state=args.random_state)
        pred = model.fit_predict(x)
        score = -model.decision_function(x)  # flip so higher = more anomalous
        return {
            "anomaly_score": [float(v) for v in score],
            "is_outlier": [1 if v == -1 else 0 for v in pred],
        }


# ===========================================================================
# Manifold learning (non-linear embeddings -> component_1..k, like pca)
# ===========================================================================


@dataclass(slots=True, frozen=True)
class TsneArgs(ComponentsArgs):
    """Arguments for the tsne function."""

    perplexity: Annotated[float, Arg("perplexity", default=30.0, doc="Nearest-neighbour count proxy (< n_samples).")]
    random_state: Annotated[int, Arg("random_state", default=0, doc="Random seed.")]


class TsneFn(_BufferingTransform[TsneArgs]):
    """t-SNE non-linear embedding (great for 2-D visualization)."""

    FunctionArguments: ClassVar[type] = TsneArgs

    class Meta:
        """VGI metadata for the tsne function."""

        name = "tsne"
        description = "t-SNE non-linear embedding (great for 2-D visualization)"
        categories = ["manifold", "dimensionality-reduction"]
        examples = _ex("tsne", "n_components => 2")
        tags = {"vgi.columns_md": _COMPONENTS_MD}

    output_fields = staticmethod(_component_fields)

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        """Run the scikit-learn computation and return the output columns."""
        from sklearn.manifold import TSNE

        k = _effective_components(args.n_components, len(feature_names))
        emb = TSNE(n_components=k, perplexity=args.perplexity, random_state=args.random_state).fit_transform(x)
        return {f"component_{i + 1}": emb[:, i].tolist() for i in range(k)}


@dataclass(slots=True, frozen=True)
class IsomapArgs(ComponentsArgs):
    """Arguments for the isomap function."""

    n_neighbors: Annotated[int, Arg("n_neighbors", default=5, doc="Number of neighbours per point.")]


class IsomapFn(_BufferingTransform[IsomapArgs]):
    """Isomap non-linear embedding (geodesic distance preservation)."""

    FunctionArguments: ClassVar[type] = IsomapArgs

    class Meta:
        """VGI metadata for the isomap function."""

        name = "isomap"
        description = "Isomap non-linear embedding (geodesic distance preservation)"
        categories = ["manifold", "dimensionality-reduction"]
        examples = _ex("isomap", "n_components => 2")
        tags = {"vgi.columns_md": _COMPONENTS_MD}

    output_fields = staticmethod(_component_fields)

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        """Run the scikit-learn computation and return the output columns."""
        from sklearn.manifold import Isomap

        k = _effective_components(args.n_components, len(feature_names))
        emb = Isomap(n_components=k, n_neighbors=args.n_neighbors).fit_transform(x)
        return {f"component_{i + 1}": emb[:, i].tolist() for i in range(k)}


@dataclass(slots=True, frozen=True)
class SpectralEmbeddingArgs(ComponentsArgs):
    """Arguments for the spectral_embedding function."""

    random_state: Annotated[int, Arg("random_state", default=0, doc="Random seed.")]


class SpectralEmbeddingFn(_BufferingTransform[SpectralEmbeddingArgs]):
    """Spectral (Laplacian eigenmaps) non-linear embedding."""

    FunctionArguments: ClassVar[type] = SpectralEmbeddingArgs

    class Meta:
        """VGI metadata for the spectral_embedding function."""

        name = "spectral_embedding"
        description = "Spectral (Laplacian eigenmaps) non-linear embedding"
        categories = ["manifold", "dimensionality-reduction"]
        examples = _ex("spectral_embedding", "n_components => 2")
        tags = {"vgi.columns_md": _COMPONENTS_MD}

    output_fields = staticmethod(_component_fields)

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        """Run the scikit-learn computation and return the output columns."""
        from sklearn.manifold import SpectralEmbedding

        k = _effective_components(args.n_components, len(feature_names))
        emb = SpectralEmbedding(n_components=k, random_state=args.random_state).fit_transform(x)
        return {f"component_{i + 1}": emb[:, i].tolist() for i in range(k)}


@dataclass(slots=True, frozen=True)
class MdsArgs(ComponentsArgs):
    """Arguments for the mds function."""

    random_state: Annotated[int, Arg("random_state", default=0, doc="Random seed.")]


class MdsFn(_BufferingTransform[MdsArgs]):
    """Multidimensional scaling embedding (distance preservation)."""

    FunctionArguments: ClassVar[type] = MdsArgs

    class Meta:
        """VGI metadata for the mds function."""

        name = "mds"
        description = "Multidimensional scaling embedding (distance preservation)"
        categories = ["manifold", "dimensionality-reduction"]
        examples = _ex("mds", "n_components => 2")
        tags = {"vgi.columns_md": _COMPONENTS_MD}

    output_fields = staticmethod(_component_fields)

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        """Run the scikit-learn computation and return the output columns."""
        from sklearn.manifold import MDS

        k = _effective_components(args.n_components, len(feature_names))
        emb = MDS(n_components=k, random_state=args.random_state).fit_transform(x)
        return {f"component_{i + 1}": emb[:, i].tolist() for i in range(k)}


# ===========================================================================
# Categorical encoders (string input)
#
# Unlike the transforms above, the input columns here are *strings*. fit/predict
# auto-encode categoricals internally, but these expose the encoding directly so
# you can inspect or materialize it. ``ordinal_encoder`` keeps a fixed width
# (one integer code column per feature); ``one_hot_encoder`` emits long format
# (one row per active cell), sidestepping the data-dependent-width limit.
# ===========================================================================


class _EncoderBase[TArgs: _BaseArgs](SinkBuffer[TArgs, DrainState]):
    """Buffer the whole input, encode string features once in finalize, stream out."""

    @classmethod
    def feature_names(cls, input_schema: pa.Schema, id_col: str) -> list[str]:
        return [n for n in input_schema.names if n != id_col]

    @classmethod
    def output_schema(cls, input_schema: pa.Schema, feats: list[str], args: Any) -> pa.Schema:
        raise NotImplementedError

    @classmethod
    def encode(cls, table: pa.Table, feats: list[str], args: Any) -> dict[str, list[Any]]:
        raise NotImplementedError

    @classmethod
    def on_bind(cls, params: BindParams[TArgs]) -> BindResponse:
        ins = params.bind_call.input_schema
        assert ins is not None
        feats = cls.feature_names(ins, params.args.id)
        return BindResponse(output_schema=cls.output_schema(ins, feats, params.args))

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[TArgs]) -> DrainState:
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[TArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        if state.done:
            out.finish()
            return
        state.done = True

        ins = input_schema_of(params)
        feats = cls.feature_names(ins, params.args.id)
        table = cls.buffered_table(params, ins)
        out_schema = params.output_schema
        if table is None or table.num_rows == 0:
            out.emit(pa.RecordBatch.from_pydict({n: [] for n in out_schema.names}, schema=out_schema))
            return
        out.emit(pa.RecordBatch.from_pydict(cls.encode(table, feats, params.args), schema=out_schema))


def _str_matrix(table: pa.Table, feats: list[str]) -> np.ndarray:
    """Object matrix of the feature columns, NULL preserved as NaN, else string."""
    recs = rows_from_table(table, feats)
    return np.array([[float("nan") if v is None else str(v) for v in row] for row in recs], dtype=object)


class OrdinalEncoderFn(_EncoderBase[_BaseArgs]):
    """Encode each categorical (string) column as integer codes (one column per feature)."""

    FunctionArguments: ClassVar[type] = _BaseArgs

    class Meta:
        """VGI metadata for the ordinal_encoder function."""

        name = "ordinal_encoder"
        description = "Encode each categorical (string) column as integer codes (one column per feature)"
        categories = ["preprocessing", "encoding"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM sklearn.ordinal_encoder("
                    "(SELECT sample_id, target_name FROM sklearn.iris()), id => 'sample_id')"
                ),
                description="Encode the iris species name as an integer code",
            )
        ]
        tags = {"vgi.columns_md": _scaler_md("BIGINT", "the integer code (-1 = missing/unseen).")}

    @classmethod
    def output_schema(cls, input_schema: pa.Schema, feats: list[str], args: Any) -> pa.Schema:
        """Build the full output schema (optional id column plus encoded columns)."""
        fields: list[pa.Field] = []
        if args.id:
            fields.append(input_schema.field(args.id))
        fields.extend(
            sfield(f, pa.int64(), f"Integer code for {f} (-1 = missing/unseen).", nullable=False) for f in feats
        )
        return pa.schema(fields)

    @classmethod
    def encode(cls, table: pa.Table, feats: list[str], args: Any) -> dict[str, list[Any]]:
        """Encode the buffered string features and return the output columns."""
        from sklearn.preprocessing import OrdinalEncoder

        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)
        codes = enc.fit_transform(_str_matrix(table, feats))
        cols: dict[str, list[Any]] = {}
        if args.id:
            cols[args.id] = table.column(args.id).to_pylist()
        for j, f in enumerate(feats):
            cols[f] = [int(v) for v in codes[:, j]]
        return cols


class OneHotEncoderFn(_EncoderBase[_BaseArgs]):
    """One-hot encode categorical (string) columns in long format (id, feature, category, value)."""

    FunctionArguments: ClassVar[type] = _BaseArgs

    class Meta:
        """VGI metadata for the one_hot_encoder function."""

        name = "one_hot_encoder"
        description = "One-hot encode categorical (string) columns in long format (id, feature, category, value)"
        categories = ["preprocessing", "encoding"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM sklearn.one_hot_encoder("
                    "(SELECT sample_id, target_name FROM sklearn.iris()), id => 'sample_id')"
                ),
                description="One-hot encode the iris species name (one row per active category)",
            )
        ]
        tags = {
            "vgi.columns_md": columns_md_rows(
                [
                    ("feature", "VARCHAR", "Source feature column."),
                    ("category", "VARCHAR", "Category value that is set for this row."),
                    ("value", "DOUBLE", "Indicator value (always 1.0 for an active cell)."),
                ],
                note=_ID_NOTE,
            )
        }

    @classmethod
    def output_schema(cls, input_schema: pa.Schema, feats: list[str], args: Any) -> pa.Schema:
        """Build the full output schema (optional id column plus encoded columns)."""
        fields: list[pa.Field] = []
        if args.id:
            fields.append(input_schema.field(args.id))
        fields.extend(
            [
                sfield("feature", pa.string(), "Source feature column.", nullable=False),
                sfield("category", pa.string(), "Category value that is set for this row.", nullable=False),
                sfield("value", pa.float64(), "Indicator value (always 1.0 for an active cell).", nullable=False),
            ]
        )
        return pa.schema(fields)

    @classmethod
    def encode(cls, table: pa.Table, feats: list[str], args: Any) -> dict[str, list[Any]]:
        """Encode the buffered string features and return the output columns."""
        from sklearn.preprocessing import OneHotEncoder

        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        matrix_oh = enc.fit_transform(_str_matrix(table, feats)).tocoo()
        # Map each one-hot column back to its (feature, category).
        col_map: list[tuple[str, Any]] = [(feats[fi], cat) for fi, cats in enumerate(enc.categories_) for cat in cats]
        id_vals = table.column(args.id).to_pylist() if args.id else None

        ids: list[Any] = []
        feature_col: list[str] = []
        category_col: list[str] = []
        value_col: list[float] = []
        for r, c, v in zip(matrix_oh.row.tolist(), matrix_oh.col.tolist(), matrix_oh.data.tolist(), strict=True):
            feature, category = col_map[c]
            if isinstance(category, float) and np.isnan(category):  # skip the missing-value category
                continue
            if id_vals is not None:
                ids.append(id_vals[r])
            feature_col.append(feature)
            category_col.append(str(category))
            value_col.append(float(v))

        cols: dict[str, list[Any]] = {}
        if args.id:
            cols[args.id] = ids
        cols["feature"] = feature_col
        cols["category"] = category_col
        cols["value"] = value_col
        return cols


@dataclass(slots=True, frozen=True)
class TargetEncoderArgs(_BaseArgs):
    """Arguments for the target_encoder function."""

    target: Annotated[str, Arg("target", default="", doc="Target column to encode against (required).")]


class TargetEncoderFn(_EncoderBase[TargetEncoderArgs]):
    """Encode each categorical column by its (cross-fitted) mean target value."""

    FunctionArguments: ClassVar[type] = TargetEncoderArgs

    class Meta:
        """VGI metadata for the target_encoder function."""

        name = "target_encoder"
        description = "Encode each categorical column by its (cross-fitted) mean target value"
        categories = ["preprocessing", "encoding"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM sklearn.target_encoder("
                    "(SELECT sample_id, target_name, target FROM sklearn.iris()), "
                    "id := 'sample_id', target := 'target')"
                ),
                description="Encode the species name by its mean target",
            )
        ]
        tags = {"vgi.columns_md": _scaler_md("DOUBLE", "the target-mean encoding.")}

    @staticmethod
    def _feats(schema: pa.Schema, args: TargetEncoderArgs) -> list[str]:
        return [n for n in schema.names if n not in {args.id, args.target} - {""}]

    @classmethod
    def output_schema(cls, input_schema: pa.Schema, feats: list[str], args: TargetEncoderArgs) -> pa.Schema:
        """Build the full output schema (optional id column plus encoded columns)."""
        fields: list[pa.Field] = []
        if args.id:
            fields.append(input_schema.field(args.id))
        fields.extend(
            sfield(f, pa.float64(), f"Target-mean encoding of {f}.", nullable=False)
            for f in cls._feats(input_schema, args)
        )
        return pa.schema(fields)

    @classmethod
    def on_bind(cls, params: BindParams[TargetEncoderArgs]) -> BindResponse:
        """Validate arguments and resolve the output schema at bind time."""
        a = params.args
        if not a.target:
            raise ValueError("target_encoder requires 'target' (the column to encode against)")
        ins = params.bind_call.input_schema
        assert ins is not None
        if a.target not in ins.names:
            raise ValueError(f"target column {a.target!r} not found in input; columns: {', '.join(ins.names)}")
        return BindResponse(output_schema=cls.output_schema(ins, [], a))

    @classmethod
    def encode(cls, table: pa.Table, feats: list[str], args: TargetEncoderArgs) -> dict[str, list[Any]]:
        """Encode the buffered string features and return the output columns."""
        from sklearn.preprocessing import TargetEncoder

        feats = cls._feats(table.schema, args)
        x = _str_matrix(table, feats)
        y = np.asarray(table.column(args.target).to_numpy(zero_copy_only=False))
        encoded = TargetEncoder(target_type="auto").fit_transform(x, y)
        if encoded.shape[1] != len(feats):
            raise ValueError(
                "target_encoder needs a binary or continuous target (multiclass would expand the width); "
                "one-hot the target or encode one class at a time"
            )
        cols: dict[str, list[Any]] = {}
        if args.id:
            cols[args.id] = table.column(args.id).to_pylist()
        for j, f in enumerate(feats):
            cols[f] = [float(v) for v in encoded[:, j]]
        return cols


@dataclass(slots=True, frozen=True)
class PolynomialFeaturesArgs(_BaseArgs):
    """Arguments for the polynomial_features function."""

    degree: Annotated[int, Arg("degree", default=2, doc="Maximum degree of the polynomial features.")]
    interaction_only: Annotated[bool, Arg("interaction_only", default=False, doc="Only products of distinct features.")]
    include_bias: Annotated[
        bool, Arg("include_bias", default=False, doc="Include the constant (all-ones) bias column.")
    ]


def _poly_names(feats: list[str], args: PolynomialFeaturesArgs) -> list[str]:
    """The (sanitized) output column names — deterministic from n_features + degree."""
    from sklearn.preprocessing import PolynomialFeatures

    pf = PolynomialFeatures(
        degree=args.degree, interaction_only=args.interaction_only, include_bias=args.include_bias
    ).fit(np.zeros((1, len(feats))))
    out = []
    for name in pf.get_feature_names_out(feats):
        out.append(str(name).replace("^", "_pow").replace(" ", "_x_") or "bias")
    return out


class PolynomialFeaturesFn(_BufferingTransform[PolynomialFeaturesArgs]):
    """Expand features into polynomial and interaction terms (e.g. a, b -> a, b, a^2, a*b, b^2)."""

    FunctionArguments: ClassVar[type] = PolynomialFeaturesArgs

    class Meta:
        """VGI metadata for the polynomial_features function."""

        name = "polynomial_features"
        description = "Expand features into polynomial and interaction terms (e.g. a, b -> a, b, a^2, a*b, b^2)"
        categories = ["preprocessing", "feature-engineering"]
        examples = _ex("polynomial_features", "degree => 2")
        tags = {
            "vgi.columns_md": columns_md_rows(
                [],
                note=(
                    "One DOUBLE column per generated polynomial/interaction term; names are derived "
                    "from the input features (e.g. `a`, `b`, `a_pow2`, `a_x_b`). " + _ID_NOTE
                ),
            )
        }

    @staticmethod
    def output_fields(feature_names: list[str], args: Any) -> list[pa.Field]:
        """Return the non-id output fields for the given features."""
        return [
            sfield(n, pa.float64(), f"Polynomial term {n}.", nullable=False) for n in _poly_names(feature_names, args)
        ]

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        """Run the scikit-learn computation and return the output columns."""
        from sklearn.preprocessing import PolynomialFeatures

        expanded = PolynomialFeatures(
            degree=args.degree, interaction_only=args.interaction_only, include_bias=args.include_bias
        ).fit_transform(x)
        names = _poly_names(feature_names, args)
        return {names[j]: expanded[:, j].tolist() for j in range(len(names))}


TRANSFORM_FUNCTIONS: list[type] = [
    StandardScalerFn,
    MinMaxScalerFn,
    RobustScalerFn,
    NormalizerFn,
    SimpleImputerFn,
    PcaFn,
    TruncatedSvdFn,
    KMeansFn,
    DbscanFn,
    IsolationForestFn,
    MaxAbsScalerFn,
    PowerTransformerFn,
    QuantileTransformerFn,
    BinarizerFn,
    KBinsDiscretizerFn,
    AgglomerativeFn,
    SpectralClusteringFn,
    MeanShiftFn,
    BirchFn,
    OpticsFn,
    MiniBatchKMeansFn,
    GaussianMixtureFn,
    LocalOutlierFactorFn,
    OneClassSvmFn,
    EllipticEnvelopeFn,
    TsneFn,
    IsomapFn,
    SpectralEmbeddingFn,
    MdsFn,
    OrdinalEncoderFn,
    OneHotEncoderFn,
    TargetEncoderFn,
    PolynomialFeaturesFn,
]
