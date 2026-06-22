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
from vgi.table_buffering_function import OutputCollector, TableBufferingParams
from vgi.table_function import BindParams

from .buffering import DrainState, SinkBuffer, input_schema_of, matrix
from .features import rows_from_table
from .schema_utils import field as sfield


@dataclass(slots=True, frozen=True)
class _BaseArgs:
    data: Annotated[TableInput, Arg(0, doc="Input feature table (each non-id column is a feature)")]
    id: Annotated[str, Arg("id", default="", doc="Optional column to carry through unchanged to the output")]


class _BufferingTransform[TArgs](SinkBuffer[TArgs, DrainState]):
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
            empty = {name: [] for name in params.output_schema.names}
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
    FunctionArguments: ClassVar[type] = _BaseArgs

    class Meta:
        name = "standard_scaler"
        description = "Standardize features to zero mean and unit variance"
        categories = ["preprocessing", "scaling"]
        examples = _ex("standard_scaler")

    output_fields = staticmethod(_scaler_fields)  # type: ignore[assignment]

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        from sklearn.preprocessing import StandardScaler

        z = StandardScaler().fit_transform(x)
        return {f: z[:, j].tolist() for j, f in enumerate(feature_names)}


class MinMaxScalerFn(_BufferingTransform[_BaseArgs]):
    FunctionArguments: ClassVar[type] = _BaseArgs

    class Meta:
        name = "minmax_scaler"
        description = "Scale features to the [0, 1] range"
        categories = ["preprocessing", "scaling"]
        examples = _ex("minmax_scaler")

    output_fields = staticmethod(_scaler_fields)  # type: ignore[assignment]

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        from sklearn.preprocessing import MinMaxScaler

        z = MinMaxScaler().fit_transform(x)
        return {f: z[:, j].tolist() for j, f in enumerate(feature_names)}


class RobustScalerFn(_BufferingTransform[_BaseArgs]):
    FunctionArguments: ClassVar[type] = _BaseArgs

    class Meta:
        name = "robust_scaler"
        description = "Scale features using statistics robust to outliers (median/IQR)"
        categories = ["preprocessing", "scaling"]
        examples = _ex("robust_scaler")

    output_fields = staticmethod(_scaler_fields)  # type: ignore[assignment]

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        from sklearn.preprocessing import RobustScaler

        z = RobustScaler().fit_transform(x)
        return {f: z[:, j].tolist() for j, f in enumerate(feature_names)}


@dataclass(slots=True, frozen=True)
class NormalizerArgs(_BaseArgs):
    norm: Annotated[str, Arg("norm", default="l2", doc="Norm to use: 'l1', 'l2', or 'max'.")]


class NormalizerFn(_BufferingTransform[NormalizerArgs]):
    FunctionArguments: ClassVar[type] = NormalizerArgs

    class Meta:
        name = "normalizer"
        description = "Scale each sample (row) to unit norm"
        categories = ["preprocessing", "scaling"]
        examples = _ex("normalizer", "norm => 'l2'")

    output_fields = staticmethod(_scaler_fields)  # type: ignore[assignment]

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        from sklearn.preprocessing import Normalizer

        z = Normalizer(norm=args.norm).fit_transform(x)
        return {f: z[:, j].tolist() for j, f in enumerate(feature_names)}


@dataclass(slots=True, frozen=True)
class ImputerArgs(_BaseArgs):
    strategy: Annotated[str, Arg("strategy", default="mean", doc="mean, median, most_frequent, or constant.")]


class SimpleImputerFn(_BufferingTransform[ImputerArgs]):
    FunctionArguments: ClassVar[type] = ImputerArgs

    class Meta:
        name = "simple_imputer"
        description = "Fill missing (NULL/NaN) feature values using a column statistic"
        categories = ["preprocessing", "imputation"]
        examples = _ex("simple_imputer", "strategy => 'median'")

    output_fields = staticmethod(_scaler_fields)  # type: ignore[assignment]

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        from sklearn.impute import SimpleImputer

        z = SimpleImputer(strategy=args.strategy).fit_transform(x)
        return {f: z[:, j].tolist() for j, f in enumerate(feature_names)}


# ===========================================================================
# Dimensionality reduction (output = component_1..component_k)
# ===========================================================================


@dataclass(slots=True, frozen=True)
class ComponentsArgs(_BaseArgs):
    n_components: Annotated[int, Arg("n_components", default=2, doc="Number of components to keep.")]


def _effective_components(n_components: int, n_features: int) -> int:
    return max(1, min(n_components, n_features))


def _component_fields(feature_names: list[str], args: Any) -> list[pa.Field]:
    k = _effective_components(args.n_components, len(feature_names))
    return [sfield(f"component_{i + 1}", pa.float64(), f"Projection onto component {i + 1}.") for i in range(k)]


class PcaFn(_BufferingTransform[ComponentsArgs]):
    FunctionArguments: ClassVar[type] = ComponentsArgs

    class Meta:
        name = "pca"
        description = "Principal component analysis (linear dimensionality reduction)"
        categories = ["decomposition", "dimensionality-reduction"]
        examples = _ex("pca", "n_components => 2")

    output_fields = staticmethod(_component_fields)  # type: ignore[assignment]

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        from sklearn.decomposition import PCA

        k = _effective_components(args.n_components, len(feature_names))
        comps = PCA(n_components=k).fit_transform(x)
        return {f"component_{i + 1}": comps[:, i].tolist() for i in range(k)}


class TruncatedSvdFn(_BufferingTransform[ComponentsArgs]):
    FunctionArguments: ClassVar[type] = ComponentsArgs

    class Meta:
        name = "truncated_svd"
        description = "Truncated SVD (LSA) dimensionality reduction"
        categories = ["decomposition", "dimensionality-reduction"]
        examples = _ex("truncated_svd", "n_components => 2")

    @staticmethod
    def _svd_k(n_components: int, n_features: int) -> int:
        # TruncatedSVD requires n_components < n_features.
        return max(1, min(n_components, n_features - 1))

    @staticmethod
    def output_fields(feature_names: list[str], args: Any) -> list[pa.Field]:
        k = TruncatedSvdFn._svd_k(args.n_components, len(feature_names))
        return [sfield(f"component_{i + 1}", pa.float64(), f"SVD component {i + 1}.") for i in range(k)]

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
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
    n_clusters: Annotated[int, Arg("n_clusters", default=8, doc="Number of clusters.")]
    random_state: Annotated[int, Arg("random_state", default=0, doc="Random seed.")]


class KMeansFn(_BufferingTransform[KMeansArgs]):
    FunctionArguments: ClassVar[type] = KMeansArgs

    class Meta:
        name = "kmeans"
        description = "K-Means clustering; emits a cluster label per row"
        categories = ["clustering"]
        examples = _ex("kmeans", "n_clusters => 3")

    output_fields = staticmethod(_cluster_fields)  # type: ignore[assignment]

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        from sklearn.cluster import KMeans

        labels = KMeans(n_clusters=args.n_clusters, random_state=args.random_state, n_init=10).fit_predict(x)
        return {"cluster": [int(v) for v in labels]}


@dataclass(slots=True, frozen=True)
class DbscanArgs(_BaseArgs):
    eps: Annotated[float, Arg("eps", default=0.5, doc="Max neighbourhood distance.")]
    min_samples: Annotated[int, Arg("min_samples", default=5, doc="Min samples to form a dense region.")]


class DbscanFn(_BufferingTransform[DbscanArgs]):
    FunctionArguments: ClassVar[type] = DbscanArgs

    class Meta:
        name = "dbscan"
        description = "DBSCAN density clustering; emits a cluster label per row (-1 = noise)"
        categories = ["clustering"]
        examples = _ex("dbscan", "eps => 0.5")

    output_fields = staticmethod(_cluster_fields)  # type: ignore[assignment]

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
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
    contamination: Annotated[float, Arg("contamination", default=0.1, doc="Expected proportion of outliers (0-0.5).")]
    random_state: Annotated[int, Arg("random_state", default=0, doc="Random seed.")]


class IsolationForestFn(_BufferingTransform[IsolationForestArgs]):
    FunctionArguments: ClassVar[type] = IsolationForestArgs

    class Meta:
        name = "isolation_forest"
        description = "Isolation Forest outlier detection; emits an anomaly score and flag per row"
        categories = ["outlier-detection"]
        examples = _ex("isolation_forest", "contamination => 0.1")

    output_fields = staticmethod(_outlier_fields)  # type: ignore[assignment]

    @classmethod
    def transform(cls, x: np.ndarray, feature_names: list[str], args: Any) -> dict[str, list[Any]]:
        from sklearn.ensemble import IsolationForest

        model = IsolationForest(contamination=args.contamination, random_state=args.random_state)
        pred = model.fit_predict(x)
        score = -model.decision_function(x)  # flip so higher = more anomalous
        return {
            "anomaly_score": [float(v) for v in score],
            "is_outlier": [1 if v == -1 else 0 for v in pred],
        }


# ===========================================================================
# Categorical encoders (string input)
#
# Unlike the transforms above, the input columns here are *strings*. fit/predict
# auto-encode categoricals internally, but these expose the encoding directly so
# you can inspect or materialize it. ``ordinal_encoder`` keeps a fixed width
# (one integer code column per feature); ``one_hot_encoder`` emits long format
# (one row per active cell), sidestepping the data-dependent-width limit.
# ===========================================================================


class _EncoderBase[TArgs](SinkBuffer[TArgs, DrainState]):
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
    FunctionArguments: ClassVar[type] = _BaseArgs

    class Meta:
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

    @classmethod
    def output_schema(cls, input_schema: pa.Schema, feats: list[str], args: Any) -> pa.Schema:
        fields: list[pa.Field] = []
        if args.id:
            fields.append(input_schema.field(args.id))
        fields.extend(
            sfield(f, pa.int64(), f"Integer code for {f} (-1 = missing/unseen).", nullable=False) for f in feats
        )
        return pa.schema(fields)

    @classmethod
    def encode(cls, table: pa.Table, feats: list[str], args: Any) -> dict[str, list[Any]]:
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
    FunctionArguments: ClassVar[type] = _BaseArgs

    class Meta:
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

    @classmethod
    def output_schema(cls, input_schema: pa.Schema, feats: list[str], args: Any) -> pa.Schema:
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
        from sklearn.preprocessing import OneHotEncoder

        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        matrix_oh = enc.fit_transform(_str_matrix(table, feats)).tocoo()
        # Map each one-hot column back to its (feature, category).
        col_map: list[tuple[str, Any]] = [
            (feats[fi], cat) for fi, cats in enumerate(enc.categories_) for cat in cats
        ]
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
    OrdinalEncoderFn,
    OneHotEncoderFn,
]
