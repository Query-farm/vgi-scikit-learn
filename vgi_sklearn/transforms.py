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

    SELECT * FROM sklearn.preprocessing.kmeans((SELECT id, x, y FROM points), id => 'id', n_clusters => 3);
    SELECT * FROM sklearn.preprocessing.pca(
      (SELECT * FROM sklearn.datasets.iris()), id => 'sample_id', n_components => 2);
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
            sql=f"SELECT * FROM sklearn.{name}((SELECT * FROM sklearn.datasets.iris()), id => 'sample_id'{args})",
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
        tags = {
            "vgi.result_columns_md": _scaler_md("DOUBLE", "the scaled value."),
            "vgi.doc_llm": (
                "Table function that z-score standardizes every numeric feature: it buffers the whole input "
                "relation `(SELECT ...)` (Arg(0)), fits a `StandardScaler` once, and emits `(value - mean) / "
                "std` per feature. Each non-`id` input column becomes a same-named `DOUBLE` output column "
                "(zero mean, unit variance across rows); name an `id` column to carry a key through unchanged "
                "as the first column. Use it before distance- or gradient-based estimators (SVM, KNN, linear "
                "models, neural nets) that assume comparably scaled inputs; it is sensitive to outliers, so "
                "prefer `robust_scaler` when they dominate."
            ),
            "vgi.doc_md": (
                "**Standard scaler** — center and scale each feature to zero mean and unit variance.\n\n"
                "Buffers the input matrix, fits a single `StandardScaler`, and rewrites every feature as "
                "`(x - mean) / std` over the column.\n\n"
                "- Input: `(SELECT ...)` feature table; optional `id :=` column passed through untouched\n"
                "- Output: one `DOUBLE` column per input feature (same names), standardized across rows\n"
                "- The default preprocessing for scale-sensitive models; outlier-sensitive (a few extremes "
                "shift mean/std) — reach for `robust_scaler` then"
            ),
        }

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
        tags = {
            "vgi.result_columns_md": _scaler_md("DOUBLE", "the scaled value."),
            "vgi.doc_llm": (
                "Table function that linearly rescales every numeric feature into `[0, 1]` via "
                "`(x - min) / (max - min)`. It buffers the input relation `(SELECT ...)` (Arg(0)), fits a "
                "`MinMaxScaler` once, and emits a same-named `DOUBLE` column per feature; the column minimum "
                "maps to 0 and the maximum to 1. Name an `id` column to carry a key through as the first "
                "column. Use it when you want bounded inputs (e.g. for image-like data or models expecting a "
                "fixed range); unlike `standard_scaler` it preserves the original distribution shape but is "
                "very sensitive to outlier min/max values."
            ),
            "vgi.doc_md": (
                "**Min-max scaler** — rescale each feature into the `[0, 1]` range.\n\n"
                "Fits one `MinMaxScaler` over the buffered matrix and maps every feature with "
                "`(x - min) / (max - min)`.\n\n"
                "- Input: `(SELECT ...)` feature table; optional `id :=` passthrough column\n"
                "- Output: one `DOUBLE` column per input feature (same names), bounded in `[0, 1]`\n"
                "- Keeps the distribution shape (just shifts/scales) but a single extreme value sets the "
                "bounds — use `robust_scaler` or `standard_scaler` if outliers are present"
            ),
        }

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
        tags = {
            "vgi.result_columns_md": _scaler_md("DOUBLE", "the scaled value."),
            "vgi.doc_llm": (
                "Table function that scales each numeric feature using outlier-robust statistics: it "
                "subtracts the column median and divides by the interquartile range (IQR, the 75th-25th "
                "percentile spread). It buffers the input relation `(SELECT ...)` (Arg(0)), fits a "
                "`RobustScaler` once, and emits a same-named `DOUBLE` column per feature; an optional `id` "
                "column is carried through as the first column. Prefer it over `standard_scaler` when the "
                "data has heavy tails or extreme values, since median/IQR are barely moved by outliers and "
                "the bulk of the data ends up roughly centered and unit-scaled."
            ),
            "vgi.doc_md": (
                "**Robust scaler** — center and scale with the median and IQR, resisting outliers.\n\n"
                "Fits a single `RobustScaler` over the buffered matrix and rewrites each feature as "
                "`(x - median) / IQR`.\n\n"
                "- Input: `(SELECT ...)` feature table; optional `id :=` passthrough column\n"
                "- Output: one `DOUBLE` column per input feature (same names)\n"
                "- The outlier-resistant alternative to `standard_scaler`: median and interquartile range "
                "shrug off extreme values that would otherwise distort the mean and standard deviation"
            ),
        }

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
        tags = {
            "vgi.result_columns_md": _scaler_md("DOUBLE", "the normalized value."),
            "vgi.doc_llm": (
                "Table function that rescales each *row* (sample) to unit norm — it divides every feature in "
                "a row by that row's norm, so the per-row vector length becomes 1. Pass the feature relation "
                "as `(SELECT ...)` (Arg(0)) and choose the norm with `norm :=` (`'l2'` Euclidean default, "
                "`'l1'` sum of absolute values, or `'max'`); each non-`id` column is emitted as a same-named "
                "`DOUBLE`, with an optional `id` carried through first. Unlike the column-wise scalers this "
                "normalizes across columns within a row, which is what text/TF-IDF vectors and cosine-"
                "similarity workflows expect."
            ),
            "vgi.doc_md": (
                "**Normalizer** — scale each row (not column) to unit length.\n\n"
                "Divides every value in a row by that row's norm so each sample vector has magnitude 1.\n\n"
                "- Input: `(SELECT ...)` feature table; optional `id :=` passthrough column\n"
                "- `norm :=` `'l2'` (Euclidean, default), `'l1'` (sum of |values|), or `'max'`\n"
                "- Output: one `DOUBLE` column per input feature (same names), row-normalized\n"
                "- Row-wise (the only transform here that works across columns), standard before cosine "
                "similarity and for sparse text/TF-IDF vectors"
            ),
        }

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
        tags = {
            "vgi.result_columns_md": _scaler_md("DOUBLE", "the imputed value."),
            "vgi.doc_llm": (
                "Table function that fills missing (NULL/NaN) feature values with a per-column statistic. It "
                "buffers the input relation `(SELECT ...)` (Arg(0)), fits a `SimpleImputer` once, and replaces "
                "each missing cell with that column's `strategy :=` summary — `'mean'` (default), `'median'`, "
                "`'most_frequent'`, or `'constant'`. Every non-`id` column is emitted as a same-named `DOUBLE` "
                "with no remaining NULLs; an optional `id` is carried through first. Use it as the first step "
                "before models that cannot accept missing values; `'median'` is the robust choice for skewed "
                "numeric features."
            ),
            "vgi.doc_md": (
                "**Simple imputer** — fill NULL/NaN cells with a column statistic.\n\n"
                "Fits one `SimpleImputer` over the buffered matrix and substitutes each missing value with "
                "the chosen per-column summary.\n\n"
                "- Input: `(SELECT ...)` feature table; optional `id :=` passthrough column\n"
                "- `strategy :=` `'mean'` (default), `'median'`, `'most_frequent'`, or `'constant'`\n"
                "- Output: one `DOUBLE` column per input feature (same names), with NULLs replaced\n"
                "- The standard fix before estimators that reject missing data; `'median'` resists skew"
            ),
        }

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
        tags = {
            "vgi.result_columns_md": _COMPONENTS_MD,
            "vgi.doc_llm": (
                "Table function performing principal component analysis (PCA): a linear projection of the "
                "numeric features onto the `n_components :=` orthogonal directions of greatest variance. It "
                "buffers the input relation `(SELECT ...)` (Arg(0)), fits `PCA` once, and emits one `DOUBLE` "
                "`component_<i>` column per kept component (defaults to 2; capped at the feature count), with "
                "an optional `id` carried through first. Use it to compress correlated features, visualize "
                "data in 2-D/3-D, or decorrelate inputs before modeling; standardize the features first "
                "(`standard_scaler`) since PCA is scale-dependent."
            ),
            "vgi.doc_md": (
                "**PCA** — linear dimensionality reduction onto the top-variance directions.\n\n"
                "Fits one `PCA` over the buffered matrix and projects every row onto the leading "
                "`n_components` principal axes.\n\n"
                "- Input: `(SELECT ...)` feature table; optional `id :=` passthrough column\n"
                "- `n_components :=` number of components to keep (default 2, clamped to the feature count)\n"
                "- Output: `component_1 .. component_k` `DOUBLE` columns (projection coordinates)\n"
                "- Scale-sensitive — run `standard_scaler` first; great for compression and 2-D/3-D "
                "visualization of correlated features"
            ),
        }

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
        tags = {
            "vgi.result_columns_md": _COMPONENTS_MD,
            "vgi.doc_llm": (
                "Table function performing truncated singular value decomposition (LSA): a linear "
                "dimensionality reduction to `n_components :=` factors that, unlike PCA, does NOT center the "
                "data, so it works directly on sparse/non-negative matrices like TF-IDF term-document "
                "counts. It buffers the input relation `(SELECT ...)` (Arg(0)), fits `TruncatedSVD` once, and "
                "emits `component_<i>` `DOUBLE` columns (default 2, capped at n_features - 1), optionally "
                "after an `id` passthrough. Use it for latent semantic analysis of text features or whenever "
                "you need PCA-like reduction without mean-centering a sparse matrix."
            ),
            "vgi.doc_md": (
                "**Truncated SVD (LSA)** — uncentered linear reduction, ideal for sparse text features.\n\n"
                "Fits one `TruncatedSVD` over the buffered matrix and projects each row onto the top "
                "singular vectors without subtracting the mean.\n\n"
                "- Input: `(SELECT ...)` feature table; optional `id :=` passthrough column\n"
                "- `n_components :=` factors to keep (default 2, clamped to `n_features - 1`)\n"
                "- Output: `component_1 .. component_k` `DOUBLE` columns\n"
                "- The reducer to pair with TF-IDF/count vectors (it tolerates sparsity); use `pca` when "
                "centering is appropriate"
            ),
        }

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
        tags = {
            "vgi.result_columns_md": _CLUSTER_MD,
            "vgi.doc_llm": (
                "Table function that runs K-Means clustering and labels every input row with its cluster. It "
                "buffers the numeric feature relation `(SELECT ...)` (Arg(0)), partitions the rows into "
                "`n_clusters :=` groups (default 8) by minimizing within-cluster squared distance to the "
                "centroids, and emits an `INTEGER` `cluster` column (0..n_clusters-1); an optional `id` is "
                "carried through first and `random_state :=` makes the result reproducible. Use it for "
                "fast, general-purpose partitioning when you know roughly how many clusters to expect; it "
                "assumes roughly spherical, similarly sized clusters, so standardize features first."
            ),
            "vgi.doc_md": (
                "**K-Means** — partition rows into `k` spherical clusters by centroid distance.\n\n"
                "Buffers the feature matrix and assigns each row the nearest of `n_clusters` learned "
                "centroids (Lloyd's algorithm, `n_init=10`).\n\n"
                "- Input: `(SELECT ...)` feature table; optional `id :=` passthrough column\n"
                "- `n_clusters :=` number of clusters (default 8); `random_state :=` seed for reproducibility\n"
                "- Output: an `INTEGER` `cluster` label per row (`0 .. n_clusters - 1`)\n"
                "- Needs `k` known up front and assumes round, balanced clusters — scale features first; "
                "use `dbscan`/`mean_shift` when the cluster count is unknown"
            ),
        }

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
        tags = {
            "vgi.result_columns_md": _CLUSTER_MD,
            "vgi.doc_llm": (
                "Table function that runs DBSCAN density-based clustering. It buffers the numeric feature "
                "relation `(SELECT ...)` (Arg(0)) and groups rows that are densely packed: a point is a core "
                "point if at least `min_samples :=` neighbours (default 5) lie within distance `eps :=` "
                "(default 0.5), and connected core regions form clusters. It emits an `INTEGER` `cluster` "
                "label per row, with **-1 for noise/outliers**; an optional `id` is carried through first. "
                "Unlike K-Means it discovers the cluster count itself and finds arbitrary shapes, but it is "
                "very sensitive to `eps`/`min_samples` and to feature scaling."
            ),
            "vgi.doc_md": (
                "**DBSCAN** — density clustering that discovers cluster count and flags noise.\n\n"
                "Groups rows reachable through dense neighbourhoods (>= `min_samples` points within `eps`) "
                "and marks the rest as outliers.\n\n"
                "- Input: `(SELECT ...)` feature table; optional `id :=` passthrough column\n"
                "- `eps :=` neighbourhood radius (default 0.5); `min_samples :=` points for a dense core "
                "(default 5)\n"
                "- Output: an `INTEGER` `cluster` label per row, **`-1` = noise**\n"
                "- Finds arbitrary-shaped clusters without knowing `k`, but the result hinges on tuning "
                "`eps`/`min_samples`; scale features first"
            ),
        }

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
        tags = {
            "vgi.result_columns_md": _OUTLIER_MD,
            "vgi.doc_llm": (
                "Table function that runs Isolation Forest outlier detection. It buffers the numeric feature "
                "relation `(SELECT ...)` (Arg(0)), fits an ensemble of random trees that isolate points by "
                "recursive splits, and scores each row by how few splits it took to isolate (anomalies "
                "isolate quickly). It emits two columns per row: `anomaly_score` (`DOUBLE`, higher = more "
                "anomalous) and `is_outlier` (`INTEGER`, 1/0); set the expected outlier fraction with "
                "`contamination :=` (default 0.1, 0..0.5) and `random_state :=` for reproducibility, with an "
                "optional `id` first. It scales well to many features/rows and makes no distributional "
                "assumptions, the go-to general-purpose anomaly detector."
            ),
            "vgi.doc_md": (
                "**Isolation Forest** — tree-based anomaly detection that scales to wide data.\n\n"
                "Fits an ensemble that isolates points with random splits; points isolated in few splits "
                "score as anomalies.\n\n"
                "- Input: `(SELECT ...)` feature table; optional `id :=` passthrough column\n"
                "- `contamination :=` expected outlier proportion (default 0.1, range 0-0.5); "
                "`random_state :=` seed\n"
                "- Output: `anomaly_score` `DOUBLE` (higher = more anomalous) and `is_outlier` `INTEGER` "
                "(1/0) per row\n"
                "- Distribution-free and efficient on high-dimensional data; the default choice for "
                "general anomaly detection"
            ),
        }

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
        tags = {
            "vgi.result_columns_md": _scaler_md("DOUBLE", "the scaled value."),
            "vgi.doc_llm": (
                "Table function that scales each numeric feature by its maximum absolute value, mapping it "
                "into `[-1, 1]` without shifting (zeros stay zero, so sparsity is preserved). It buffers the "
                "input relation `(SELECT ...)` (Arg(0)), fits a `MaxAbsScaler` once, and emits a same-named "
                "`DOUBLE` column per feature; an optional `id` is carried through first. Use it for sparse or "
                "already-centered data where you want bounded magnitudes but must not destroy zeros — it is "
                "the sparse-friendly cousin of `minmax_scaler`."
            ),
            "vgi.doc_md": (
                "**MaxAbs scaler** — divide each feature by its max absolute value into `[-1, 1]`.\n\n"
                "Fits one `MaxAbsScaler` over the buffered matrix; it only scales (no centering), so zeros "
                "remain zero.\n\n"
                "- Input: `(SELECT ...)` feature table; optional `id :=` passthrough column\n"
                "- Output: one `DOUBLE` column per input feature (same names), in `[-1, 1]`\n"
                "- Sparsity-preserving (unlike `minmax_scaler`/`standard_scaler`); ideal for sparse or "
                "sign-centered data; still sensitive to the single largest magnitude"
            ),
        }

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
        tags = {
            "vgi.result_columns_md": _scaler_md("DOUBLE", "the transformed value."),
            "vgi.doc_llm": (
                "Table function that applies a monotone power transform to make each numeric feature more "
                "Gaussian (stabilizing variance and reducing skew). It buffers the input relation "
                "`(SELECT ...)` (Arg(0)), fits a `PowerTransformer` per feature, and emits same-named "
                "`DOUBLE` columns; choose `method :=` `'yeo-johnson'` (default, handles any sign) or "
                "`'box-cox'` (strictly positive inputs only). The output is also standardized to zero mean "
                "and unit variance. An optional `id` is carried through first. Use it when skewed features "
                "hurt models that prefer normal-ish inputs (linear/Gaussian methods)."
            ),
            "vgi.doc_md": (
                "**Power transformer** — Gaussianize skewed features (Yeo-Johnson / Box-Cox).\n\n"
                "Fits a per-feature power transform that reduces skew and stabilizes variance, then "
                "standardizes the result.\n\n"
                "- Input: `(SELECT ...)` feature table; optional `id :=` passthrough column\n"
                "- `method :=` `'yeo-johnson'` (default, any sign) or `'box-cox'` (positive values only)\n"
                "- Output: one `DOUBLE` column per input feature (same names), near-normal and standardized\n"
                "- Use when heavy skew degrades models assuming normality; `quantile_transformer` is the "
                "more aggressive, fully nonparametric alternative"
            ),
        }

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
        tags = {
            "vgi.result_columns_md": _scaler_md("DOUBLE", "the transformed value."),
            "vgi.doc_llm": (
                "Table function that maps each numeric feature onto a target distribution using its rank/"
                "quantiles — a nonparametric transform that forces every feature to the same shape and "
                "collapses outliers onto the tails. It buffers the input relation `(SELECT ...)` (Arg(0)), "
                "fits a `QuantileTransformer` per feature, and emits same-named `DOUBLE` columns; pick "
                "`output_distribution :=` `'uniform'` (default, values in [0,1]) or `'normal'`, and "
                "`n_quantiles :=` the resolution (default 1000, capped at the row count). An optional `id` is "
                "carried through first. Use it for strongly non-Gaussian features where `power_transformer` "
                "is not enough; it can distort linear relationships."
            ),
            "vgi.doc_md": (
                "**Quantile transformer** — rank-map each feature to a uniform or normal shape.\n\n"
                "Fits a per-feature quantile mapping that reshapes the distribution and squashes outliers "
                "onto the tails.\n\n"
                "- Input: `(SELECT ...)` feature table; optional `id :=` passthrough column\n"
                "- `output_distribution :=` `'uniform'` (default) or `'normal'`; `n_quantiles :=` resolution "
                "(default 1000, capped at n_samples)\n"
                "- Output: one `DOUBLE` column per input feature (same names)\n"
                "- The strongest, fully nonparametric reshaping (more aggressive than `power_transformer`); "
                "very outlier-robust but can warp inter-feature relationships"
            ),
        }

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
        tags = {
            "vgi.result_columns_md": _scaler_md("DOUBLE", "the 0/1 thresholded value."),
            "vgi.doc_llm": (
                "Table function that binarizes each numeric feature against a fixed cutoff: values strictly "
                "greater than `threshold :=` (default 0.0) become 1.0, all others 0.0. It buffers the input "
                "relation `(SELECT ...)` (Arg(0)) and emits a same-named `DOUBLE` column per feature holding "
                "the 0/1 indicator, with an optional `id` carried through first. Use it to turn counts or "
                "scores into simple presence/absence flags (e.g. bag-of-words occurrence) before models that "
                "want binary inputs."
            ),
            "vgi.doc_md": (
                "**Binarizer** — threshold each feature to a 0/1 indicator.\n\n"
                "Maps every value to `1.0` if it exceeds `threshold`, otherwise `0.0`, column by column.\n\n"
                "- Input: `(SELECT ...)` feature table; optional `id :=` passthrough column\n"
                "- `threshold :=` cutoff above which a value maps to 1 (default 0.0)\n"
                "- Output: one `DOUBLE` column per input feature (same names), valued 0.0 or 1.0\n"
                "- Turns counts/scores into presence flags; for multi-level bucketing use "
                "`kbins_discretizer` instead"
            ),
        }

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
        tags = {
            "vgi.result_columns_md": _scaler_md("BIGINT", "the bin index."),
            "vgi.doc_llm": (
                "Table function that discretizes continuous features into integer bins (ordinal encoding). "
                "It buffers the input relation `(SELECT ...)` (Arg(0)), fits a `KBinsDiscretizer` per "
                "feature, and replaces each value with its `BIGINT` bin index (0..n_bins-1). Set "
                "`n_bins :=` (default 5) and the edge `strategy :=` `'quantile'` (default, equal-count bins), "
                "`'uniform'` (equal-width), or `'kmeans'` (1-D clustered edges); an optional `id` is carried "
                "through first. Use it to bucket numeric features for tree splits, to model nonlinear "
                "effects via bins, or to coarsen noisy continuous values."
            ),
            "vgi.doc_md": (
                "**K-bins discretizer** — bucket continuous features into ordinal integer bins.\n\n"
                "Fits a per-feature binning and replaces each value with its zero-based bin index.\n\n"
                "- Input: `(SELECT ...)` feature table; optional `id :=` passthrough column\n"
                "- `n_bins :=` bins per feature (default 5); `strategy :=` `'quantile'` (default), "
                "`'uniform'`, or `'kmeans'`\n"
                "- Output: one `BIGINT` column per input feature (same names), valued `0 .. n_bins - 1`\n"
                "- Coarsens noisy numerics and lets linear models capture nonlinear, per-bin effects"
            ),
        }

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
        tags = {
            "vgi.result_columns_md": _CLUSTER_MD,
            "vgi.doc_llm": (
                "Table function that runs hierarchical (agglomerative) clustering: it buffers the numeric "
                "feature relation `(SELECT ...)` (Arg(0)) and repeatedly merges the two closest groups "
                "bottom-up until `n_clusters :=` (default 2) remain, emitting an `INTEGER` `cluster` label "
                "per row. The merge rule is set by `linkage :=` `'ward'` (default, minimizes variance — "
                "needs Euclidean features), `'complete'`, `'average'`, or `'single'`; an optional `id` is "
                "carried through first. Use it when you want a deterministic hierarchy or non-spherical "
                "clusters of a known count; it is O(n^2) so it suits small/medium tables, not huge ones."
            ),
            "vgi.doc_md": (
                "**Agglomerative clustering** — bottom-up hierarchical merging to `k` clusters.\n\n"
                "Buffers the feature matrix and repeatedly fuses the closest groups until `n_clusters` "
                "remain.\n\n"
                "- Input: `(SELECT ...)` feature table; optional `id :=` passthrough column\n"
                "- `n_clusters :=` final cluster count (default 2); `linkage :=` `'ward'` (default), "
                "`'complete'`, `'average'`, or `'single'`\n"
                "- Output: an `INTEGER` `cluster` label per row\n"
                "- Deterministic and good for non-spherical shapes, but O(n^2) — best on small/medium "
                "tables; `'ward'` assumes Euclidean features"
            ),
        }

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
        tags = {
            "vgi.result_columns_md": _CLUSTER_MD,
            "vgi.doc_llm": (
                "Table function that runs spectral clustering: it buffers the numeric feature relation "
                "`(SELECT ...)` (Arg(0)), builds an affinity graph between rows, embeds them with the "
                "eigenvectors of the graph Laplacian, and clusters that embedding into `n_clusters :=` "
                "groups (default 2), emitting an `INTEGER` `cluster` label per row. `random_state :=` makes "
                "the embedding reproducible and an optional `id` is carried through first. Use it for "
                "clusters that are connected but not convex (rings, spirals, manifold structure) where "
                "K-Means fails; it is compute/memory heavy (O(n^2) affinities) so keep the table modest."
            ),
            "vgi.doc_md": (
                "**Spectral clustering** — cluster the graph-Laplacian embedding, for non-convex shapes.\n\n"
                "Builds a row-affinity graph, embeds via its Laplacian eigenvectors, then clusters that "
                "low-dimensional embedding.\n\n"
                "- Input: `(SELECT ...)` feature table; optional `id :=` passthrough column\n"
                "- `n_clusters :=` cluster count (default 2); `random_state :=` seed for reproducibility\n"
                "- Output: an `INTEGER` `cluster` label per row\n"
                "- Captures connected, non-convex clusters (rings/spirals) that defeat K-Means; expensive "
                "(O(n^2)) so use on small/medium tables"
            ),
        }

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
        tags = {
            "vgi.result_columns_md": _CLUSTER_MD,
            "vgi.doc_llm": (
                "Table function that runs mean-shift clustering, which discovers the number of clusters "
                "automatically by shifting each point toward the local density mode within a kernel window. "
                "It buffers the numeric feature relation `(SELECT ...)` (Arg(0)) and emits an `INTEGER` "
                "`cluster` label per row; set the kernel `bandwidth :=` window size or leave it 0 (default) "
                "to estimate it from the data, with an optional `id` carried through first. Use it when you "
                "do not know `k` and clusters are blob-like; the bandwidth controls granularity and it is "
                "O(n^2), so it fits small/medium tables."
            ),
            "vgi.doc_md": (
                "**Mean shift** — density-mode clustering that picks the cluster count for you.\n\n"
                "Shifts every point uphill to its local density peak within a kernel window; points sharing "
                "a peak form a cluster.\n\n"
                "- Input: `(SELECT ...)` feature table; optional `id :=` passthrough column\n"
                "- `bandwidth :=` kernel window (default 0 = estimate automatically from the data)\n"
                "- Output: an `INTEGER` `cluster` label per row\n"
                "- No `k` required; bandwidth tunes granularity. O(n^2) — keep tables modest; use "
                "`dbscan` for arbitrary shapes or `kmeans` when `k` is known"
            ),
        }

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
        tags = {
            "vgi.result_columns_md": _CLUSTER_MD,
            "vgi.doc_llm": (
                "Table function that runs BIRCH clustering, a memory-efficient method for large datasets "
                "that incrementally summarizes rows into a tree of compact subclusters before a final "
                "clustering. It buffers the numeric feature relation `(SELECT ...)` (Arg(0)) and emits an "
                "`INTEGER` `cluster` label per row; set the final group count with `n_clusters :=` (default "
                "3) and the subcluster absorption radius with `threshold :=` (default 0.5; smaller = more, "
                "finer subclusters), with an optional `id` carried through first. Use it when the table is "
                "large and you want a single fast pass; it assumes roughly spherical clusters like K-Means."
            ),
            "vgi.doc_md": (
                "**BIRCH** — memory-efficient clustering via an incremental subcluster tree.\n\n"
                "Summarizes rows into compact subclusters in one pass, then clusters those summaries — "
                "scaling to large tables.\n\n"
                "- Input: `(SELECT ...)` feature table; optional `id :=` passthrough column\n"
                "- `n_clusters :=` final clusters (default 3); `threshold :=` subcluster radius (default "
                "0.5; smaller = finer)\n"
                "- Output: an `INTEGER` `cluster` label per row\n"
                "- The large-data option; assumes spherical clusters like K-Means, with `threshold` "
                "trading detail for memory"
            ),
        }

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
        tags = {
            "vgi.result_columns_md": _CLUSTER_MD,
            "vgi.doc_llm": (
                "Table function that runs OPTICS density clustering, a generalization of DBSCAN that orders "
                "points by reachability and so handles clusters of *varying* density without a single fixed "
                "`eps`. It buffers the numeric feature relation `(SELECT ...)` (Arg(0)) and emits an "
                "`INTEGER` `cluster` label per row, with **-1 for noise/outliers**; `min_samples :=` (default "
                "5) sets how many neighbours define a dense core, and an optional `id` is carried through "
                "first. Use it instead of `dbscan` when clusters differ in density; it auto-discovers the "
                "cluster count but is O(n^2), so keep tables modest."
            ),
            "vgi.doc_md": (
                "**OPTICS** — density clustering for clusters of varying density (DBSCAN, no fixed `eps`).\n\n"
                "Orders points by reachability distance and extracts clusters across density levels, "
                "marking sparse points as noise.\n\n"
                "- Input: `(SELECT ...)` feature table; optional `id :=` passthrough column\n"
                "- `min_samples :=` points needed for a dense core (default 5)\n"
                "- Output: an `INTEGER` `cluster` label per row, **`-1` = noise**\n"
                "- Handles mixed-density clusters that a single-`eps` `dbscan` misses; discovers the count "
                "itself but is O(n^2)"
            ),
        }

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
        tags = {
            "vgi.result_columns_md": _CLUSTER_MD,
            "vgi.doc_llm": (
                "Table function that runs mini-batch K-Means: the same centroid-distance clustering as "
                "`kmeans` but fitted on small random mini-batches, trading a little cluster quality for much "
                "faster convergence on large datasets. It buffers the numeric feature relation "
                "`(SELECT ...)` (Arg(0)), partitions rows into `n_clusters :=` groups (default 8), and emits "
                "an `INTEGER` `cluster` label per row; `random_state :=` makes it reproducible and an "
                "optional `id` is carried through first. Use it as the scalable substitute for `kmeans` when "
                "the table is large; standardize features first and expect the same spherical-cluster "
                "assumptions."
            ),
            "vgi.doc_md": (
                "**Mini-batch K-Means** — the scalable, mini-batch variant of K-Means.\n\n"
                "Updates centroids on small random batches, converging far faster than full K-Means with "
                "only a slight quality cost.\n\n"
                "- Input: `(SELECT ...)` feature table; optional `id :=` passthrough column\n"
                "- `n_clusters :=` number of clusters (default 8); `random_state :=` seed\n"
                "- Output: an `INTEGER` `cluster` label per row (`0 .. n_clusters - 1`)\n"
                "- The large-data drop-in for `kmeans`; same need for known `k`, scaled features, and "
                "round/balanced clusters"
            ),
        }

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
        tags = {
            "vgi.result_columns_md": _CLUSTER_MD,
            "vgi.doc_llm": (
                "Table function that fits a Gaussian Mixture Model (soft, probabilistic clustering): it "
                "models the data as a mix of `n_components :=` Gaussian blobs (default 2) and assigns each "
                "row to its most-likely component. It buffers the numeric feature relation `(SELECT ...)` "
                "(Arg(0)) and emits an `INTEGER` `cluster` label per row; `covariance_type :=` "
                "(`'full'` default, `'tied'`, `'diag'`, `'spherical'`) controls the cluster shape "
                "flexibility, `random_state :=` makes it reproducible, and an optional `id` is carried "
                "through first. Use it when clusters are elliptical or overlapping — unlike K-Means it allows "
                "stretched, rotated covariances rather than only spheres."
            ),
            "vgi.doc_md": (
                "**Gaussian mixture** — probabilistic clustering with elliptical, overlapping components.\n\n"
                "Fits a mixture of Gaussians (EM) and labels each row by its most probable component.\n\n"
                "- Input: `(SELECT ...)` feature table; optional `id :=` passthrough column\n"
                "- `n_components :=` number of components (default 2); `covariance_type :=` `'full'` "
                "(default), `'tied'`, `'diag'`, or `'spherical'`; `random_state :=` seed\n"
                "- Output: an `INTEGER` `cluster` label per row (the most likely component)\n"
                "- More flexible than `kmeans` (covariances can stretch/rotate), suiting elliptical or "
                "overlapping clusters"
            ),
        }

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
        tags = {
            "vgi.result_columns_md": _OUTLIER_MD,
            "vgi.doc_llm": (
                "Table function that runs Local Outlier Factor (LOF) detection: it scores each row by how "
                "much sparser its local neighbourhood is than its neighbours', so it flags points that are "
                "anomalous *relative to their local density* even when global density varies. It buffers the "
                "numeric feature relation `(SELECT ...)` (Arg(0)) and emits `anomaly_score` (`DOUBLE`, higher "
                "= more anomalous) and `is_outlier` (`INTEGER` 1/0) per row; `n_neighbors :=` (default 20) "
                "sets the locality and `contamination :=` (default 0.1) the expected outlier fraction, with "
                "an optional `id` first. Prefer it over `isolation_forest` when clusters have differing "
                "densities."
            ),
            "vgi.doc_md": (
                "**Local Outlier Factor (LOF)** — density-relative anomaly detection.\n\n"
                "Compares each point's local density to its neighbours' so outliers are judged against their "
                "own region, not a global threshold.\n\n"
                "- Input: `(SELECT ...)` feature table; optional `id :=` passthrough column\n"
                "- `n_neighbors :=` neighbourhood size (default 20); `contamination :=` expected outlier "
                "share (default 0.1, range 0-0.5)\n"
                "- Output: `anomaly_score` `DOUBLE` (higher = more anomalous) and `is_outlier` `INTEGER` "
                "(1/0) per row\n"
                "- Best when density varies across the data (local view), where `isolation_forest`'s "
                "global view can miss local anomalies"
            ),
        }

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
        tags = {
            "vgi.result_columns_md": _OUTLIER_MD,
            "vgi.doc_llm": (
                "Table function that runs One-Class SVM novelty/outlier detection: it learns a (kernelized) "
                "boundary enclosing the bulk of the data and flags points falling outside it. It buffers the "
                "numeric feature relation `(SELECT ...)` (Arg(0)) and emits `anomaly_score` (`DOUBLE`, higher "
                "= more anomalous) and `is_outlier` (`INTEGER` 1/0) per row. `nu :=` (default 0.5) upper-"
                "bounds the outlier fraction and lower-bounds the support-vector count, `kernel :=` "
                "(`'rbf'` default, `'linear'`, `'poly'`, `'sigmoid'`) and `gamma :=` (`'scale'`/`'auto'`) "
                "shape the boundary, with an optional `id` first. Use it for nonlinear/complex boundaries; it "
                "is sensitive to scaling and `nu`, so standardize features."
            ),
            "vgi.doc_md": (
                "**One-Class SVM** — kernel boundary around the normal data; flag what lies outside.\n\n"
                "Learns a frontier enclosing most points and scores rows by their signed distance from it.\n\n"
                "- Input: `(SELECT ...)` feature table; optional `id :=` passthrough column\n"
                "- `nu :=` outlier fraction upper bound (default 0.5); `kernel :=` `'rbf'` (default), "
                "`'linear'`, `'poly'`, `'sigmoid'`; `gamma :=` `'scale'`/`'auto'`\n"
                "- Output: `anomaly_score` `DOUBLE` (higher = more anomalous) and `is_outlier` `INTEGER` "
                "(1/0) per row\n"
                "- Captures complex nonlinear boundaries; very sensitive to feature scaling and `nu`, so "
                "standardize first"
            ),
        }

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
        tags = {
            "vgi.result_columns_md": _OUTLIER_MD,
            "vgi.doc_llm": (
                "Table function that runs Elliptic Envelope outlier detection: it assumes the inliers are "
                "Gaussian, robustly fits their mean and covariance, and flags rows lying far outside the "
                "resulting confidence ellipse (by Mahalanobis distance). It buffers the numeric feature "
                "relation `(SELECT ...)` (Arg(0)) and emits `anomaly_score` (`DOUBLE`, higher = more "
                "anomalous) and `is_outlier` (`INTEGER` 1/0) per row; `contamination :=` (default 0.1) sets "
                "the expected outlier fraction and `random_state :=` makes the robust fit reproducible, with "
                "an optional `id` first. Best when the normal data is roughly unimodal and elliptical; it "
                "struggles with multi-cluster or highly non-Gaussian data (use `isolation_forest`/`lof` "
                "there)."
            ),
            "vgi.doc_md": (
                "**Elliptic Envelope** — Gaussian (Mahalanobis) outlier detection.\n\n"
                "Robustly fits a covariance ellipse to the inliers and flags points far outside it.\n\n"
                "- Input: `(SELECT ...)` feature table; optional `id :=` passthrough column\n"
                "- `contamination :=` expected outlier proportion (default 0.1, range 0-0.5); "
                "`random_state :=` seed\n"
                "- Output: `anomaly_score` `DOUBLE` (higher = more anomalous) and `is_outlier` `INTEGER` "
                "(1/0) per row\n"
                "- Ideal for unimodal, roughly elliptical normal data; switch to `isolation_forest` or "
                "`local_outlier_factor` for multi-cluster or non-Gaussian data"
            ),
        }

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
        tags = {
            "vgi.doc_llm": (
                "Table function computing a t-SNE non-linear embedding: it buffers the numeric feature "
                "relation `(SELECT ...)` (Arg(0)) and maps each row into a low-dimensional space "
                "(`n_components :=`, default 2) that preserves *local* neighbourhood structure, making it "
                "excellent for visualizing clusters in 2-D/3-D. It emits `component_<i>` `DOUBLE` columns "
                "with an optional `id` first; `perplexity :=` (default 30, must be < n_samples) balances "
                "local vs. global structure and `random_state :=` fixes the layout. Note the embedding is "
                "for visualization only — distances/areas are not meaningful, it cannot transform new rows, "
                "and it is slow on large tables."
            ),
            "vgi.doc_md": (
                "**t-SNE** — non-linear embedding that excels at 2-D cluster visualization.\n\n"
                "Buffers the feature matrix and lays rows out so local neighbourhoods are preserved, "
                "revealing cluster structure.\n\n"
                "- Input: `(SELECT ...)` feature table; optional `id :=` passthrough column\n"
                "- `n_components :=` embedding dims (default 2); `perplexity :=` neighbour-count proxy "
                "(default 30, < n_samples); `random_state :=` seed\n"
                "- Output: `component_1 .. component_k` `DOUBLE` columns (layout coordinates)\n"
                "- Visualization only: global distances are not meaningful, no out-of-sample transform, "
                "and slow on large data — use `pca`/`umap`-style methods for reusable reduction"
            ),
            "vgi.result_columns_md": _COMPONENTS_MD,
        }

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
        tags = {
            "vgi.doc_llm": (
                "Table function computing an Isomap non-linear embedding: it buffers the numeric feature "
                "relation `(SELECT ...)` (Arg(0)), builds a `n_neighbors :=` (default 5) nearest-neighbour "
                "graph, and embeds rows into `n_components :=` dimensions (default 2) so that *geodesic* "
                "(along-the-manifold) distances are preserved. It emits `component_<i>` `DOUBLE` columns with "
                "an optional `id` first. Use it to unroll curved manifolds (e.g. a swiss-roll) where Euclidean "
                "distance is misleading; unlike t-SNE it preserves global structure, but the neighbour graph "
                "must be connected and the result is sensitive to `n_neighbors`."
            ),
            "vgi.doc_md": (
                "**Isomap** — manifold embedding that preserves geodesic (along-surface) distances.\n\n"
                "Builds a neighbour graph, measures distances along it, and embeds rows so that "
                "global manifold structure is kept.\n\n"
                "- Input: `(SELECT ...)` feature table; optional `id :=` passthrough column\n"
                "- `n_components :=` embedding dims (default 2); `n_neighbors :=` graph neighbours per point "
                "(default 5)\n"
                "- Output: `component_1 .. component_k` `DOUBLE` columns\n"
                "- Unrolls curved manifolds where straight-line distance lies; preserves global structure "
                "better than t-SNE but needs a connected graph and good `n_neighbors`"
            ),
            "vgi.result_columns_md": _COMPONENTS_MD,
        }

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
        tags = {
            "vgi.doc_llm": (
                "Table function computing a spectral (Laplacian eigenmaps) embedding: it buffers the numeric "
                "feature relation `(SELECT ...)` (Arg(0)), forms an affinity graph between rows, and projects "
                "them using the lowest-frequency eigenvectors of the graph Laplacian into `n_components :=` "
                "dimensions (default 2). It emits `component_<i>` `DOUBLE` columns with an optional `id` "
                "first; `random_state :=` fixes the eigensolver. The embedding keeps nearby points close, so "
                "it is well suited to clustering on a manifold and is the dimensionality-reduction analogue "
                "of `spectral_clustering`; it is O(n^2) in affinities, so keep tables modest."
            ),
            "vgi.doc_md": (
                "**Spectral embedding (Laplacian eigenmaps)** — non-linear embedding from graph "
                "eigenvectors.\n\n"
                "Builds a row-affinity graph and projects rows onto the lowest eigenvectors of its "
                "Laplacian, keeping neighbours close.\n\n"
                "- Input: `(SELECT ...)` feature table; optional `id :=` passthrough column\n"
                "- `n_components :=` embedding dims (default 2); `random_state :=` seed\n"
                "- Output: `component_1 .. component_k` `DOUBLE` columns\n"
                "- The embedding counterpart of `spectral_clustering`; preserves local manifold structure "
                "but is O(n^2) — best on small/medium tables"
            ),
            "vgi.result_columns_md": _COMPONENTS_MD,
        }

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
        tags = {
            "vgi.doc_llm": (
                "Table function computing a multidimensional scaling (MDS) embedding: it buffers the numeric "
                "feature relation `(SELECT ...)` (Arg(0)) and places each row in `n_components :=` "
                "dimensions (default 2) so that pairwise distances in the embedding match the original "
                "high-dimensional distances as closely as possible. It emits `component_<i>` `DOUBLE` columns "
                "with an optional `id` first; `random_state :=` fixes the (iterative, stress-minimizing) "
                "layout. Use it when faithfully reproducing the *global* distance structure matters more "
                "than local detail; it is O(n^2) per iteration, so it suits small/medium tables and gives no "
                "out-of-sample transform."
            ),
            "vgi.doc_md": (
                "**MDS (multidimensional scaling)** — embed rows preserving pairwise distances.\n\n"
                "Iteratively positions points in low-D so their distances best match the original "
                "high-dimensional distances (stress minimization).\n\n"
                "- Input: `(SELECT ...)` feature table; optional `id :=` passthrough column\n"
                "- `n_components :=` embedding dims (default 2); `random_state :=` seed\n"
                "- Output: `component_1 .. component_k` `DOUBLE` columns\n"
                "- Prioritizes global distance fidelity (vs. t-SNE's local focus); O(n^2) per iteration "
                "and no transform for new rows — best on small/medium tables"
            ),
            "vgi.result_columns_md": _COMPONENTS_MD,
        }

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
                    "SELECT * FROM sklearn.preprocessing.ordinal_encoder("
                    "(SELECT sample_id, target_name FROM sklearn.datasets.iris()), id => 'sample_id')"
                ),
                description="Encode the iris species name as an integer code",
            )
        ]
        tags = {
            "vgi.result_columns_md": _scaler_md("BIGINT", "the integer code (-1 = missing/unseen)."),
            "vgi.doc_llm": (
                "Table function that ordinal-encodes categorical *string* columns: each distinct category in "
                "a feature is mapped to an integer code, one fixed-width `BIGINT` column per input feature. "
                "It buffers the string feature relation `(SELECT ...)` (Arg(0)), fits an `OrdinalEncoder`, "
                "and emits a same-named code column (with `-1` for missing/NULL or unseen values), carrying "
                "an optional `id` through first. Note the integer codes imply an (arbitrary) ordering, so "
                "this suits tree-based models; for linear/distance models that would misread the order, use "
                "`one_hot_encoder` instead. (`fit`/`predict` auto-encode internally — this exposes the "
                "encoding as queryable data.)"
            ),
            "vgi.doc_md": (
                "**Ordinal encoder** — map each string category to an integer code.\n\n"
                "Fits one `OrdinalEncoder` and replaces every categorical value with its code, keeping one "
                "column per feature.\n\n"
                "- Input: `(SELECT ...)` table of string features; optional `id :=` passthrough column\n"
                "- Output: one `BIGINT` column per input feature (same names); **`-1` = missing/unseen**\n"
                "- Compact and tree-friendly, but the codes impose an artificial order — prefer "
                "`one_hot_encoder` for linear/distance models"
            ),
        }

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
                    "SELECT * FROM sklearn.preprocessing.one_hot_encoder("
                    "(SELECT sample_id, target_name FROM sklearn.datasets.iris()), id => 'sample_id')"
                ),
                description="One-hot encode the iris species name (one row per active category)",
            )
        ]
        tags = {
            "vgi.result_columns_md": columns_md_rows(
                [
                    ("feature", "VARCHAR", "Source feature column."),
                    ("category", "VARCHAR", "Category value that is set for this row."),
                    ("value", "DOUBLE", "Indicator value (always 1.0 for an active cell)."),
                ],
                note=_ID_NOTE,
            ),
            "vgi.doc_llm": (
                "Table function that one-hot encodes categorical *string* columns, emitting **long format** "
                "(one row per active cell) to sidestep the fixed-output-width limit. It buffers the string "
                "feature relation `(SELECT ...)` (Arg(0)), fits a `OneHotEncoder`, and emits `(feature, "
                "category, value)` rows -- `feature` is the source column, `category` is the active value, "
                "and `value` is always 1.0 -- plus the optional `id` first. Pivot this back to wide in SQL "
                "if you need indicator columns. Unlike `ordinal_encoder` it imposes no false ordering, so it "
                "is the right encoding for linear/distance models; one row is produced per active category "
                "per input row."
            ),
            "vgi.doc_md": (
                "**One-hot encoder** — explode string categories into indicator rows (long format).\n\n"
                "Fits a `OneHotEncoder` and emits one row per active (row, feature, category) cell instead "
                "of a wide column per category, dodging the data-dependent-width limit.\n\n"
                "- Input: `(SELECT ...)` table of string features; optional `id :=` passthrough column\n"
                "- Output columns: `feature` (source column), `category` (the set value), `value` "
                "(always `1.0`)\n"
                "- No artificial ordering (unlike `ordinal_encoder`), so it suits linear/distance models; "
                "pivot to wide in SQL when you need one column per category"
            ),
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
                    "SELECT * FROM sklearn.preprocessing.target_encoder("
                    "(SELECT sample_id, target_name, target FROM sklearn.datasets.iris()), "
                    "id := 'sample_id', target := 'target')"
                ),
                description="Encode the species name by its mean target",
            )
        ]
        tags = {
            "vgi.result_columns_md": _scaler_md("DOUBLE", "the target-mean encoding."),
            "vgi.doc_llm": (
                "Table function that target-encodes categorical *string* columns: each category is replaced "
                "by the mean of the `target :=` column for that category, computed with internal "
                "cross-fitting to avoid leakage. It buffers the relation `(SELECT ...)` (Arg(0)), encodes "
                "every feature except `id :=` and `target :=`, and emits a same-named `DOUBLE` column per "
                "feature holding the target-mean encoding, with an optional `id` first. `target :=` is "
                "**required** and must be binary or continuous -- a multiclass target would expand the output "
                "width and raises an error. Use it for high-cardinality categoricals where one-hot would "
                "explode; it is the supervised, leakage-aware alternative to `ordinal_encoder`."
            ),
            "vgi.doc_md": (
                "**Target encoder** — replace each category with its cross-fitted mean target value.\n\n"
                "Fits a `TargetEncoder` (with internal cross-fitting to curb leakage) and encodes every "
                "categorical feature by its per-category target average.\n\n"
                "- Input: `(SELECT ...)` table of string features plus the target; optional `id :=` "
                "passthrough column\n"
                "- `target :=` **required** target column; must be binary or continuous (multiclass "
                "errors — it would widen the output)\n"
                "- Output: one `DOUBLE` column per encoded feature (same names)\n"
                "- The supervised choice for high-cardinality categoricals where `one_hot_encoder` would "
                "blow up the column count"
            ),
        }

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
            "vgi.result_columns_md": columns_md_rows(
                [],
                note=(
                    "One DOUBLE column per generated polynomial/interaction term; names are derived "
                    "from the input features (e.g. `a`, `b`, `a_pow2`, `a_x_b`). " + _ID_NOTE
                ),
            ),
            "vgi.doc_llm": (
                "Table function that expands numeric features into polynomial and interaction terms up to "
                "`degree :=` (default 2): given features `a, b` it produces `a, b, a^2, a*b, b^2`, etc. It "
                "buffers the relation `(SELECT ...)` (Arg(0)) and emits one `DOUBLE` column per generated "
                "term, named from the inputs (`^` -> `_pow`, ` ` -> `_x_`, e.g. `a_pow2`, `a_x_b`), with an "
                "optional `id` first. `interaction_only :=` true drops pure powers (keeps only cross "
                "products) and `include_bias :=` true adds an all-ones constant column. The output width is "
                "deterministic from the feature count and degree. Use it to let linear models capture "
                "nonlinear and interaction effects; the term count grows fast, so keep `degree` small."
            ),
            "vgi.doc_md": (
                "**Polynomial features** — expand features into powers and interaction terms.\n\n"
                "Generates every monomial up to `degree` (e.g. `a, b -> a, b, a^2, a*b, b^2`) so linear "
                "models can fit nonlinear and interaction effects.\n\n"
                "- Input: `(SELECT ...)` numeric feature table; optional `id :=` passthrough column\n"
                "- `degree :=` max degree (default 2); `interaction_only :=` cross products only (no pure "
                "powers); `include_bias :=` add an all-ones constant column\n"
                "- Output: one `DOUBLE` column per term, names sanitized (`^`->`_pow`, ` `->`_x_`)\n"
                "- Term count explodes with degree and feature count — keep `degree` low to avoid a "
                "feature blowup"
            ),
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
