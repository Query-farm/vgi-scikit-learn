"""scikit-learn datasets exposed as DuckDB table functions.

Two families:

* **Toy datasets** -- ``iris()``, ``wine()``, ``digits()``, ``breast_cancer()``
  (classification) and ``diabetes()`` (regression). Zero-argument, fixed schema.
* **Synthetic generators** -- ``make_classification()``, ``make_regression()``,
  ``make_blobs()``, ``make_moons()``, ``make_circles()``. Their column count
  depends on arguments, so they build their schema in ``on_bind``.

    SELECT * FROM sklearn.iris();
    SELECT * FROM sklearn.make_blobs(n_samples => 300, centers => 4);
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import pyarrow as pa
from sklearn import datasets as skd
from vgi.arguments import Arg
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_function import (
    BindParams,
    ProcessParams,
    TableCardinality,
    TableFunctionGenerator,
    bind_fixed_schema,
    init_single_worker,
)
from vgi_rpc.rpc import OutputCollector

from .schema_utils import NoArgs, dedupe_names, field, snake_case

_RESERVED = {"sample_id", "target", "target_name", "cluster"}


def _feature_labels(bunch: Any, n_features: int) -> list[str]:
    """scikit-learn feature names if present and well-sized, else ``feature_{i}``."""
    names = getattr(bunch, "feature_names", None)
    if names is not None and len(names) == n_features:
        return [str(n) for n in names]
    return [f"feature_{i}" for i in range(n_features)]


def _feature_fields(labels: list[str]) -> list[pa.Field]:
    cols = dedupe_names([snake_case(label) for label in labels])
    return [field(col, pa.float64(), f"Feature: {label}.", nullable=False) for col, label in zip(cols, labels)]


def _classification_schema(labels: list[str], target_names: list[str]) -> pa.Schema:
    """id + float features + integer target + human-readable target name."""
    fields = [field("sample_id", pa.int32(), "Row index within the dataset (0-based).", nullable=False)]
    fields.extend(_feature_fields(labels))
    fields.append(field("target", pa.int32(), "Integer class label.", nullable=False))
    fields.append(
        field(
            "target_name",
            pa.dictionary(pa.int8(), pa.string()),
            f"Human-readable class name (one of: {', '.join(target_names)}).",
            nullable=False,
        )
    )
    return pa.schema(fields)


def _regression_schema(labels: list[str]) -> pa.Schema:
    """id + float features + continuous float target."""
    fields = [field("sample_id", pa.int32(), "Row index within the dataset (0-based).", nullable=False)]
    fields.extend(_feature_fields(labels))
    fields.append(field("target", pa.float64(), "Continuous regression target.", nullable=False))
    return pa.schema(fields)


def _synthetic_schema(n_features: int, target_col: str, target_type: pa.DataType, target_doc: str) -> pa.Schema:
    fields = [field("sample_id", pa.int32(), "Row index within the generated sample (0-based).", nullable=False)]
    fields.extend(_feature_fields([f"feature_{i}" for i in range(n_features)]))
    fields.append(field(target_col, target_type, target_doc, nullable=False))
    return pa.schema(fields)


def _emit_matrix(
    data: Any,
    targets: dict[str, list[Any]],
    schema: pa.Schema,
    out: OutputCollector,
    output_schema: pa.Schema,
) -> None:
    """Emit a feature matrix + named target column(s) as one record batch."""
    n_rows, _ = data.shape
    feature_cols = [name for name in schema.names if name not in _RESERVED]
    columns: dict[str, Any] = {"sample_id": list(range(n_rows))}
    for j, col in enumerate(feature_cols):
        columns[col] = data[:, j].tolist()
    columns.update(targets)
    out.emit(pa.RecordBatch.from_pydict(columns, schema=output_schema))
    out.finish()


# ===========================================================================
# Toy datasets
# ===========================================================================


class _ToyDataset(TableFunctionGenerator[NoArgs]):
    """Base for fixed-schema toy datasets. Subclasses set BUNCH + REGRESSION + Meta."""

    BUNCH: ClassVar[Any]
    REGRESSION: ClassVar[bool] = False

    @classmethod
    def cardinality(cls, params: BindParams[NoArgs]) -> TableCardinality:
        n = int(cls.BUNCH.data.shape[0])
        return TableCardinality(estimate=n, max=n)

    @classmethod
    def process(cls, params: ProcessParams[NoArgs], state: None, out: OutputCollector) -> None:
        bunch = cls.BUNCH
        target = bunch.target
        if cls.REGRESSION:
            targets = {"target": [float(t) for t in target]}
        else:
            names = list(bunch.target_names)
            targets = {
                "target": [int(t) for t in target],
                "target_name": [str(names[int(t)]) for t in target],
            }
        _emit_matrix(bunch.data, targets, cls.FIXED_SCHEMA, out, params.output_schema)


def _classification_toy(loader: Any) -> tuple[Any, pa.Schema]:
    bunch = loader()
    labels = _feature_labels(bunch, bunch.data.shape[1])
    return bunch, _classification_schema(labels, [str(n) for n in bunch.target_names])


def _regression_toy(loader: Any) -> tuple[Any, pa.Schema]:
    bunch = loader()
    labels = _feature_labels(bunch, bunch.data.shape[1])
    return bunch, _regression_schema(labels)


_IRIS, _IRIS_SCHEMA = _classification_toy(skd.load_iris)
_WINE, _WINE_SCHEMA = _classification_toy(skd.load_wine)
_DIGITS, _DIGITS_SCHEMA = _classification_toy(skd.load_digits)
_CANCER, _CANCER_SCHEMA = _classification_toy(skd.load_breast_cancer)
_DIABETES, _DIABETES_SCHEMA = _regression_toy(skd.load_diabetes)


@init_single_worker
@bind_fixed_schema
class IrisFunction(_ToyDataset):
    """Fisher's iris dataset: 150 flowers, 4 measurements, 3 species."""

    BUNCH = _IRIS
    FIXED_SCHEMA: ClassVar[pa.Schema] = _IRIS_SCHEMA

    class Meta:
        name = "iris"
        description = "Fisher's iris dataset (150 samples, 4 features, 3 species)"
        categories = ["datasets", "classification"]
        projection_pushdown = True
        examples = [
            FunctionExample(sql="SELECT * FROM sklearn.iris()", description="Load the full iris dataset"),
            FunctionExample(
                sql="SELECT target_name, avg(petal_length_cm) FROM sklearn.iris() GROUP BY target_name",
                description="Mean petal length per species",
            ),
        ]


@init_single_worker
@bind_fixed_schema
class WineFunction(_ToyDataset):
    """Wine recognition dataset: 178 samples, 13 chemical features, 3 cultivars."""

    BUNCH = _WINE
    FIXED_SCHEMA: ClassVar[pa.Schema] = _WINE_SCHEMA

    class Meta:
        name = "wine"
        description = "Wine recognition dataset (178 samples, 13 features, 3 classes)"
        categories = ["datasets", "classification"]
        projection_pushdown = True
        examples = [FunctionExample(sql="SELECT * FROM sklearn.wine()", description="Load the wine dataset")]


@init_single_worker
@bind_fixed_schema
class DigitsFunction(_ToyDataset):
    """Handwritten digits: 1797 samples, 64 pixel features (8x8), 10 classes."""

    BUNCH = _DIGITS
    FIXED_SCHEMA: ClassVar[pa.Schema] = _DIGITS_SCHEMA

    class Meta:
        name = "digits"
        description = "Handwritten digits (1797 samples, 64 pixel features, 10 classes)"
        categories = ["datasets", "classification"]
        projection_pushdown = True
        examples = [FunctionExample(sql="SELECT * FROM sklearn.digits()", description="Load the digits dataset")]


@init_single_worker
@bind_fixed_schema
class BreastCancerFunction(_ToyDataset):
    """Breast cancer Wisconsin: 569 samples, 30 features, 2 classes."""

    BUNCH = _CANCER
    FIXED_SCHEMA: ClassVar[pa.Schema] = _CANCER_SCHEMA

    class Meta:
        name = "breast_cancer"
        description = "Breast cancer Wisconsin diagnostic (569 samples, 30 features, 2 classes)"
        categories = ["datasets", "classification"]
        projection_pushdown = True
        examples = [
            FunctionExample(sql="SELECT * FROM sklearn.breast_cancer()", description="Load the breast cancer dataset")
        ]


_CALIFORNIA_SCHEMA = _regression_schema(
    ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
)


@init_single_worker
@bind_fixed_schema
class CaliforniaHousingFunction(TableFunctionGenerator[NoArgs]):
    """California housing regression: 20640 districts, 8 features, median house value.

    Downloaded from scikit-learn on first use and cached under the standard
    scikit-learn data home (``~/scikit_learn_data`` or ``SCIKIT_LEARN_DATA``).
    """

    FIXED_SCHEMA: ClassVar[pa.Schema] = _CALIFORNIA_SCHEMA

    class Meta:
        name = "california_housing"
        description = "California housing prices (20640 districts, 8 features, regression)"
        categories = ["datasets", "regression", "fetched"]
        projection_pushdown = True
        examples = [
            FunctionExample(
                sql="SELECT * FROM sklearn.california_housing()",
                description="Load the California housing dataset (downloads on first use)",
            )
        ]

    @classmethod
    def cardinality(cls, params: BindParams[NoArgs]) -> TableCardinality:
        return TableCardinality(estimate=20640, max=20640)

    @classmethod
    def process(cls, params: ProcessParams[NoArgs], state: None, out: OutputCollector) -> None:
        bunch = skd.fetch_california_housing()
        _emit_matrix(bunch.data, {"target": [float(t) for t in bunch.target]}, cls.FIXED_SCHEMA, out, params.output_schema)


@init_single_worker
@bind_fixed_schema
class DiabetesFunction(_ToyDataset):
    """Diabetes regression: 442 samples, 10 baseline features, continuous target."""

    BUNCH = _DIABETES
    REGRESSION = True
    FIXED_SCHEMA: ClassVar[pa.Schema] = _DIABETES_SCHEMA

    class Meta:
        name = "diabetes"
        description = "Diabetes progression regression (442 samples, 10 features)"
        categories = ["datasets", "regression"]
        projection_pushdown = True
        examples = [FunctionExample(sql="SELECT * FROM sklearn.diabetes()", description="Load the diabetes dataset")]


# ===========================================================================
# Synthetic generators (schema depends on arguments -> custom on_bind)
# ===========================================================================


@dataclass(slots=True, frozen=True)
class MakeClassificationArgs:
    n_samples: Annotated[int, Arg("n_samples", default=100, doc="Number of samples to generate.")]
    n_features: Annotated[int, Arg("n_features", default=20, doc="Total number of features.")]
    n_informative: Annotated[int, Arg("n_informative", default=2, doc="Number of informative features.")]
    n_classes: Annotated[int, Arg("n_classes", default=2, doc="Number of target classes.")]
    random_state: Annotated[int, Arg("random_state", default=0, doc="Random seed for reproducibility.")]


@init_single_worker
class MakeClassificationFunction(TableFunctionGenerator[MakeClassificationArgs]):
    """Generate a random n-class classification problem."""

    class Meta:
        name = "make_classification"
        description = "Generate a synthetic classification dataset"
        categories = ["datasets", "synthetic", "classification"]
        projection_pushdown = True
        examples = [
            FunctionExample(
                sql="SELECT * FROM sklearn.make_classification(n_samples => 500, n_features => 5, n_classes => 3)",
                description="500 rows, 5 features, 3 classes",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[MakeClassificationArgs]) -> BindResponse:
        return BindResponse(
            output_schema=_synthetic_schema(
                params.args.n_features, "target", pa.int32(), "Integer class label."
            )
        )

    @classmethod
    def cardinality(cls, params: BindParams[MakeClassificationArgs]) -> TableCardinality:
        n = params.args.n_samples
        return TableCardinality(estimate=n, max=n)

    @classmethod
    def process(cls, params: ProcessParams[MakeClassificationArgs], state: None, out: OutputCollector) -> None:
        a = params.args
        # sklearn requires n_classes * n_clusters_per_class <= 2**n_informative.
        # Use one cluster per class and raise n_informative just enough (capped
        # at n_features) so any valid n_classes/n_features combination works.
        needed = max(1, math.ceil(math.log2(max(2, a.n_classes))))
        n_informative = min(max(a.n_informative, needed), a.n_features)
        x, y = skd.make_classification(
            n_samples=a.n_samples,
            n_features=a.n_features,
            n_informative=n_informative,
            n_redundant=0,
            n_classes=a.n_classes,
            n_clusters_per_class=1,
            random_state=a.random_state,
        )
        _emit_matrix(x, {"target": [int(v) for v in y]}, params.output_schema, out, params.output_schema)


@dataclass(slots=True, frozen=True)
class MakeRegressionArgs:
    n_samples: Annotated[int, Arg("n_samples", default=100, doc="Number of samples to generate.")]
    n_features: Annotated[int, Arg("n_features", default=20, doc="Total number of features.")]
    n_informative: Annotated[int, Arg("n_informative", default=10, doc="Number of informative features.")]
    noise: Annotated[float, Arg("noise", default=0.0, doc="Std-dev of gaussian noise on the output.")]
    random_state: Annotated[int, Arg("random_state", default=0, doc="Random seed for reproducibility.")]


@init_single_worker
class MakeRegressionFunction(TableFunctionGenerator[MakeRegressionArgs]):
    """Generate a random regression problem."""

    class Meta:
        name = "make_regression"
        description = "Generate a synthetic regression dataset"
        categories = ["datasets", "synthetic", "regression"]
        projection_pushdown = True
        examples = [
            FunctionExample(
                sql="SELECT * FROM sklearn.make_regression(n_samples => 500, n_features => 4, noise => 5.0)",
                description="500 rows, 4 features, noisy target",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[MakeRegressionArgs]) -> BindResponse:
        return BindResponse(
            output_schema=_synthetic_schema(
                params.args.n_features, "target", pa.float64(), "Continuous regression target."
            )
        )

    @classmethod
    def cardinality(cls, params: BindParams[MakeRegressionArgs]) -> TableCardinality:
        n = params.args.n_samples
        return TableCardinality(estimate=n, max=n)

    @classmethod
    def process(cls, params: ProcessParams[MakeRegressionArgs], state: None, out: OutputCollector) -> None:
        a = params.args
        x, y = skd.make_regression(
            n_samples=a.n_samples,
            n_features=a.n_features,
            n_informative=a.n_informative,
            noise=a.noise,
            random_state=a.random_state,
        )
        _emit_matrix(x, {"target": [float(v) for v in y]}, params.output_schema, out, params.output_schema)


@dataclass(slots=True, frozen=True)
class MakeBlobsArgs:
    n_samples: Annotated[int, Arg("n_samples", default=100, doc="Number of samples to generate.")]
    n_features: Annotated[int, Arg("n_features", default=2, doc="Number of features per sample.")]
    centers: Annotated[int, Arg("centers", default=3, doc="Number of cluster centers.")]
    cluster_std: Annotated[float, Arg("cluster_std", default=1.0, doc="Std-dev of the clusters.")]
    random_state: Annotated[int, Arg("random_state", default=0, doc="Random seed for reproducibility.")]


@init_single_worker
class MakeBlobsFunction(TableFunctionGenerator[MakeBlobsArgs]):
    """Generate isotropic Gaussian blobs for clustering."""

    class Meta:
        name = "make_blobs"
        description = "Generate Gaussian blobs for clustering"
        categories = ["datasets", "synthetic", "clustering"]
        projection_pushdown = True
        examples = [
            FunctionExample(
                sql="SELECT * FROM sklearn.make_blobs(n_samples => 300, centers => 4)",
                description="300 points in 4 clusters",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[MakeBlobsArgs]) -> BindResponse:
        return BindResponse(
            output_schema=_synthetic_schema(
                params.args.n_features, "cluster", pa.int32(), "Ground-truth cluster index."
            )
        )

    @classmethod
    def cardinality(cls, params: BindParams[MakeBlobsArgs]) -> TableCardinality:
        n = params.args.n_samples
        return TableCardinality(estimate=n, max=n)

    @classmethod
    def process(cls, params: ProcessParams[MakeBlobsArgs], state: None, out: OutputCollector) -> None:
        a = params.args
        x, y = skd.make_blobs(
            n_samples=a.n_samples,
            n_features=a.n_features,
            centers=a.centers,
            cluster_std=a.cluster_std,
            random_state=a.random_state,
        )
        _emit_matrix(x, {"cluster": [int(v) for v in y]}, params.output_schema, out, params.output_schema)


@dataclass(slots=True, frozen=True)
class TwoFeatureArgs:
    n_samples: Annotated[int, Arg("n_samples", default=100, doc="Number of samples to generate.")]
    noise: Annotated[float, Arg("noise", default=0.1, doc="Std-dev of gaussian noise added to the data.")]
    random_state: Annotated[int, Arg("random_state", default=0, doc="Random seed for reproducibility.")]


class _TwoFeatureShape(TableFunctionGenerator[TwoFeatureArgs]):
    """Base for 2-feature binary toy shapes (moons, circles)."""

    @classmethod
    def on_bind(cls, params: BindParams[TwoFeatureArgs]) -> BindResponse:
        return BindResponse(
            output_schema=_synthetic_schema(2, "target", pa.int32(), "Binary class label (0 or 1).")
        )

    @classmethod
    def cardinality(cls, params: BindParams[TwoFeatureArgs]) -> TableCardinality:
        n = params.args.n_samples
        return TableCardinality(estimate=n, max=n)


@init_single_worker
class MakeMoonsFunction(_TwoFeatureShape):
    """Generate two interleaving half-circles (the classic 'moons')."""

    class Meta:
        name = "make_moons"
        description = "Generate two interleaving half-moons (2 features, binary)"
        categories = ["datasets", "synthetic", "classification"]
        projection_pushdown = True
        examples = [
            FunctionExample(
                sql="SELECT * FROM sklearn.make_moons(n_samples => 200, noise => 0.1)",
                description="200 points in a two-moon shape",
            )
        ]

    @classmethod
    def process(cls, params: ProcessParams[TwoFeatureArgs], state: None, out: OutputCollector) -> None:
        a = params.args
        x, y = skd.make_moons(n_samples=a.n_samples, noise=a.noise, random_state=a.random_state)
        _emit_matrix(x, {"target": [int(v) for v in y]}, params.output_schema, out, params.output_schema)


@init_single_worker
class MakeCirclesFunction(_TwoFeatureShape):
    """Generate a large circle containing a smaller circle in 2D."""

    class Meta:
        name = "make_circles"
        description = "Generate two concentric circles (2 features, binary)"
        categories = ["datasets", "synthetic", "classification"]
        projection_pushdown = True
        examples = [
            FunctionExample(
                sql="SELECT * FROM sklearn.make_circles(n_samples => 200, noise => 0.05)",
                description="200 points in two concentric rings",
            )
        ]

    @classmethod
    def process(cls, params: ProcessParams[TwoFeatureArgs], state: None, out: OutputCollector) -> None:
        a = params.args
        x, y = skd.make_circles(n_samples=a.n_samples, noise=a.noise, random_state=a.random_state)
        _emit_matrix(x, {"target": [int(v) for v in y]}, params.output_schema, out, params.output_schema)


DATASET_FUNCTIONS: list[type] = [
    IrisFunction,
    WineFunction,
    DigitsFunction,
    BreastCancerFunction,
    DiabetesFunction,
    CaliforniaHousingFunction,
    MakeClassificationFunction,
    MakeRegressionFunction,
    MakeBlobsFunction,
    MakeMoonsFunction,
    MakeCirclesFunction,
]
