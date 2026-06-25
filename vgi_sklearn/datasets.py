"""scikit-learn datasets exposed as DuckDB table functions.

Two families:

* **Toy datasets** -- ``iris()``, ``wine()``, ``digits()``, ``breast_cancer()``
  (classification) and ``diabetes()`` (regression). Zero-argument, fixed schema.
* **Synthetic generators** -- ``make_classification()``, ``make_regression()``,
  ``make_blobs()``, ``make_moons()``, ``make_circles()``. Their column count
  depends on arguments, so they build their schema in ``on_bind``.

    SELECT * FROM sklearn.datasets.iris();
    SELECT * FROM sklearn.datasets.make_blobs(n_samples => 300, centers => 4);
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

from .schema_utils import NoArgs, columns_md, columns_md_rows, dedupe_names, field, snake_case

_RESERVED = {"sample_id", "target", "target_name", "cluster"}


def _feature_labels(bunch: Any, n_features: int) -> list[str]:
    """scikit-learn feature names if present and well-sized, else ``feature_{i}``."""
    names = getattr(bunch, "feature_names", None)
    if names is not None and len(names) == n_features:
        return [str(n) for n in names]
    return [f"feature_{i}" for i in range(n_features)]


def _feature_fields(labels: list[str]) -> list[pa.Field]:
    cols = dedupe_names([snake_case(label) for label in labels])
    return [
        field(col, pa.float64(), f"Feature: {label}.", nullable=False) for col, label in zip(cols, labels, strict=False)
    ]


def _classification_schema(labels: list[str], target_names: list[str]) -> pa.Schema:
    """Build id + float features + integer target + human-readable target name schema."""
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
    """Build id + float features + continuous float target schema."""
    fields = [field("sample_id", pa.int32(), "Row index within the dataset (0-based).", nullable=False)]
    fields.extend(_feature_fields(labels))
    fields.append(field("target", pa.float64(), "Continuous regression target.", nullable=False))
    return pa.schema(fields)


def _synthetic_schema(n_features: int, target_col: str, target_type: pa.DataType, target_doc: str) -> pa.Schema:
    """Build id + float features + one named target column schema for a generator."""
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
    FIXED_SCHEMA: ClassVar[pa.Schema]

    @classmethod
    def cardinality(cls, params: BindParams[NoArgs]) -> TableCardinality:
        """Estimate row count as the dataset size (exact for toy bunches)."""
        n = int(cls.BUNCH.data.shape[0])
        return TableCardinality(estimate=n, max=n)

    @classmethod
    def process(cls, params: ProcessParams[NoArgs], state: None, out: OutputCollector) -> None:
        """Emit the toy dataset matrix plus its target (+ class name) columns."""
        bunch = cls.BUNCH
        target = bunch.target
        targets: dict[str, list[Any]]
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
        """VGI metadata for the iris dataset function."""

        name = "iris"
        description = "Fisher's iris dataset (150 samples, 4 features, 3 species)"
        categories = ["datasets", "classification"]
        projection_pushdown = True
        tags = {
            "vgi.result_columns_md": columns_md(_IRIS_SCHEMA),
            "vgi.doc_llm": (
                "Zero-argument table function returning Fisher's classic iris dataset: 150 rows, 4 numeric "
                "flower measurements (`sepal_length_cm`, `sepal_width_cm`, `petal_length_cm`, "
                "`petal_width_cm`), an integer `target` (0/1/2), a human-readable `target_name` species, and a "
                "0-based `sample_id`. Call it as `SELECT * FROM sklearn.datasets.iris()` (no args) — it is the default "
                "demo input for nearly every other function here (fit/predict, scalers, clustering). Use it as "
                "a small, well-separated 3-class benchmark for classification or to smoke-test a pipeline."
            ),
            "vgi.doc_md": (
                "**Iris dataset** — 150 flowers, 4 measurements, 3 species (the canonical ML toy set).\n\n"
                "- `sample_id` INTEGER — 0-based row index\n"
                "- four `*_cm` DOUBLE feature columns (sepal/petal length & width)\n"
                "- `target` INTEGER (0-2) and `target_name` the species name\n"
                "- Takes no arguments; the go-to small, balanced 3-class classification demo and the input "
                "used in most examples here"
            ),
        }
        examples = [
            FunctionExample(sql="SELECT * FROM sklearn.datasets.iris()", description="Load the full iris dataset"),
            FunctionExample(
                sql="SELECT target_name, avg(petal_length_cm) FROM sklearn.datasets.iris() GROUP BY target_name",
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
        """VGI metadata for the wine dataset function."""

        name = "wine"
        description = "Wine recognition dataset (178 samples, 13 features, 3 classes)"
        categories = ["datasets", "classification"]
        projection_pushdown = True
        tags = {
            "vgi.result_columns_md": columns_md(_WINE_SCHEMA),
            "vgi.doc_llm": (
                "Zero-argument table function returning the wine-recognition dataset: 178 rows of 13 numeric "
                "chemical-analysis features (alcohol, malic acid, flavanoids, etc.), an integer `target` "
                "cultivar (0/1/2), a `target_name`, and a 0-based `sample_id`. Call it as "
                "`SELECT * FROM sklearn.datasets.wine()` with no arguments. It is a small all-numeric 3-class "
                "classification benchmark whose features live on very different scales, so it is a good demo "
                "for `standard_scaler` followed by a classifier."
            ),
            "vgi.doc_md": (
                "**Wine dataset** — 178 samples, 13 chemical features, 3 cultivars.\n\n"
                "- `sample_id` INTEGER plus 13 DOUBLE feature columns (chemical measurements)\n"
                "- `target` INTEGER (0-2) and `target_name` the cultivar name\n"
                "- No arguments; an all-numeric multiclass benchmark whose unequal feature scales make it "
                "a natural fit for scaling + classification pipelines"
            ),
        }
        examples = [FunctionExample(sql="SELECT * FROM sklearn.datasets.wine()", description="Load the wine dataset")]


@init_single_worker
@bind_fixed_schema
class DigitsFunction(_ToyDataset):
    """Handwritten digits: 1797 samples, 64 pixel features (8x8), 10 classes."""

    BUNCH = _DIGITS
    FIXED_SCHEMA: ClassVar[pa.Schema] = _DIGITS_SCHEMA

    class Meta:
        """VGI metadata for the digits dataset function."""

        name = "digits"
        description = "Handwritten digits (1797 samples, 64 pixel features, 10 classes)"
        categories = ["datasets", "classification"]
        projection_pushdown = True
        tags = {
            "vgi.result_columns_md": columns_md(_DIGITS_SCHEMA),
            "vgi.doc_llm": (
                "Zero-argument table function returning the handwritten-digits dataset: 1797 rows, 64 pixel "
                "features (the flattened 8x8 grayscale image, values 0-16), an integer `target` digit (0-9), a "
                "`target_name`, and a 0-based `sample_id`. Call it as `SELECT * FROM sklearn.datasets.digits()` with no "  # noqa: E501
                "arguments. With 64 features and 10 classes it is the largest-dimensional toy set here — handy "
                "for demonstrating dimensionality reduction (`pca`) or multiclass classifiers on image-like data."
            ),
            "vgi.doc_md": (
                "**Digits dataset** — 1797 samples, 64 pixel features (8x8 images), 10 classes.\n\n"
                "- `sample_id` INTEGER plus 64 DOUBLE pixel-intensity columns (0-16)\n"
                "- `target` INTEGER (0-9, the digit) and `target_name`\n"
                "- No arguments; the highest-dimensional toy set here — good for showcasing `pca`/SVD "
                "reduction and 10-class classification"
            ),
        }
        examples = [
            FunctionExample(sql="SELECT * FROM sklearn.datasets.digits()", description="Load the digits dataset")
        ]


@init_single_worker
@bind_fixed_schema
class BreastCancerFunction(_ToyDataset):
    """Breast cancer Wisconsin: 569 samples, 30 features, 2 classes."""

    BUNCH = _CANCER
    FIXED_SCHEMA: ClassVar[pa.Schema] = _CANCER_SCHEMA

    class Meta:
        """VGI metadata for the breast cancer dataset function."""

        name = "breast_cancer"
        description = "Breast cancer Wisconsin diagnostic (569 samples, 30 features, 2 classes)"
        categories = ["datasets", "classification"]
        projection_pushdown = True
        tags = {
            "vgi.result_columns_md": columns_md(_CANCER_SCHEMA),
            "vgi.doc_llm": (
                "Zero-argument table function returning the Breast Cancer Wisconsin diagnostic dataset: 569 "
                "rows, 30 numeric features summarizing cell-nucleus measurements (mean/standard-error/worst of "
                "radius, texture, etc.), a binary integer `target` (0 = malignant, 1 = benign), a "
                "`target_name`, and a 0-based `sample_id`. Call it as `SELECT * FROM sklearn.datasets.breast_cancer()` "
                "with no arguments. It is the standard binary-classification benchmark here, useful for "
                "ROC/AUC, precision-recall, and probability-calibration demos."
            ),
            "vgi.doc_md": (
                "**Breast cancer (Wisconsin diagnostic)** — 569 samples, 30 features, 2 classes.\n\n"
                "- `sample_id` INTEGER plus 30 DOUBLE nucleus-measurement columns\n"
                "- `target` INTEGER (0 malignant / 1 benign) and `target_name`\n"
                "- No arguments; the canonical binary-classification toy set — ideal for ROC AUC, "
                "precision/recall, and calibration examples"
            ),
        }
        examples = [
            FunctionExample(
                sql="SELECT * FROM sklearn.datasets.breast_cancer()", description="Load the breast cancer dataset"
            )
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
        """VGI metadata for the California housing dataset function."""

        name = "california_housing"
        description = "California housing prices (20640 districts, 8 features, regression)"
        categories = ["datasets", "regression", "fetched"]
        projection_pushdown = True
        tags = {
            "vgi.result_columns_md": columns_md(_CALIFORNIA_SCHEMA),
            "vgi.doc_llm": (
                "Zero-argument table function returning the California housing regression dataset: 20640 "
                "census-block-group rows with 8 numeric features (`MedInc`, `HouseAge`, `AveRooms`, "
                "`AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`), a continuous `target` (median "
                "house value in $100k), and a 0-based `sample_id`. Call it as "
                "`SELECT * FROM sklearn.datasets.california_housing()` with no arguments; it is **downloaded from "
                "scikit-learn on first use** and cached under the standard data home. The largest dataset here "
                "and the main regression benchmark — good for regressors, geographic feature engineering, and "
                "scaling demos."
            ),
            "vgi.doc_md": (
                "**California housing** — 20640 districts, 8 features, a continuous regression target.\n\n"
                "- `sample_id` INTEGER plus 8 DOUBLE features (income, house age, rooms, location, ...)\n"
                "- `target` DOUBLE — median house value (units of $100,000)\n"
                "- No arguments, but **downloaded and cached on first call**; the largest set here and the "
                "primary regression benchmark"
            ),
        }
        examples = [
            FunctionExample(
                sql="SELECT * FROM sklearn.datasets.california_housing()",
                description="Load the California housing dataset (downloads on first use)",
            )
        ]

    @classmethod
    def cardinality(cls, params: BindParams[NoArgs]) -> TableCardinality:
        """Report the fixed California housing row count."""
        return TableCardinality(estimate=20640, max=20640)

    @classmethod
    def process(cls, params: ProcessParams[NoArgs], state: None, out: OutputCollector) -> None:
        """Fetch (and cache) the dataset and emit it as one batch."""
        bunch = skd.fetch_california_housing()
        _emit_matrix(
            bunch.data, {"target": [float(t) for t in bunch.target]}, cls.FIXED_SCHEMA, out, params.output_schema
        )


@init_single_worker
@bind_fixed_schema
class DiabetesFunction(_ToyDataset):
    """Diabetes regression: 442 samples, 10 baseline features, continuous target."""

    BUNCH = _DIABETES
    REGRESSION = True
    FIXED_SCHEMA: ClassVar[pa.Schema] = _DIABETES_SCHEMA

    class Meta:
        """VGI metadata for the diabetes dataset function."""

        name = "diabetes"
        description = "Diabetes progression regression (442 samples, 10 features)"
        categories = ["datasets", "regression"]
        projection_pushdown = True
        tags = {
            "vgi.result_columns_md": columns_md(_DIABETES_SCHEMA),
            "vgi.doc_llm": (
                "Zero-argument table function returning the diabetes-progression regression dataset: 442 rows, "
                "10 mean-centered, scaled baseline features (age, sex, BMI, blood pressure, and six serum "
                "measurements), a continuous `target` (disease progression one year later), and a 0-based "
                "`sample_id`. Call it as `SELECT * FROM sklearn.datasets.diabetes()` with no arguments. It is a small, "
                "already-standardized regression benchmark — convenient for linear/regularized regressors and "
                "permutation-importance demos without any preprocessing."
            ),
            "vgi.doc_md": (
                "**Diabetes dataset** — 442 samples, 10 baseline features, a continuous regression target.\n\n"
                "- `sample_id` INTEGER plus 10 DOUBLE features (pre-standardized: age, sex, BMI, BP, serum)\n"
                "- `target` DOUBLE — quantitative disease progression after one year\n"
                "- No arguments; a compact regression benchmark whose features are already scaled, so it "
                "drops straight into linear/regularized regressors"
            ),
        }
        examples = [
            FunctionExample(sql="SELECT * FROM sklearn.datasets.diabetes()", description="Load the diabetes dataset")
        ]


# ===========================================================================
# Synthetic generators (schema depends on arguments -> custom on_bind)
# ===========================================================================


@dataclass(slots=True, frozen=True)
class MakeClassificationArgs:
    """Arguments for the make_classification generator."""

    n_samples: Annotated[int, Arg("n_samples", default=100, doc="Number of samples to generate.")]
    n_features: Annotated[int, Arg("n_features", default=20, doc="Total number of features.")]
    n_informative: Annotated[int, Arg("n_informative", default=2, doc="Number of informative features.")]
    n_classes: Annotated[int, Arg("n_classes", default=2, doc="Number of target classes.")]
    random_state: Annotated[int, Arg("random_state", default=0, doc="Random seed for reproducibility.")]


@init_single_worker
class MakeClassificationFunction(TableFunctionGenerator[MakeClassificationArgs]):
    """Generate a random n-class classification problem."""

    class Meta:
        """VGI metadata for the make_classification generator."""

        name = "make_classification"
        description = "Generate a synthetic classification dataset"
        categories = ["datasets", "synthetic", "classification"]
        projection_pushdown = True
        tags = {
            "vgi.result_columns_md": columns_md_rows(
                [
                    ("sample_id", "INTEGER", "Row index within the generated sample (0-based)."),
                    ("target", "INTEGER", "Integer class label."),
                ],
                note="Plus one `feature_<i>` DOUBLE column per feature (count set by `n_features`).",
            ),
            "vgi.doc_llm": (
                "Table function that generates a synthetic n-class classification problem on the fly (wraps "
                "`make_classification`). Set `n_samples :=` rows (default 100), `n_features :=` total feature "
                "count (default 20), `n_informative :=` how many actually drive the label, `n_classes :=` "
                "(default 2), and `random_state :=` for reproducibility; the output schema is built at bind "
                "from `n_features`. It emits a 0-based `sample_id`, one `feature_<i>` `DOUBLE` per feature, and "
                "an integer `target`. Use it to fabricate labeled training data of a chosen size/shape for "
                "testing classifiers, scaling, or CV without a real dataset."
            ),
            "vgi.doc_md": (
                "**make_classification** — generate a random labeled classification dataset.\n\n"
                "- `n_samples :=` rows (default 100); `n_features :=` total features (default 20); "
                "`n_informative :=` label-driving features; `n_classes :=` classes (default 2); "
                "`random_state :=` seed\n"
                "- Output: `sample_id` INTEGER, one `feature_<i>` DOUBLE per feature, `target` INTEGER label\n"
                "- Schema width follows `n_features`; the quick way to manufacture training data for "
                "classifier/pipeline testing"
            ),
        }
        examples = [
            FunctionExample(
                sql="SELECT * FROM sklearn.datasets.make_classification(n_samples => 500, n_features => 5, n_classes => 3)",  # noqa: E501
                description="500 rows, 5 features, 3 classes",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[MakeClassificationArgs]) -> BindResponse:
        """Build the output schema from the requested feature count."""
        return BindResponse(
            output_schema=_synthetic_schema(params.args.n_features, "target", pa.int32(), "Integer class label.")
        )

    @classmethod
    def cardinality(cls, params: BindParams[MakeClassificationArgs]) -> TableCardinality:
        """Report the requested sample count as the exact row count."""
        n = params.args.n_samples
        return TableCardinality(estimate=n, max=n)

    @classmethod
    def process(cls, params: ProcessParams[MakeClassificationArgs], state: None, out: OutputCollector) -> None:
        """Generate the classification dataset and emit it as one batch."""
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
    """Arguments for the make_regression generator."""

    n_samples: Annotated[int, Arg("n_samples", default=100, doc="Number of samples to generate.")]
    n_features: Annotated[int, Arg("n_features", default=20, doc="Total number of features.")]
    n_informative: Annotated[int, Arg("n_informative", default=10, doc="Number of informative features.")]
    noise: Annotated[float, Arg("noise", default=0.0, doc="Std-dev of gaussian noise on the output.")]
    random_state: Annotated[int, Arg("random_state", default=0, doc="Random seed for reproducibility.")]


@init_single_worker
class MakeRegressionFunction(TableFunctionGenerator[MakeRegressionArgs]):
    """Generate a random regression problem."""

    class Meta:
        """VGI metadata for the make_regression generator."""

        name = "make_regression"
        description = "Generate a synthetic regression dataset"
        categories = ["datasets", "synthetic", "regression"]
        projection_pushdown = True
        tags = {
            "vgi.result_columns_md": columns_md_rows(
                [
                    ("sample_id", "INTEGER", "Row index within the generated sample (0-based)."),
                    ("target", "DOUBLE", "Continuous regression target."),
                ],
                note="Plus one `feature_<i>` DOUBLE column per feature (count set by `n_features`).",
            ),
            "vgi.doc_llm": (
                "Table function that generates a synthetic linear regression problem (wraps "
                "`make_regression`): the continuous `target` is a linear combination of the informative "
                "features plus optional Gaussian noise. Set `n_samples :=` rows (default 100), `n_features :=` "
                "total features (default 20), `n_informative :=` how many contribute to the target (default "
                "10), `noise :=` the output noise std-dev (default 0), and `random_state :=`; the schema is "
                "built at bind from `n_features`. Emits a 0-based `sample_id`, `feature_<i>` `DOUBLE` columns, "
                "and a `DOUBLE` `target`. Use it to fabricate regression training data of a chosen size and "
                "signal-to-noise level."
            ),
            "vgi.doc_md": (
                "**make_regression** — generate a random linear-regression dataset.\n\n"
                "- `n_samples :=` rows (default 100); `n_features :=` total (default 20); `n_informative :=` "
                "target-driving features (default 10); `noise :=` output noise std-dev (default 0); "
                "`random_state :=` seed\n"
                "- Output: `sample_id` INTEGER, one `feature_<i>` DOUBLE per feature, `target` DOUBLE\n"
                "- Schema width follows `n_features`; `noise` tunes the signal-to-noise ratio for testing "
                "regressors"
            ),
        }
        examples = [
            FunctionExample(
                sql="SELECT * FROM sklearn.datasets.make_regression(n_samples => 500, n_features => 4, noise => 5.0)",
                description="500 rows, 4 features, noisy target",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[MakeRegressionArgs]) -> BindResponse:
        """Build the output schema from the requested feature count."""
        return BindResponse(
            output_schema=_synthetic_schema(
                params.args.n_features, "target", pa.float64(), "Continuous regression target."
            )
        )

    @classmethod
    def cardinality(cls, params: BindParams[MakeRegressionArgs]) -> TableCardinality:
        """Report the requested sample count as the exact row count."""
        n = params.args.n_samples
        return TableCardinality(estimate=n, max=n)

    @classmethod
    def process(cls, params: ProcessParams[MakeRegressionArgs], state: None, out: OutputCollector) -> None:
        """Generate the regression dataset and emit it as one batch."""
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
    """Arguments for the make_blobs generator."""

    n_samples: Annotated[int, Arg("n_samples", default=100, doc="Number of samples to generate.")]
    n_features: Annotated[int, Arg("n_features", default=2, doc="Number of features per sample.")]
    centers: Annotated[int, Arg("centers", default=3, doc="Number of cluster centers.")]
    cluster_std: Annotated[float, Arg("cluster_std", default=1.0, doc="Std-dev of the clusters.")]
    random_state: Annotated[int, Arg("random_state", default=0, doc="Random seed for reproducibility.")]


@init_single_worker
class MakeBlobsFunction(TableFunctionGenerator[MakeBlobsArgs]):
    """Generate isotropic Gaussian blobs for clustering."""

    class Meta:
        """VGI metadata for the make_blobs generator."""

        name = "make_blobs"
        description = "Generate Gaussian blobs for clustering"
        categories = ["datasets", "synthetic", "clustering"]
        projection_pushdown = True
        tags = {
            "vgi.result_columns_md": columns_md_rows(
                [
                    ("sample_id", "INTEGER", "Row index within the generated sample (0-based)."),
                    ("cluster", "INTEGER", "Ground-truth cluster index."),
                ],
                note="Plus one `feature_<i>` DOUBLE column per feature (count set by `n_features`).",
            ),
            "vgi.doc_llm": (
                "Table function that generates isotropic Gaussian blobs for clustering (wraps `make_blobs`): "
                "points are drawn around `centers :=` cluster centers (default 3) with spread `cluster_std :=` "
                "(default 1.0). Set `n_samples :=` rows (default 100), `n_features :=` dimensions (default 2, "
                "so it plots in 2-D), and `random_state :=`; the schema is built at bind from `n_features`. "
                "Unlike the classification generators it emits a ground-truth `cluster` index (not `target`) "
                "alongside `sample_id` and the `feature_<i>` columns. Use it to demo and validate clustering "
                "(`kmeans`, `dbscan`) where you know the true grouping."
            ),
            "vgi.doc_md": (
                "**make_blobs** — generate Gaussian blobs with known cluster labels.\n\n"
                "- `n_samples :=` rows (default 100); `n_features :=` dims (default 2); `centers :=` blob "
                "count (default 3); `cluster_std :=` spread (default 1.0); `random_state :=` seed\n"
                "- Output: `sample_id` INTEGER, one `feature_<i>` DOUBLE per feature, `cluster` INTEGER "
                "ground-truth label\n"
                "- The ground-truth `cluster` (not a `target`) lets you score clustering quality; the "
                "standard input for `kmeans`/`dbscan` demos"
            ),
        }
        examples = [
            FunctionExample(
                sql="SELECT * FROM sklearn.datasets.make_blobs(n_samples => 300, centers => 4)",
                description="300 points in 4 clusters",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[MakeBlobsArgs]) -> BindResponse:
        """Build the output schema from the requested feature count."""
        return BindResponse(
            output_schema=_synthetic_schema(
                params.args.n_features, "cluster", pa.int32(), "Ground-truth cluster index."
            )
        )

    @classmethod
    def cardinality(cls, params: BindParams[MakeBlobsArgs]) -> TableCardinality:
        """Report the requested sample count as the exact row count."""
        n = params.args.n_samples
        return TableCardinality(estimate=n, max=n)

    @classmethod
    def process(cls, params: ProcessParams[MakeBlobsArgs], state: None, out: OutputCollector) -> None:
        """Generate the Gaussian blobs and emit them as one batch."""
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
    """Arguments for the 2-feature binary toy shapes (moons, circles)."""

    n_samples: Annotated[int, Arg("n_samples", default=100, doc="Number of samples to generate.")]
    noise: Annotated[float, Arg("noise", default=0.1, doc="Std-dev of gaussian noise added to the data.")]
    random_state: Annotated[int, Arg("random_state", default=0, doc="Random seed for reproducibility.")]


class _TwoFeatureShape(TableFunctionGenerator[TwoFeatureArgs]):
    """Base for 2-feature binary toy shapes (moons, circles)."""

    @classmethod
    def on_bind(cls, params: BindParams[TwoFeatureArgs]) -> BindResponse:
        """Build the fixed 2-feature binary output schema."""
        return BindResponse(output_schema=_synthetic_schema(2, "target", pa.int32(), "Binary class label (0 or 1)."))

    @classmethod
    def cardinality(cls, params: BindParams[TwoFeatureArgs]) -> TableCardinality:
        """Report the requested sample count as the exact row count."""
        n = params.args.n_samples
        return TableCardinality(estimate=n, max=n)


@init_single_worker
class MakeMoonsFunction(_TwoFeatureShape):
    """Generate two interleaving half-circles (the classic 'moons')."""

    class Meta:
        """VGI metadata for the make_moons generator."""

        name = "make_moons"
        description = "Generate two interleaving half-moons (2 features, binary)"
        categories = ["datasets", "synthetic", "classification"]
        projection_pushdown = True
        tags = {
            "vgi.result_columns_md": columns_md(
                _synthetic_schema(2, "target", pa.int32(), "Binary class label (0 or 1).")
            ),
            "vgi.doc_llm": (
                "Table function that generates the classic 'two moons': two interleaving half-circles in 2-D "
                "(wraps `make_moons`), a fixed 2-feature binary problem. Set `n_samples :=` rows (default 100), "
                "`noise :=` the Gaussian jitter added (default 0.1), and `random_state :=`. It emits "
                "`sample_id`, exactly two `feature_<i>` `DOUBLE` columns, and a binary integer `target` (0/1). "
                "The classes are not linearly separable, so it is the go-to demo for nonlinear classifiers "
                "(SVM-RBF, KNN, trees) and for showing where linear models fail."
            ),
            "vgi.doc_md": (
                "**make_moons** — two interleaving half-moons (a 2-feature, non-linearly-separable problem).\n\n"
                "- `n_samples :=` rows (default 100); `noise :=` jitter std-dev (default 0.1); "
                "`random_state :=` seed\n"
                "- Output: `sample_id` INTEGER, exactly `feature_0`/`feature_1` DOUBLE, `target` INTEGER (0/1)\n"
                "- Fixed 2-D binary shape that defeats linear models — the standard test for nonlinear "
                "classifiers and decision-boundary demos"
            ),
        }
        examples = [
            FunctionExample(
                sql="SELECT * FROM sklearn.datasets.make_moons(n_samples => 200, noise => 0.1)",
                description="200 points in a two-moon shape",
            )
        ]

    @classmethod
    def process(cls, params: ProcessParams[TwoFeatureArgs], state: None, out: OutputCollector) -> None:
        """Generate the two-moons shape and emit it as one batch."""
        a = params.args
        x, y = skd.make_moons(n_samples=a.n_samples, noise=a.noise, random_state=a.random_state)
        _emit_matrix(x, {"target": [int(v) for v in y]}, params.output_schema, out, params.output_schema)


@init_single_worker
class MakeCirclesFunction(_TwoFeatureShape):
    """Generate a large circle containing a smaller circle in 2D."""

    class Meta:
        """VGI metadata for the make_circles generator."""

        name = "make_circles"
        description = "Generate two concentric circles (2 features, binary)"
        categories = ["datasets", "synthetic", "classification"]
        projection_pushdown = True
        tags = {
            "vgi.result_columns_md": columns_md(
                _synthetic_schema(2, "target", pa.int32(), "Binary class label (0 or 1).")
            ),
            "vgi.doc_llm": (
                "Table function that generates two concentric circles in 2-D — a large ring enclosing a "
                "smaller one (wraps `make_circles`), a fixed 2-feature binary problem. Set `n_samples :=` rows "
                "(default 100), `noise :=` the Gaussian jitter (default 0.1), and `random_state :=`. It emits "
                "`sample_id`, exactly two `feature_<i>` `DOUBLE` columns, and a binary integer `target` (inner "
                "vs. outer ring). The classes form nested rings that no linear boundary can split, so like "
                "`make_moons` it is a benchmark for nonlinear/kernel classifiers."
            ),
            "vgi.doc_md": (
                "**make_circles** — two concentric rings (a 2-feature, non-linearly-separable problem).\n\n"
                "- `n_samples :=` rows (default 100); `noise :=` jitter std-dev (default 0.1); "
                "`random_state :=` seed\n"
                "- Output: `sample_id` INTEGER, exactly `feature_0`/`feature_1` DOUBLE, `target` INTEGER "
                "(inner/outer ring)\n"
                "- Nested-ring binary shape needing a kernel/nonlinear model; pairs with `make_moons` for "
                "decision-boundary demos"
            ),
        }
        examples = [
            FunctionExample(
                sql="SELECT * FROM sklearn.datasets.make_circles(n_samples => 200, noise => 0.05)",
                description="200 points in two concentric rings",
            )
        ]

    @classmethod
    def process(cls, params: ProcessParams[TwoFeatureArgs], state: None, out: OutputCollector) -> None:
        """Generate the two concentric circles and emit them as one batch."""
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
