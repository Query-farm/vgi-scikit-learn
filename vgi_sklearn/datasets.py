"""scikit-learn datasets exposed as DuckDB table functions.

Each toy dataset becomes a zero-argument table function:

    SELECT * FROM sklearn.iris();

Phase 1 adds the remaining toy datasets and the synthetic generators
(``make_classification`` etc.), which derive their schema from arguments.
"""

from __future__ import annotations

from typing import ClassVar

import pyarrow as pa
from sklearn import datasets as skd
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


def _toy_schema(feature_labels: list[str], target_names: list[str]) -> pa.Schema:
    """Build a standard toy-dataset schema: id, features, integer target, label.

    ``feature_labels`` are scikit-learn's raw feature names (e.g.
    ``"sepal length (cm)"``); they are snake_cased and de-duplicated for SQL.
    """
    feature_cols = dedupe_names([snake_case(label) for label in feature_labels])
    fields = [field("sample_id", pa.int32(), "Row index within the dataset (0-based).", nullable=False)]
    for col, label in zip(feature_cols, feature_labels):
        fields.append(field(col, pa.float64(), f"Feature: {label}.", nullable=False))
    fields.append(field("target", pa.int32(), "Integer class/target label.", nullable=False))
    fields.append(
        field(
            "target_name",
            pa.dictionary(pa.int8(), pa.string()),
            f"Human-readable target name (one of: {', '.join(target_names)}).",
            nullable=False,
        )
    )
    return pa.schema(fields)


def _emit_toy(
    bunch: object,
    schema: pa.Schema,
    out: OutputCollector,
    output_schema: pa.Schema,
) -> None:
    """Emit a loaded scikit-learn classification Bunch as one record batch."""
    data = bunch.data  # type: ignore[attr-defined]
    target = bunch.target  # type: ignore[attr-defined]
    target_names = list(bunch.target_names)  # type: ignore[attr-defined]
    n_rows, n_features = data.shape

    feature_cols = schema.names[1 : 1 + n_features]
    columns: dict[str, object] = {"sample_id": list(range(n_rows))}
    for j, col in enumerate(feature_cols):
        columns[col] = data[:, j].tolist()
    columns["target"] = [int(t) for t in target]
    columns["target_name"] = [target_names[int(t)] for t in target]

    out.emit(pa.RecordBatch.from_pydict(columns, schema=output_schema))
    out.finish()


# ---------------------------------------------------------------------------
# Iris
# ---------------------------------------------------------------------------

_IRIS_BUNCH = skd.load_iris()
_IRIS_SCHEMA = _toy_schema(list(_IRIS_BUNCH.feature_names), list(_IRIS_BUNCH.target_names))


@init_single_worker
@bind_fixed_schema
class IrisFunction(TableFunctionGenerator[NoArgs]):
    """The classic Fisher iris dataset: 150 flowers, 4 measurements, 3 species."""

    FIXED_SCHEMA: ClassVar[pa.Schema] = _IRIS_SCHEMA

    class Meta:
        name = "iris"
        description = "Fisher's iris dataset (150 samples, 4 features, 3 species)"
        categories = ["datasets", "classification"]
        projection_pushdown = True
        examples = [
            FunctionExample(
                sql="SELECT * FROM sklearn.iris()",
                description="Load the full iris dataset",
            ),
            FunctionExample(
                sql="SELECT target_name, avg(petal_length) FROM sklearn.iris() GROUP BY target_name",
                description="Mean petal length per species",
            ),
        ]

    @classmethod
    def cardinality(cls, params: BindParams[NoArgs]) -> TableCardinality:
        return TableCardinality(estimate=150, max=150)

    @classmethod
    def process(cls, params: ProcessParams[NoArgs], state: None, out: OutputCollector) -> None:
        _emit_toy(_IRIS_BUNCH, cls.FIXED_SCHEMA, out, params.output_schema)


DATASET_FUNCTIONS: list[type] = [IrisFunction]
