"""Metrics that need a whole table at once (buffering functions).

Unlike the column aggregates in ``metrics.py``, these consume a table:

* ``confusion_matrix`` -- long-format counts of (actual, predicted) label pairs.
* ``silhouette_score`` -- one score from a feature matrix + cluster labels.

    SELECT * FROM sklearn.confusion_matrix((SELECT y, yhat FROM preds), actual => 'y', predicted => 'yhat');
    SELECT * FROM sklearn.silhouette_score((SELECT * FROM clustered), label => 'cluster', id => 'id');
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
import pyarrow as pa
from sklearn import metrics as skm
from vgi.arguments import Arg, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_buffering_function import OutputCollector, TableBufferingParams
from vgi.table_function import BindParams

from .buffering import DrainState, SinkBuffer, input_schema_of, matrix
from .schema_utils import field as sfield


@dataclass(slots=True, frozen=True)
class ConfusionMatrixArgs:
    data: Annotated[TableInput, Arg(0, doc="Table containing the actual and predicted label columns.")]
    actual: Annotated[str, Arg("actual", default="actual", doc="Name of the true-label column.")]
    predicted: Annotated[str, Arg("predicted", default="predicted", doc="Name of the predicted-label column.")]


_CONFUSION_SCHEMA = pa.schema(
    [
        sfield("actual", pa.int64(), "True class label.", nullable=False),
        sfield("predicted", pa.int64(), "Predicted class label.", nullable=False),
        sfield("count", pa.int64(), "Number of rows with this (actual, predicted) pair.", nullable=False),
    ]
)


class ConfusionMatrix(SinkBuffer[ConfusionMatrixArgs, DrainState]):
    FunctionArguments: ClassVar[type] = ConfusionMatrixArgs

    class Meta:
        name = "confusion_matrix"
        description = "Confusion matrix in long format: (actual, predicted, count)"
        categories = ["metrics", "classification"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM sklearn.confusion_matrix((SELECT y, yhat FROM preds), "
                    "actual := 'y', predicted := 'yhat')"
                ),
                description="Long-format confusion matrix",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[ConfusionMatrixArgs]) -> BindResponse:
        return BindResponse(output_schema=_CONFUSION_SCHEMA)

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[ConfusionMatrixArgs]
    ) -> DrainState:
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[ConfusionMatrixArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        if state.done:
            out.finish()
            return
        state.done = True

        a = params.args
        table = cls.buffered_table(params, input_schema_of(params))
        if table is None:
            out.emit(
                pa.RecordBatch.from_pydict({"actual": [], "predicted": [], "count": []}, schema=params.output_schema)
            )
            return

        yt = np.rint(np.asarray(table.column(a.actual).to_numpy(zero_copy_only=False), dtype=float)).astype(int)
        yp = np.rint(np.asarray(table.column(a.predicted).to_numpy(zero_copy_only=False), dtype=float)).astype(int)
        counts = Counter(zip(yt.tolist(), yp.tolist(), strict=False))
        rows = sorted(counts.items())
        out.emit(
            pa.RecordBatch.from_pydict(
                {
                    "actual": [int(act) for (act, _), _ in rows],
                    "predicted": [int(pred) for (_, pred), _ in rows],
                    "count": [int(c) for _, c in rows],
                },
                schema=params.output_schema,
            )
        )


@dataclass(slots=True, frozen=True)
class SilhouetteArgs:
    data: Annotated[TableInput, Arg(0, doc="Table of features plus a cluster-label column.")]
    label: Annotated[str, Arg("label", default="cluster", doc="Name of the cluster-label column.")]
    id: Annotated[str, Arg("id", default="", doc="Optional id column to exclude from features.")]


_SILHOUETTE_SCHEMA = pa.schema(
    [sfield("silhouette_score", pa.float64(), "Mean silhouette coefficient over all samples (NULL if undefined).")]
)


class SilhouetteScore(SinkBuffer[SilhouetteArgs, DrainState]):
    FunctionArguments: ClassVar[type] = SilhouetteArgs

    class Meta:
        name = "silhouette_score"
        description = "Mean silhouette coefficient of a clustering (features + label column)"
        categories = ["metrics", "clustering"]
        examples = [
            FunctionExample(
                sql="SELECT * FROM sklearn.silhouette_score((SELECT * FROM clustered), label => 'cluster', id => 'id')",
                description="Silhouette score of a clustering",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[SilhouetteArgs]) -> BindResponse:
        return BindResponse(output_schema=_SILHOUETTE_SCHEMA)

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[SilhouetteArgs]
    ) -> DrainState:
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[SilhouetteArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        if state.done:
            out.finish()
            return
        state.done = True

        a = params.args
        input_schema = input_schema_of(params)
        feats = [n for n in input_schema.names if n not in {a.label, a.id} - {""}]
        table = cls.buffered_table(params, input_schema)

        score: float | None = None
        if table is not None and table.num_rows > 0:
            x = matrix(table, feats)
            labels = np.rint(np.asarray(table.column(a.label).to_numpy(zero_copy_only=False), dtype=float)).astype(int)
            n_labels = len(set(labels.tolist()))
            if 2 <= n_labels <= len(labels) - 1:
                score = float(skm.silhouette_score(x, labels))
        out.emit(pa.RecordBatch.from_pydict({"silhouette_score": [score]}, schema=params.output_schema))


TABLE_METRIC_FUNCTIONS: list[type] = [ConfusionMatrix, SilhouetteScore]
