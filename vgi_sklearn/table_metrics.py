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
from vgi.table_buffering_function import TableBufferingParams
from vgi.table_function import BindParams
from vgi_rpc.rpc import OutputCollector

from .buffering import DrainState, SinkBuffer, input_schema_of, matrix
from .schema_utils import columns_md
from .schema_utils import field as sfield


@dataclass(slots=True, frozen=True)
class ConfusionMatrixArgs:
    """Arguments for the confusion_matrix function."""

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
    """Count (actual, predicted) label pairs over a whole table, long format."""

    FunctionArguments: ClassVar[type] = ConfusionMatrixArgs

    class Meta:
        """VGI metadata for the confusion_matrix function."""

        name = "confusion_matrix"
        description = "Confusion matrix in long format: (actual, predicted, count)"
        categories = ["metrics", "classification"]
        tags = {"vgi.columns_md": columns_md(_CONFUSION_SCHEMA)}
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
        """Declare the fixed (actual, predicted, count) output schema."""
        return BindResponse(output_schema=_CONFUSION_SCHEMA)

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[ConfusionMatrixArgs]
    ) -> DrainState:
        """Start with an unfinished drain state."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[ConfusionMatrixArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Tally label pairs over the buffered table and emit the counts."""
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
    """Arguments for the silhouette_score function."""

    data: Annotated[TableInput, Arg(0, doc="Table of features plus a cluster-label column.")]
    label: Annotated[str, Arg("label", default="cluster", doc="Name of the cluster-label column.")]
    id: Annotated[str, Arg("id", default="", doc="Optional id column to exclude from features.")]


_SILHOUETTE_SCHEMA = pa.schema(
    [sfield("silhouette_score", pa.float64(), "Mean silhouette coefficient over all samples (NULL if undefined).")]
)


class SilhouetteScore(SinkBuffer[SilhouetteArgs, DrainState]):
    """Compute the mean silhouette coefficient of a clustering from a whole table."""

    FunctionArguments: ClassVar[type] = SilhouetteArgs

    class Meta:
        """VGI metadata for the silhouette_score function."""

        name = "silhouette_score"
        description = "Mean silhouette coefficient of a clustering (features + label column)"
        categories = ["metrics", "clustering"]
        tags = {"vgi.columns_md": columns_md(_SILHOUETTE_SCHEMA)}
        examples = [
            FunctionExample(
                sql="SELECT * FROM sklearn.silhouette_score((SELECT * FROM clustered), label => 'cluster', id => 'id')",
                description="Silhouette score of a clustering",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[SilhouetteArgs]) -> BindResponse:
        """Declare the single-column silhouette_score output schema."""
        return BindResponse(output_schema=_SILHOUETTE_SCHEMA)

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[SilhouetteArgs]
    ) -> DrainState:
        """Start with an unfinished drain state."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[SilhouetteArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Score the buffered feature matrix against its cluster labels."""
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


# ===========================================================================
# Curve metrics (binary; y_true labels + y_score probabilities -> a curve table)
# ===========================================================================


@dataclass(slots=True, frozen=True)
class _CurveArgs:
    data: Annotated[TableInput, Arg(0, doc="Table with the true-label and score columns.")]
    y_true: Annotated[str, Arg("y_true", default="y_true", doc="Name of the true binary-label column (0/1).")]
    y_score: Annotated[str, Arg("y_score", default="y_score", doc="Name of the score/probability column.")]


class _CurveFunction(SinkBuffer[_CurveArgs, DrainState]):
    """Base for binary curve metrics: buffer (y_true, y_score), compute once."""

    FunctionArguments: ClassVar[type] = _CurveArgs
    OUTPUT_SCHEMA: ClassVar[pa.Schema]

    @classmethod
    def curve(cls, y_true: np.ndarray, y_score: np.ndarray) -> dict[str, list[float | None]]:
        raise NotImplementedError

    @classmethod
    def on_bind(cls, params: BindParams[_CurveArgs]) -> BindResponse:
        return BindResponse(output_schema=cls.OUTPUT_SCHEMA)

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[_CurveArgs]) -> DrainState:
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[_CurveArgs],
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
        if table is None or table.num_rows == 0:
            out.emit(pa.RecordBatch.from_pydict({n: [] for n in cls.OUTPUT_SCHEMA.names}, schema=params.output_schema))
            return

        yt = np.rint(np.asarray(table.column(a.y_true).to_numpy(zero_copy_only=False), dtype=float)).astype(int)
        ys = np.asarray(table.column(a.y_score).to_numpy(zero_copy_only=False), dtype=float)
        out.emit(pa.RecordBatch.from_pydict(cls.curve(yt, ys), schema=params.output_schema))


_ROC_SCHEMA = pa.schema(
    [
        sfield("threshold", pa.float64(), "Decision threshold for this point."),
        sfield("fpr", pa.float64(), "False positive rate.", nullable=False),
        sfield("tpr", pa.float64(), "True positive rate.", nullable=False),
    ]
)


class RocCurve(_CurveFunction):
    """Receiver operating characteristic curve points for a binary classifier."""

    OUTPUT_SCHEMA: ClassVar[pa.Schema] = _ROC_SCHEMA

    class Meta:
        """VGI metadata for the roc_curve function."""

        name = "roc_curve"
        description = "ROC curve points (threshold, fpr, tpr) for a binary classifier"
        categories = ["metrics", "classification", "ranking"]
        tags = {"vgi.columns_md": columns_md(_ROC_SCHEMA)}
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM sklearn.roc_curve((SELECT y, p FROM preds), "
                    "y_true := 'y', y_score := 'p') ORDER BY fpr"
                ),
                description="ROC curve points",
            )
        ]

    @classmethod
    def curve(cls, y_true: np.ndarray, y_score: np.ndarray) -> dict[str, list[float | None]]:
        """Return ROC (threshold, fpr, tpr) points from sklearn.roc_curve."""
        fpr, tpr, thresholds = skm.roc_curve(y_true, y_score)
        # sklearn prepends an `inf` threshold (the all-negative point); NULL it.
        thr = [None if not np.isfinite(t) else float(t) for t in thresholds]
        return {"threshold": thr, "fpr": [float(v) for v in fpr], "tpr": [float(v) for v in tpr]}


_PR_SCHEMA = pa.schema(
    [
        sfield("threshold", pa.float64(), "Decision threshold (NULL for the final point)."),
        sfield("precision", pa.float64(), "Precision at this threshold.", nullable=False),
        sfield("recall", pa.float64(), "Recall at this threshold.", nullable=False),
    ]
)


class PrecisionRecallCurve(_CurveFunction):
    """Precision-recall curve points for a binary classifier."""

    OUTPUT_SCHEMA: ClassVar[pa.Schema] = _PR_SCHEMA

    class Meta:
        """VGI metadata for the precision_recall_curve function."""

        name = "precision_recall_curve"
        description = "Precision-recall curve points (threshold, precision, recall) for a binary classifier"
        categories = ["metrics", "classification", "ranking"]
        tags = {"vgi.columns_md": columns_md(_PR_SCHEMA)}
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM sklearn.precision_recall_curve((SELECT y, p FROM preds), "
                    "y_true := 'y', y_score := 'p') ORDER BY recall"
                ),
                description="Precision-recall curve points",
            )
        ]

    @classmethod
    def curve(cls, y_true: np.ndarray, y_score: np.ndarray) -> dict[str, list[float | None]]:
        """Return precision-recall (threshold, precision, recall) points."""
        precision, recall, thresholds = skm.precision_recall_curve(y_true, y_score)
        # precision/recall have one more entry than thresholds (the (1, 0) endpoint).
        thr = [float(t) for t in thresholds] + [None]
        return {
            "threshold": thr,
            "precision": [float(v) for v in precision],
            "recall": [float(v) for v in recall],
        }


@dataclass(slots=True, frozen=True)
class _CalibrationArgs:
    data: Annotated[TableInput, Arg(0, doc="Table with the true-label and probability columns.")]
    y_true: Annotated[str, Arg("y_true", default="y_true", doc="Name of the true binary-label column (0/1).")]
    y_score: Annotated[str, Arg("y_score", default="y_score", doc="Name of the predicted-probability column.")]
    n_bins: Annotated[int, Arg("n_bins", default=10, doc="Number of bins to group predicted probabilities into.")]


_CALIBRATION_SCHEMA = pa.schema(
    [
        sfield("prob_pred", pa.float64(), "Mean predicted probability in the bin.", nullable=False),
        sfield("prob_true", pa.float64(), "Observed fraction of positives in the bin.", nullable=False),
    ]
)


class CalibrationCurve(SinkBuffer[_CalibrationArgs, DrainState]):
    """Reliability (calibration) curve: predicted vs. observed probability per bin."""

    FunctionArguments: ClassVar[type] = _CalibrationArgs

    OUTPUT_SCHEMA: ClassVar[pa.Schema] = _CALIBRATION_SCHEMA

    class Meta:
        """VGI metadata for the calibration_curve function."""

        name = "calibration_curve"
        description = "Reliability (calibration) curve: predicted vs. observed probability per bin"
        categories = ["metrics", "classification"]
        tags = {"vgi.columns_md": columns_md(_CALIBRATION_SCHEMA)}
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM sklearn.calibration_curve((SELECT y, p FROM preds), "
                    "y_true := 'y', y_score := 'p', n_bins := 10) ORDER BY prob_pred"
                ),
                description="Calibration curve points",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[_CalibrationArgs]) -> BindResponse:
        """Declare the (prob_pred, prob_true) output schema."""
        return BindResponse(output_schema=cls.OUTPUT_SCHEMA)

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[_CalibrationArgs]
    ) -> DrainState:
        """Start with an unfinished drain state."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[_CalibrationArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Bin the buffered probabilities and emit the calibration curve."""
        if state.done:
            out.finish()
            return
        state.done = True

        from sklearn.calibration import calibration_curve

        a = params.args
        table = cls.buffered_table(params, input_schema_of(params))
        if table is None or table.num_rows == 0:
            out.emit(pa.RecordBatch.from_pydict({"prob_pred": [], "prob_true": []}, schema=params.output_schema))
            return

        yt = np.rint(np.asarray(table.column(a.y_true).to_numpy(zero_copy_only=False), dtype=float)).astype(int)
        ys = np.asarray(table.column(a.y_score).to_numpy(zero_copy_only=False), dtype=float)
        prob_true, prob_pred = calibration_curve(yt, ys, n_bins=a.n_bins)
        out.emit(
            pa.RecordBatch.from_pydict(
                {"prob_pred": [float(v) for v in prob_pred], "prob_true": [float(v) for v in prob_true]},
                schema=params.output_schema,
            )
        )


TABLE_METRIC_FUNCTIONS: list[type] = [
    ConfusionMatrix,
    SilhouetteScore,
    RocCurve,
    PrecisionRecallCurve,
    CalibrationCurve,
]
