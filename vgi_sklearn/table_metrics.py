"""Metrics that need a whole table at once (buffering functions).

Unlike the column aggregates in ``metrics.py``, these consume a table:

* ``confusion_matrix`` -- long-format counts of (actual, predicted) label pairs.
* ``silhouette_score`` -- one score from a feature matrix + cluster labels.

    SELECT * FROM sklearn.metrics.confusion_matrix((SELECT y, yhat FROM preds), actual => 'y', predicted => 'yhat');
    SELECT * FROM sklearn.metrics.silhouette_score((SELECT * FROM clustered), label => 'cluster', id => 'id');
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

    data: Annotated[TableInput, Arg(0, doc="The actual and predicted label columns.")]
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
        tags = {
            "vgi.result_columns_md": columns_md(_CONFUSION_SCHEMA),
            "vgi.doc_llm": (
                "Table function that tabulates a confusion matrix from a buffered table of label pairs, "
                "emitting it in long format. The table arg is `(SELECT actual_col, predicted_col FROM ...)`; "
                "name the columns with `actual :=` and `predicted :=` (default `'actual'`/`'predicted'`). "
                "Labels are read as integer codes (rounded). Returns one row per observed "
                "`(actual, predicted)` pair with its `count` — pivot it for a square matrix, or filter "
                "`actual <> predicted` to inspect specific misclassifications. Use it to see exactly which "
                "classes a model confuses, not just an aggregate score."
            ),
            "vgi.doc_md": (
                "**Confusion matrix (long format)** — counts of every true/predicted label combination.\n\n"
                "- Table arg: `(SELECT actual, predicted FROM ...)`; `actual :=` / `predicted :=` name "
                "the columns (integer-coded labels)\n"
                "- Returns one row per observed pair:\n"
                "  - `actual` — true class label\n"
                "  - `predicted` — predicted class label\n"
                "  - `count` — number of rows with that pair\n"
                "- Long shape sidesteps a fixed matrix width; `PIVOT` to a grid or keep "
                "`actual <> predicted` rows to enumerate error types"
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM sklearn.metrics.confusion_matrix("
                    "(SELECT * FROM (VALUES (0, 0), (1, 1), (0, 1), (1, 0)) AS preds(y, yhat)), "
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

    data: Annotated[TableInput, Arg(0, doc="Feature columns plus a cluster-label column.")]
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
        tags = {
            "vgi.result_columns_md": columns_md(_SILHOUETTE_SCHEMA),
            "vgi.doc_llm": (
                "Table function that scores how well-separated a clustering is, returning the mean "
                "silhouette coefficient over all samples in one row. The table arg is "
                "`(SELECT feature_cols..., cluster_label FROM ...)`; `label :=` names the cluster column "
                "(default `'cluster'`) and `id :=` optionally names a column to exclude from the feature "
                "matrix — every other numeric column is a feature. The silhouette ranges -1..+1 (near +1 = "
                "dense, well-separated clusters; near 0 = overlapping; negative = likely misassigned "
                "points), and is an unsupervised quality measure needing no ground truth. NULL is returned "
                "when fewer than 2 or more than n-1 distinct clusters make it undefined."
            ),
            "vgi.doc_md": (
                "**Silhouette score** — unsupervised measure of cluster separation.\n\n"
                "- Table arg: `(SELECT <features...>, <label> FROM ...)`; `label :=` the cluster column, "
                "`id :=` an optional column to drop from the features\n"
                "- Returns a single row, `silhouette_score` (`DOUBLE`, `NULL` if undefined):\n"
                "  - `~+1` tight, well-separated clusters\n"
                "  - `~0` overlapping clusters\n"
                "  - `<0` points likely in the wrong cluster\n"
                "- Needs 2..n-1 distinct labels and uses the remaining numeric columns as the feature "
                "matrix; no ground-truth labels required"
            ),
        }
        examples = [
            FunctionExample(
                sql="SELECT * FROM sklearn.metrics.silhouette_score((SELECT * FROM (VALUES (1, 0.1, 0.2, 0), (2, 0.2, 0.1, 0), (3, 5.0, 5.1, 1), (4, 5.1, 5.0, 1)) AS clustered(id, x1, x2, cluster)), label => 'cluster', id => 'id')",  # noqa: E501
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
    data: Annotated[TableInput, Arg(0, doc="The true-label and score columns.")]
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
        tags = {
            "vgi.result_columns_md": columns_md(_ROC_SCHEMA),
            "vgi.doc_llm": (
                "Table function that traces the receiver operating characteristic (ROC) curve for a binary "
                "classifier. The table arg is `(SELECT y_true_col, y_score_col FROM ...)`; `y_true :=` "
                "names the 0/1 label column and `y_score :=` the continuous score/probability column "
                "(defaults `'y_true'`/`'y_score'`). Returns one row per distinct decision threshold giving "
                "the false-positive rate and true-positive rate at that cutoff, sweeping the full "
                "sensitivity/specificity trade-off. Order by `fpr` and plot tpr-vs-fpr; the area under it "
                "is `roc_auc_score`. Use it to choose an operating threshold rather than just summarize "
                "ranking."
            ),
            "vgi.doc_md": (
                "**ROC curve** — false- vs. true-positive rate across all thresholds.\n\n"
                "- Table arg: `(SELECT y_true, y_score FROM ...)`; `y_true :=` (0/1 labels), "
                "`y_score :=` (score/probability)\n"
                "- Returns one row per threshold:\n"
                "  - `threshold` — the decision cutoff (`NULL` for the all-negative endpoint)\n"
                "  - `fpr` — false positive rate\n"
                "  - `tpr` — true positive rate\n"
                "- Sweep it to pick an operating point; the area under the curve is `roc_auc_score`"
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM sklearn.metrics.roc_curve("
                    "(SELECT * FROM (VALUES (0, 0.1), (1, 0.9), (0, 0.2), (1, 0.8)) AS preds(y, p)), "
                    "y_true := 'y', y_score := 'p') ORDER BY fpr"
                ),
                description="ROC curve points",
            )
        ]

    @classmethod
    def curve(cls, y_true: np.ndarray, y_score: np.ndarray) -> dict[str, list[float | None]]:
        """Return ROC (threshold, fpr, tpr) points from sklearn.metrics.roc_curve."""
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
        tags = {
            "vgi.result_columns_md": columns_md(_PR_SCHEMA),
            "vgi.doc_llm": (
                "Table function that traces the precision-recall curve for a binary classifier. The table "
                "arg is `(SELECT y_true_col, y_score_col FROM ...)`; `y_true :=` names the 0/1 label column "
                "and `y_score :=` the score/probability column (defaults `'y_true'`/`'y_score'`). Returns "
                "one row per threshold with the precision and recall at that cutoff. Preferred over the ROC "
                "curve when positives are rare, since it ignores true negatives; order by `recall` and plot "
                "precision-vs-recall, and `average_precision_score` summarizes the area. Use it to pick a "
                "threshold that balances precision against recall."
            ),
            "vgi.doc_md": (
                "**Precision-recall curve** — precision vs. recall across all thresholds.\n\n"
                "- Table arg: `(SELECT y_true, y_score FROM ...)`; `y_true :=` (0/1 labels), "
                "`y_score :=` (score/probability)\n"
                "- Returns one row per threshold:\n"
                "  - `threshold` — the decision cutoff (`NULL` for the final endpoint)\n"
                "  - `precision` — precision at that cutoff\n"
                "  - `recall` — recall at that cutoff\n"
                "- More informative than ROC under class imbalance; its area is `average_precision_score`"
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM sklearn.metrics.precision_recall_curve("
                    "(SELECT * FROM (VALUES (0, 0.1), (1, 0.9), (0, 0.2), (1, 0.8)) AS preds(y, p)), "
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
    data: Annotated[TableInput, Arg(0, doc="The true-label and probability columns.")]
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
        tags = {
            "vgi.result_columns_md": columns_md(_CALIBRATION_SCHEMA),
            "vgi.doc_llm": (
                "Table function that builds a reliability (calibration) curve to check whether a binary "
                "classifier's predicted probabilities match observed outcome frequencies. The table arg is "
                "`(SELECT y_true_col, y_score_col FROM ...)`; `y_true :=` names the 0/1 label column, "
                "`y_score :=` the predicted-probability column, and `n_bins :=` sets how many probability "
                "bins to group into (default 10). Returns one row per bin with the mean predicted "
                "probability and the actual fraction of positives in that bin. Plot `prob_true` against "
                "`prob_pred`: a perfectly calibrated model lies on the diagonal, points above mean "
                "under-confidence and below mean over-confidence."
            ),
            "vgi.doc_md": (
                "**Calibration (reliability) curve** — predicted vs. observed probability, binned.\n\n"
                "- Table arg: `(SELECT y_true, y_score FROM ...)`; `y_true :=` (0/1 labels), "
                "`y_score :=` (predicted probability), `n_bins :=` (bin count, default 10)\n"
                "- Returns one row per bin:\n"
                "  - `prob_pred` — mean predicted probability in the bin\n"
                "  - `prob_true` — observed fraction of positives in the bin\n"
                "- On the `prob_true` vs `prob_pred` diagonal = well calibrated; above = under-confident, "
                "below = over-confident"
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM sklearn.metrics.calibration_curve("
                    "(SELECT * FROM (VALUES (0, 0.1), (1, 0.9), (0, 0.2), (1, 0.8)) AS preds(y, p)), "
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
