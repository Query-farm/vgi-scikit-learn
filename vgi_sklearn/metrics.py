"""scikit-learn metrics exposed as DuckDB aggregate functions.

Each metric is an aggregate over two columns -- ``(y_true, y_pred)`` -- so it
composes with ``GROUP BY``:

    SELECT model, sklearn.r2_score(actual, predicted) AS r2
    FROM predictions GROUP BY model;

Implementation: most scikit-learn metrics (f1, roc_auc, ...) need the full
arrays, not streaming sufficient statistics, so each group buffers its
``(y_true, y_pred)`` pairs and the scikit-learn function is called once in
``finalize``. Inputs are taken as float64 (DuckDB casts integers up); label
metrics round back to int. Multiclass precision/recall/f1 use macro averaging.
Rows where either value is NULL are skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

import numpy as np
import pyarrow as pa
from sklearn import metrics as skm
from vgi.aggregate_function import AggregateFunction
from vgi.arguments import Param, Returns
from vgi.metadata import FunctionExample
from vgi.table_function import ProcessParams
from vgi_rpc import ArrowSerializableDataclass


@dataclass(kw_only=True)
class PairState(ArrowSerializableDataclass):
    """Buffered ``(y_true, y_pred)`` pairs for one group."""

    y_true: list[float] = field(default_factory=list)
    y_pred: list[float] = field(default_factory=list)


def _as_int(a: np.ndarray) -> np.ndarray:
    return np.rint(a).astype(int)


class _BufferedMetric(AggregateFunction[PairState]):
    """Base: buffer all (y_true, y_pred) pairs per group, score once in finalize.

    Subclasses implement ``compute_metric`` and declare a ``Meta``.
    """

    @classmethod
    def initial_state(cls, params: ProcessParams[None]) -> PairState:
        return PairState()

    @classmethod
    def update(
        cls,
        states: dict[int, PairState],
        group_ids: pa.Int64Array,
        y_true: Annotated[pa.DoubleArray, Param(doc="True (or label) values")],
        y_pred: Annotated[pa.DoubleArray, Param(doc="Predicted values, scores, or probabilities")],
    ) -> None:
        # Accumulate this batch per group, then *reassign* states[g] (the
        # framework persists state by re-reading the dict entry, matching the
        # reassignment idiom of the shipped aggregate examples).
        batch: dict[int, tuple[list[float], list[float]]] = {}
        for g, t, p in zip(group_ids.to_pylist(), y_true.to_pylist(), y_pred.to_pylist(), strict=False):
            if t is None or p is None:
                continue
            bt, bp = batch.setdefault(g, ([], []))
            bt.append(t)
            bp.append(p)
        for g, (bt, bp) in batch.items():
            s = states[g]
            states[g] = PairState(y_true=s.y_true + bt, y_pred=s.y_pred + bp)

    @classmethod
    def combine(cls, source: PairState, target: PairState, params: ProcessParams[None]) -> PairState:
        return PairState(y_true=source.y_true + target.y_true, y_pred=source.y_pred + target.y_pred)

    @classmethod
    def finalize(
        cls,
        group_ids: pa.Int64Array,
        states: dict[int, PairState],
        params: ProcessParams[None],
    ) -> Annotated[pa.RecordBatch, Returns(pa.float64())]:
        results: list[float | None] = []
        for gid in group_ids:
            s = states.get(gid.as_py())
            if s is None or not s.y_true:
                results.append(None)
                continue
            yt = np.asarray(s.y_true, dtype=float)
            yp = np.asarray(s.y_pred, dtype=float)
            try:
                results.append(float(cls.compute_metric(yt, yp)))
            except Exception:
                results.append(None)
        return pa.record_batch({"result": pa.array(results, type=pa.float64())})

    @classmethod
    def compute_metric(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:  # pragma: no cover
        raise NotImplementedError


def _example(name: str) -> list[FunctionExample]:
    return [
        FunctionExample(
            sql=f"SELECT sklearn.{name}(actual, predicted) FROM predictions",
            description=f"{name} over a predictions table",
        )
    ]


# ===========================================================================
# Regression metrics
# ===========================================================================


class MeanSquaredError(_BufferedMetric):
    class Meta:
        name = "mean_squared_error"
        description = "Mean squared error (regression)"
        categories = ["metrics", "regression"]
        examples = _example("mean_squared_error")

    @classmethod
    def compute_metric(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return skm.mean_squared_error(y_true, y_pred)


class RootMeanSquaredError(_BufferedMetric):
    class Meta:
        name = "root_mean_squared_error"
        description = "Root mean squared error (regression)"
        categories = ["metrics", "regression"]
        examples = _example("root_mean_squared_error")

    @classmethod
    def compute_metric(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(skm.mean_squared_error(y_true, y_pred)))


class MeanAbsoluteError(_BufferedMetric):
    class Meta:
        name = "mean_absolute_error"
        description = "Mean absolute error (regression)"
        categories = ["metrics", "regression"]
        examples = _example("mean_absolute_error")

    @classmethod
    def compute_metric(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return skm.mean_absolute_error(y_true, y_pred)


class R2Score(_BufferedMetric):
    class Meta:
        name = "r2_score"
        description = "Coefficient of determination R^2 (regression)"
        categories = ["metrics", "regression"]
        examples = _example("r2_score")

    @classmethod
    def compute_metric(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return skm.r2_score(y_true, y_pred)


class ExplainedVarianceScore(_BufferedMetric):
    class Meta:
        name = "explained_variance_score"
        description = "Explained variance regression score"
        categories = ["metrics", "regression"]
        examples = _example("explained_variance_score")

    @classmethod
    def compute_metric(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return skm.explained_variance_score(y_true, y_pred)


class MeanAbsolutePercentageError(_BufferedMetric):
    class Meta:
        name = "mean_absolute_percentage_error"
        description = "Mean absolute percentage error (regression)"
        categories = ["metrics", "regression"]
        examples = _example("mean_absolute_percentage_error")

    @classmethod
    def compute_metric(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return skm.mean_absolute_percentage_error(y_true, y_pred)


class MaxError(_BufferedMetric):
    class Meta:
        name = "max_error"
        description = "Maximum residual error (regression)"
        categories = ["metrics", "regression"]
        examples = _example("max_error")

    @classmethod
    def compute_metric(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return skm.max_error(y_true, y_pred)


class MedianAbsoluteError(_BufferedMetric):
    class Meta:
        name = "median_absolute_error"
        description = "Median absolute error (regression)"
        categories = ["metrics", "regression"]
        examples = _example("median_absolute_error")

    @classmethod
    def compute_metric(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return skm.median_absolute_error(y_true, y_pred)


# ===========================================================================
# Classification metrics (numeric labels; macro-averaged where applicable)
# ===========================================================================


class AccuracyScore(_BufferedMetric):
    class Meta:
        name = "accuracy_score"
        description = "Classification accuracy"
        categories = ["metrics", "classification"]
        examples = _example("accuracy_score")

    @classmethod
    def compute_metric(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return skm.accuracy_score(_as_int(y_true), _as_int(y_pred))


class PrecisionScore(_BufferedMetric):
    class Meta:
        name = "precision_score"
        description = "Precision (macro-averaged for multiclass)"
        categories = ["metrics", "classification"]
        examples = _example("precision_score")

    @classmethod
    def compute_metric(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return skm.precision_score(_as_int(y_true), _as_int(y_pred), average="macro", zero_division=0)


class RecallScore(_BufferedMetric):
    class Meta:
        name = "recall_score"
        description = "Recall (macro-averaged for multiclass)"
        categories = ["metrics", "classification"]
        examples = _example("recall_score")

    @classmethod
    def compute_metric(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return skm.recall_score(_as_int(y_true), _as_int(y_pred), average="macro", zero_division=0)


class F1Score(_BufferedMetric):
    class Meta:
        name = "f1_score"
        description = "F1 score (macro-averaged for multiclass)"
        categories = ["metrics", "classification"]
        examples = _example("f1_score")

    @classmethod
    def compute_metric(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return skm.f1_score(_as_int(y_true), _as_int(y_pred), average="macro", zero_division=0)


class BalancedAccuracyScore(_BufferedMetric):
    class Meta:
        name = "balanced_accuracy_score"
        description = "Balanced accuracy"
        categories = ["metrics", "classification"]
        examples = _example("balanced_accuracy_score")

    @classmethod
    def compute_metric(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return skm.balanced_accuracy_score(_as_int(y_true), _as_int(y_pred))


class MatthewsCorrCoef(_BufferedMetric):
    class Meta:
        name = "matthews_corrcoef"
        description = "Matthews correlation coefficient"
        categories = ["metrics", "classification"]
        examples = _example("matthews_corrcoef")

    @classmethod
    def compute_metric(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return skm.matthews_corrcoef(_as_int(y_true), _as_int(y_pred))


class CohenKappaScore(_BufferedMetric):
    class Meta:
        name = "cohen_kappa_score"
        description = "Cohen's kappa (inter-rater agreement)"
        categories = ["metrics", "classification"]
        examples = _example("cohen_kappa_score")

    @classmethod
    def compute_metric(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return skm.cohen_kappa_score(_as_int(y_true), _as_int(y_pred))


# ===========================================================================
# Probability / score-based metrics (y_pred is a score or probability)
# ===========================================================================


class RocAucScore(_BufferedMetric):
    class Meta:
        name = "roc_auc_score"
        description = "Area under the ROC curve (y_pred = score/probability)"
        categories = ["metrics", "classification", "ranking"]
        examples = _example("roc_auc_score")

    @classmethod
    def compute_metric(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return skm.roc_auc_score(_as_int(y_true), y_pred)


class AveragePrecisionScore(_BufferedMetric):
    class Meta:
        name = "average_precision_score"
        description = "Average precision (y_pred = score/probability)"
        categories = ["metrics", "classification", "ranking"]
        examples = _example("average_precision_score")

    @classmethod
    def compute_metric(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return skm.average_precision_score(_as_int(y_true), y_pred)


class LogLoss(_BufferedMetric):
    class Meta:
        name = "log_loss"
        description = "Logistic / cross-entropy loss (y_pred = P(class 1))"
        categories = ["metrics", "classification"]
        examples = _example("log_loss")

    @classmethod
    def compute_metric(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return skm.log_loss(_as_int(y_true), np.clip(y_pred, 1e-15, 1 - 1e-15), labels=[0, 1])


# ===========================================================================
# Clustering comparison metrics (two integer label columns)
# ===========================================================================


class AdjustedRandScore(_BufferedMetric):
    class Meta:
        name = "adjusted_rand_score"
        description = "Adjusted Rand index between two clusterings"
        categories = ["metrics", "clustering"]
        examples = _example("adjusted_rand_score")

    @classmethod
    def compute_metric(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return skm.adjusted_rand_score(_as_int(y_true), _as_int(y_pred))


class NormalizedMutualInfoScore(_BufferedMetric):
    class Meta:
        name = "normalized_mutual_info_score"
        description = "Normalized mutual information between two clusterings"
        categories = ["metrics", "clustering"]
        examples = _example("normalized_mutual_info_score")

    @classmethod
    def compute_metric(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return skm.normalized_mutual_info_score(_as_int(y_true), _as_int(y_pred))


class AdjustedMutualInfoScore(_BufferedMetric):
    class Meta:
        name = "adjusted_mutual_info_score"
        description = "Adjusted mutual information between two clusterings"
        categories = ["metrics", "clustering"]
        examples = _example("adjusted_mutual_info_score")

    @classmethod
    def compute_metric(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return skm.adjusted_mutual_info_score(_as_int(y_true), _as_int(y_pred))


class HomogeneityScore(_BufferedMetric):
    class Meta:
        name = "homogeneity_score"
        description = "Homogeneity of a clustering vs. ground truth"
        categories = ["metrics", "clustering"]
        examples = _example("homogeneity_score")

    @classmethod
    def compute_metric(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return skm.homogeneity_score(_as_int(y_true), _as_int(y_pred))


class CompletenessScore(_BufferedMetric):
    class Meta:
        name = "completeness_score"
        description = "Completeness of a clustering vs. ground truth"
        categories = ["metrics", "clustering"]
        examples = _example("completeness_score")

    @classmethod
    def compute_metric(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return skm.completeness_score(_as_int(y_true), _as_int(y_pred))


class VMeasureScore(_BufferedMetric):
    class Meta:
        name = "v_measure_score"
        description = "V-measure (harmonic mean of homogeneity and completeness)"
        categories = ["metrics", "clustering"]
        examples = _example("v_measure_score")

    @classmethod
    def compute_metric(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return skm.v_measure_score(_as_int(y_true), _as_int(y_pred))


class FowlkesMallowsScore(_BufferedMetric):
    class Meta:
        name = "fowlkes_mallows_score"
        description = "Fowlkes-Mallows index between two clusterings"
        categories = ["metrics", "clustering"]
        examples = _example("fowlkes_mallows_score")

    @classmethod
    def compute_metric(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return skm.fowlkes_mallows_score(_as_int(y_true), _as_int(y_pred))


METRIC_FUNCTIONS: list[type] = [
    # regression
    MeanSquaredError,
    RootMeanSquaredError,
    MeanAbsoluteError,
    R2Score,
    ExplainedVarianceScore,
    MeanAbsolutePercentageError,
    MaxError,
    MedianAbsoluteError,
    # classification
    AccuracyScore,
    PrecisionScore,
    RecallScore,
    F1Score,
    BalancedAccuracyScore,
    MatthewsCorrCoef,
    CohenKappaScore,
    # probability / ranking
    RocAucScore,
    AveragePrecisionScore,
    LogLoss,
    # clustering comparison
    AdjustedRandScore,
    NormalizedMutualInfoScore,
    AdjustedMutualInfoScore,
    HomogeneityScore,
    CompletenessScore,
    VMeasureScore,
    FowlkesMallowsScore,
]
