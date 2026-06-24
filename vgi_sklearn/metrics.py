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
import numpy.typing as npt
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


def _as_int(a: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
    """Round to the nearest integer (label metrics take integer-coded labels)."""
    return np.rint(a).astype(np.int64)


class _BufferedMetric(AggregateFunction[PairState]):
    """Buffer all ``(y_true, y_pred)`` pairs per group, then score once in finalize.

    Subclasses implement ``compute_metric`` and declare a ``Meta``.
    """

    @classmethod
    def initial_state(cls, params: ProcessParams[None]) -> PairState:
        """Return the empty per-group pair buffer."""
        return PairState()

    @classmethod
    def update(
        cls,
        states: dict[int, PairState],
        group_ids: pa.Int64Array,
        y_true: Annotated[pa.DoubleArray, Param(doc="True (or label) values")],
        y_pred: Annotated[pa.DoubleArray, Param(doc="Predicted values, scores, or probabilities")],
    ) -> None:
        """Accumulate this batch's non-NULL pairs into each group's state."""
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
        """Merge two partial buffers for the same group."""
        return PairState(y_true=source.y_true + target.y_true, y_pred=source.y_pred + target.y_pred)

    @classmethod
    def finalize(
        cls,
        group_ids: pa.Int64Array,
        states: dict[int, PairState],
        params: ProcessParams[None],
    ) -> Annotated[pa.RecordBatch, Returns(pa.float64())]:
        """Score each group's buffered pairs, emitting NULL for empty or failing groups."""
        results: list[float | None] = []
        for gid in group_ids:
            s = states.get(gid.as_py())
            if s is None or not s.y_true:
                results.append(None)
                continue
            yt = np.asarray(s.y_true, dtype=np.float64)
            yp = np.asarray(s.y_pred, dtype=np.float64)
            try:
                results.append(cls.compute_metric(yt, yp))
            except Exception:
                results.append(None)
        return pa.record_batch({"result": pa.array(results, type=pa.float64())})

    @classmethod
    def compute_metric(
        cls, y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]
    ) -> float:  # pragma: no cover
        """Compute the metric over one group's full arrays (overridden per subclass)."""
        raise NotImplementedError


def _example(name: str) -> list[FunctionExample]:
    """Build a one-entry example list for a metric named ``name``."""
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
    """Mean squared error (regression)."""

    class Meta:
        """VGI metadata for the ``mean_squared_error`` aggregate."""

        name = "mean_squared_error"
        description = "Mean squared error (regression)"
        categories = ["metrics", "regression"]
        examples = _example("mean_squared_error")

    @classmethod
    def compute_metric(cls, y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
        """Compute the scikit-learn metric over the buffered arrays."""
        return float(skm.mean_squared_error(y_true, y_pred))


class RootMeanSquaredError(_BufferedMetric):
    """Root mean squared error (regression)."""

    class Meta:
        """VGI metadata for the ``root_mean_squared_error`` aggregate."""

        name = "root_mean_squared_error"
        description = "Root mean squared error (regression)"
        categories = ["metrics", "regression"]
        examples = _example("root_mean_squared_error")

    @classmethod
    def compute_metric(cls, y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
        """Compute the scikit-learn metric over the buffered arrays."""
        return float(np.sqrt(skm.mean_squared_error(y_true, y_pred)))


class MeanAbsoluteError(_BufferedMetric):
    """Mean absolute error (regression)."""

    class Meta:
        """VGI metadata for the ``mean_absolute_error`` aggregate."""

        name = "mean_absolute_error"
        description = "Mean absolute error (regression)"
        categories = ["metrics", "regression"]
        examples = _example("mean_absolute_error")

    @classmethod
    def compute_metric(cls, y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
        """Compute the scikit-learn metric over the buffered arrays."""
        return float(skm.mean_absolute_error(y_true, y_pred))


class R2Score(_BufferedMetric):
    """Coefficient of determination R^2 (regression)."""

    class Meta:
        """VGI metadata for the ``r2_score`` aggregate."""

        name = "r2_score"
        description = "Coefficient of determination R^2 (regression)"
        categories = ["metrics", "regression"]
        examples = _example("r2_score")

    @classmethod
    def compute_metric(cls, y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
        """Compute the scikit-learn metric over the buffered arrays."""
        return float(skm.r2_score(y_true, y_pred))


class ExplainedVarianceScore(_BufferedMetric):
    """Explained variance regression score."""

    class Meta:
        """VGI metadata for the ``explained_variance_score`` aggregate."""

        name = "explained_variance_score"
        description = "Explained variance regression score"
        categories = ["metrics", "regression"]
        examples = _example("explained_variance_score")

    @classmethod
    def compute_metric(cls, y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
        """Compute the scikit-learn metric over the buffered arrays."""
        return float(skm.explained_variance_score(y_true, y_pred))


class MeanAbsolutePercentageError(_BufferedMetric):
    """Mean absolute percentage error (regression)."""

    class Meta:
        """VGI metadata for the ``mean_absolute_percentage_error`` aggregate."""

        name = "mean_absolute_percentage_error"
        description = "Mean absolute percentage error (regression)"
        categories = ["metrics", "regression"]
        examples = _example("mean_absolute_percentage_error")

    @classmethod
    def compute_metric(cls, y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
        """Compute the scikit-learn metric over the buffered arrays."""
        return float(skm.mean_absolute_percentage_error(y_true, y_pred))


class MaxError(_BufferedMetric):
    """Maximum residual error (regression)."""

    class Meta:
        """VGI metadata for the ``max_error`` aggregate."""

        name = "max_error"
        description = "Maximum residual error (regression)"
        categories = ["metrics", "regression"]
        examples = _example("max_error")

    @classmethod
    def compute_metric(cls, y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
        """Compute the scikit-learn metric over the buffered arrays."""
        return float(skm.max_error(y_true, y_pred))


class MedianAbsoluteError(_BufferedMetric):
    """Median absolute error (regression)."""

    class Meta:
        """VGI metadata for the ``median_absolute_error`` aggregate."""

        name = "median_absolute_error"
        description = "Median absolute error (regression)"
        categories = ["metrics", "regression"]
        examples = _example("median_absolute_error")

    @classmethod
    def compute_metric(cls, y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
        """Compute the scikit-learn metric over the buffered arrays."""
        return float(skm.median_absolute_error(y_true, y_pred))


class MeanSquaredLogError(_BufferedMetric):
    """Mean squared logarithmic error (regression; non-negative values)."""

    class Meta:
        """VGI metadata for the ``mean_squared_log_error`` aggregate."""

        name = "mean_squared_log_error"
        description = "Mean squared logarithmic error (regression; non-negative values)"
        categories = ["metrics", "regression"]
        examples = _example("mean_squared_log_error")

    @classmethod
    def compute_metric(cls, y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
        """Compute the scikit-learn metric over the buffered arrays."""
        return float(skm.mean_squared_log_error(y_true, y_pred))


class MeanPinballLoss(_BufferedMetric):
    """Mean pinball loss for the median quantile (regression)."""

    class Meta:
        """VGI metadata for the ``mean_pinball_loss`` aggregate."""

        name = "mean_pinball_loss"
        description = "Mean pinball loss for the median quantile (regression)"
        categories = ["metrics", "regression"]
        examples = _example("mean_pinball_loss")

    @classmethod
    def compute_metric(cls, y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
        """Compute the scikit-learn metric over the buffered arrays."""
        return float(skm.mean_pinball_loss(y_true, y_pred))


# ===========================================================================
# Classification metrics (numeric labels; macro-averaged where applicable)
# ===========================================================================


class AccuracyScore(_BufferedMetric):
    """Classification accuracy."""

    class Meta:
        """VGI metadata for the ``accuracy_score`` aggregate."""

        name = "accuracy_score"
        description = "Classification accuracy"
        categories = ["metrics", "classification"]
        examples = _example("accuracy_score")

    @classmethod
    def compute_metric(cls, y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
        """Compute the scikit-learn metric over the buffered arrays."""
        return float(skm.accuracy_score(_as_int(y_true), _as_int(y_pred)))


class PrecisionScore(_BufferedMetric):
    """Precision (macro-averaged for multiclass)."""

    class Meta:
        """VGI metadata for the ``precision_score`` aggregate."""

        name = "precision_score"
        description = "Precision (macro-averaged for multiclass)"
        categories = ["metrics", "classification"]
        examples = _example("precision_score")

    @classmethod
    def compute_metric(cls, y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
        """Compute the scikit-learn metric over the buffered arrays."""
        return float(skm.precision_score(_as_int(y_true), _as_int(y_pred), average="macro", zero_division=0))


class RecallScore(_BufferedMetric):
    """Recall (macro-averaged for multiclass)."""

    class Meta:
        """VGI metadata for the ``recall_score`` aggregate."""

        name = "recall_score"
        description = "Recall (macro-averaged for multiclass)"
        categories = ["metrics", "classification"]
        examples = _example("recall_score")

    @classmethod
    def compute_metric(cls, y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
        """Compute the scikit-learn metric over the buffered arrays."""
        return float(skm.recall_score(_as_int(y_true), _as_int(y_pred), average="macro", zero_division=0))


class F1Score(_BufferedMetric):
    """F1 score (macro-averaged for multiclass)."""

    class Meta:
        """VGI metadata for the ``f1_score`` aggregate."""

        name = "f1_score"
        description = "F1 score (macro-averaged for multiclass)"
        categories = ["metrics", "classification"]
        examples = _example("f1_score")

    @classmethod
    def compute_metric(cls, y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
        """Compute the scikit-learn metric over the buffered arrays."""
        return float(skm.f1_score(_as_int(y_true), _as_int(y_pred), average="macro", zero_division=0))


class BalancedAccuracyScore(_BufferedMetric):
    """Balanced accuracy."""

    class Meta:
        """VGI metadata for the ``balanced_accuracy_score`` aggregate."""

        name = "balanced_accuracy_score"
        description = "Balanced accuracy"
        categories = ["metrics", "classification"]
        examples = _example("balanced_accuracy_score")

    @classmethod
    def compute_metric(cls, y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
        """Compute the scikit-learn metric over the buffered arrays."""
        return float(skm.balanced_accuracy_score(_as_int(y_true), _as_int(y_pred)))


class MatthewsCorrCoef(_BufferedMetric):
    """Matthews correlation coefficient."""

    class Meta:
        """VGI metadata for the ``matthews_corrcoef`` aggregate."""

        name = "matthews_corrcoef"
        description = "Matthews correlation coefficient"
        categories = ["metrics", "classification"]
        examples = _example("matthews_corrcoef")

    @classmethod
    def compute_metric(cls, y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
        """Compute the scikit-learn metric over the buffered arrays."""
        return float(skm.matthews_corrcoef(_as_int(y_true), _as_int(y_pred)))


class CohenKappaScore(_BufferedMetric):
    """Cohen's kappa (inter-rater agreement)."""

    class Meta:
        """VGI metadata for the ``cohen_kappa_score`` aggregate."""

        name = "cohen_kappa_score"
        description = "Cohen's kappa (inter-rater agreement)"
        categories = ["metrics", "classification"]
        examples = _example("cohen_kappa_score")

    @classmethod
    def compute_metric(cls, y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
        """Compute the scikit-learn metric over the buffered arrays."""
        return float(skm.cohen_kappa_score(_as_int(y_true), _as_int(y_pred)))


class JaccardScore(_BufferedMetric):
    """Jaccard similarity coefficient (macro-averaged for multiclass)."""

    class Meta:
        """VGI metadata for the ``jaccard_score`` aggregate."""

        name = "jaccard_score"
        description = "Jaccard similarity coefficient (macro-averaged for multiclass)"
        categories = ["metrics", "classification"]
        examples = _example("jaccard_score")

    @classmethod
    def compute_metric(cls, y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
        """Compute the scikit-learn metric over the buffered arrays."""
        return float(skm.jaccard_score(_as_int(y_true), _as_int(y_pred), average="macro", zero_division=0))


class HammingLoss(_BufferedMetric):
    """Hamming loss (fraction of labels predicted incorrectly)."""

    class Meta:
        """VGI metadata for the ``hamming_loss`` aggregate."""

        name = "hamming_loss"
        description = "Hamming loss (fraction of labels predicted incorrectly)"
        categories = ["metrics", "classification"]
        examples = _example("hamming_loss")

    @classmethod
    def compute_metric(cls, y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
        """Compute the scikit-learn metric over the buffered arrays."""
        return float(skm.hamming_loss(_as_int(y_true), _as_int(y_pred)))


class ZeroOneLoss(_BufferedMetric):
    """Zero-one classification loss (fraction of misclassifications)."""

    class Meta:
        """VGI metadata for the ``zero_one_loss`` aggregate."""

        name = "zero_one_loss"
        description = "Zero-one classification loss (fraction of misclassifications)"
        categories = ["metrics", "classification"]
        examples = _example("zero_one_loss")

    @classmethod
    def compute_metric(cls, y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
        """Compute the scikit-learn metric over the buffered arrays."""
        return float(skm.zero_one_loss(_as_int(y_true), _as_int(y_pred)))


# ===========================================================================
# Probability / score-based metrics (y_pred is a score or probability)
# ===========================================================================


class RocAucScore(_BufferedMetric):
    """Area under the ROC curve (y_pred = score/probability)."""

    class Meta:
        """VGI metadata for the ``roc_auc_score`` aggregate."""

        name = "roc_auc_score"
        description = "Area under the ROC curve (y_pred = score/probability)"
        categories = ["metrics", "classification", "ranking"]
        examples = _example("roc_auc_score")

    @classmethod
    def compute_metric(cls, y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
        """Compute the scikit-learn metric over the buffered arrays."""
        return float(skm.roc_auc_score(_as_int(y_true), y_pred))


class AveragePrecisionScore(_BufferedMetric):
    """Average precision (y_pred = score/probability)."""

    class Meta:
        """VGI metadata for the ``average_precision_score`` aggregate."""

        name = "average_precision_score"
        description = "Average precision (y_pred = score/probability)"
        categories = ["metrics", "classification", "ranking"]
        examples = _example("average_precision_score")

    @classmethod
    def compute_metric(cls, y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
        """Compute the scikit-learn metric over the buffered arrays."""
        return float(skm.average_precision_score(_as_int(y_true), y_pred))


class LogLoss(_BufferedMetric):
    """Logistic / cross-entropy loss (y_pred = P(class 1))."""

    class Meta:
        """VGI metadata for the ``log_loss`` aggregate."""

        name = "log_loss"
        description = "Logistic / cross-entropy loss (y_pred = P(class 1))"
        categories = ["metrics", "classification"]
        examples = _example("log_loss")

    @classmethod
    def compute_metric(cls, y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
        """Compute the scikit-learn metric over the buffered arrays."""
        return float(skm.log_loss(_as_int(y_true), np.clip(y_pred, 1e-15, 1 - 1e-15), labels=[0, 1]))


class BrierScoreLoss(_BufferedMetric):
    """Brier score loss (calibration of binary probabilities; y_pred = P(class 1))."""

    class Meta:
        """VGI metadata for the ``brier_score_loss`` aggregate."""

        name = "brier_score_loss"
        description = "Brier score loss (calibration of binary probabilities; y_pred = P(class 1))"
        categories = ["metrics", "classification"]
        examples = _example("brier_score_loss")

    @classmethod
    def compute_metric(cls, y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
        """Compute the scikit-learn metric over the buffered arrays."""
        return float(skm.brier_score_loss(_as_int(y_true), y_pred))


# ===========================================================================
# Clustering comparison metrics (two integer label columns)
# ===========================================================================


class AdjustedRandScore(_BufferedMetric):
    """Adjusted Rand index between two clusterings."""

    class Meta:
        """VGI metadata for the ``adjusted_rand_score`` aggregate."""

        name = "adjusted_rand_score"
        description = "Adjusted Rand index between two clusterings"
        categories = ["metrics", "clustering"]
        examples = _example("adjusted_rand_score")

    @classmethod
    def compute_metric(cls, y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
        """Compute the scikit-learn metric over the buffered arrays."""
        return float(skm.adjusted_rand_score(_as_int(y_true), _as_int(y_pred)))


class NormalizedMutualInfoScore(_BufferedMetric):
    """Normalized mutual information between two clusterings."""

    class Meta:
        """VGI metadata for the ``normalized_mutual_info_score`` aggregate."""

        name = "normalized_mutual_info_score"
        description = "Normalized mutual information between two clusterings"
        categories = ["metrics", "clustering"]
        examples = _example("normalized_mutual_info_score")

    @classmethod
    def compute_metric(cls, y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
        """Compute the scikit-learn metric over the buffered arrays."""
        return float(skm.normalized_mutual_info_score(_as_int(y_true), _as_int(y_pred)))


class AdjustedMutualInfoScore(_BufferedMetric):
    """Adjusted mutual information between two clusterings."""

    class Meta:
        """VGI metadata for the ``adjusted_mutual_info_score`` aggregate."""

        name = "adjusted_mutual_info_score"
        description = "Adjusted mutual information between two clusterings"
        categories = ["metrics", "clustering"]
        examples = _example("adjusted_mutual_info_score")

    @classmethod
    def compute_metric(cls, y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
        """Compute the scikit-learn metric over the buffered arrays."""
        return float(skm.adjusted_mutual_info_score(_as_int(y_true), _as_int(y_pred)))


class HomogeneityScore(_BufferedMetric):
    """Homogeneity of a clustering vs. ground truth."""

    class Meta:
        """VGI metadata for the ``homogeneity_score`` aggregate."""

        name = "homogeneity_score"
        description = "Homogeneity of a clustering vs. ground truth"
        categories = ["metrics", "clustering"]
        examples = _example("homogeneity_score")

    @classmethod
    def compute_metric(cls, y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
        """Compute the scikit-learn metric over the buffered arrays."""
        return float(skm.homogeneity_score(_as_int(y_true), _as_int(y_pred)))


class CompletenessScore(_BufferedMetric):
    """Completeness of a clustering vs. ground truth."""

    class Meta:
        """VGI metadata for the ``completeness_score`` aggregate."""

        name = "completeness_score"
        description = "Completeness of a clustering vs. ground truth"
        categories = ["metrics", "clustering"]
        examples = _example("completeness_score")

    @classmethod
    def compute_metric(cls, y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
        """Compute the scikit-learn metric over the buffered arrays."""
        return float(skm.completeness_score(_as_int(y_true), _as_int(y_pred)))


class VMeasureScore(_BufferedMetric):
    """V-measure (harmonic mean of homogeneity and completeness)."""

    class Meta:
        """VGI metadata for the ``v_measure_score`` aggregate."""

        name = "v_measure_score"
        description = "V-measure (harmonic mean of homogeneity and completeness)"
        categories = ["metrics", "clustering"]
        examples = _example("v_measure_score")

    @classmethod
    def compute_metric(cls, y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
        """Compute the scikit-learn metric over the buffered arrays."""
        return float(skm.v_measure_score(_as_int(y_true), _as_int(y_pred)))


class FowlkesMallowsScore(_BufferedMetric):
    """Fowlkes-Mallows index between two clusterings."""

    class Meta:
        """VGI metadata for the ``fowlkes_mallows_score`` aggregate."""

        name = "fowlkes_mallows_score"
        description = "Fowlkes-Mallows index between two clusterings"
        categories = ["metrics", "clustering"]
        examples = _example("fowlkes_mallows_score")

    @classmethod
    def compute_metric(cls, y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
        """Compute the scikit-learn metric over the buffered arrays."""
        return float(skm.fowlkes_mallows_score(_as_int(y_true), _as_int(y_pred)))


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
    MeanSquaredLogError,
    MeanPinballLoss,
    # classification
    AccuracyScore,
    PrecisionScore,
    RecallScore,
    F1Score,
    BalancedAccuracyScore,
    MatthewsCorrCoef,
    CohenKappaScore,
    JaccardScore,
    HammingLoss,
    ZeroOneLoss,
    # probability / ranking
    RocAucScore,
    AveragePrecisionScore,
    LogLoss,
    BrierScoreLoss,
    # clustering comparison
    AdjustedRandScore,
    NormalizedMutualInfoScore,
    AdjustedMutualInfoScore,
    HomogeneityScore,
    CompletenessScore,
    VMeasureScore,
    FowlkesMallowsScore,
]
