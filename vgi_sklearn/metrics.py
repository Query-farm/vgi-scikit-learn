"""scikit-learn metrics exposed as DuckDB aggregate functions.

Each metric is an aggregate over two columns -- ``(y_true, y_pred)`` -- so it
composes with ``GROUP BY``:

    SELECT model, sklearn.metrics.r2_score(actual, predicted) AS r2
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
        tags = {
            "vgi.doc_llm": (
                "Aggregate over a `(y_true, y_pred)` column pair that returns the mean squared error "
                "(MSE): the average of `(y_true - y_pred)**2`. Pass the true target column first and the "
                "predicted column second; `GROUP BY` to score per model/segment, NULL rows are skipped. "
                "Lower is better (0 is perfect); the squaring penalizes large residuals heavily, and the "
                "units are the square of the target's units. Use it to compare regression fits or as a "
                "training-style loss."
            ),
            "vgi.doc_md": (
                "**Mean squared error (MSE)** — average squared residual for a regression.\n\n"
                "- Inputs: two numeric columns, `y_true` (actuals) then `y_pred` (predictions)\n"
                "- Returns: a single `DOUBLE` per group; lower is better, `0.0` is a perfect fit\n"
                "- Large errors dominate (squared), so it is sensitive to outliers; reported in "
                "squared target units (take `root_mean_squared_error` for the original units)"
            ),
        }

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
        tags = {
            "vgi.doc_llm": (
                "Aggregate over a `(y_true, y_pred)` column pair returning the root mean squared error "
                "(RMSE) — the square root of the mean of `(y_true - y_pred)**2`. Like `mean_squared_error` "
                "but reported back in the target's own units, which makes it directly interpretable as a "
                "typical error magnitude. Pass actuals then predictions, `GROUP BY` for per-segment scores; "
                "NULL pairs are skipped and lower is better (0 is perfect)."
            ),
            "vgi.doc_md": (
                "**Root mean squared error (RMSE)** — typical regression error in the target's units.\n\n"
                "- Inputs: `y_true` then `y_pred` numeric columns\n"
                "- Returns: one `DOUBLE` per group, `sqrt(mean((y_true - y_pred)^2))`\n"
                "- Same outlier sensitivity as MSE but on the original scale, so it reads as an "
                "average-sized residual; lower is better, `0.0` is perfect"
            ),
        }

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
        tags = {
            "vgi.doc_llm": (
                "Aggregate over a `(y_true, y_pred)` column pair returning the mean absolute error (MAE): "
                "the average of `abs(y_true - y_pred)`. Pass actuals then predictions; `GROUP BY` to score "
                "per segment, NULL rows skipped. Unlike MSE/RMSE it weights every residual linearly, so it "
                "is more robust to outliers and reads directly as the average absolute miss in the target's "
                "units. Lower is better (0 is perfect)."
            ),
            "vgi.doc_md": (
                "**Mean absolute error (MAE)** — average absolute residual for a regression.\n\n"
                "- Inputs: `y_true` then `y_pred` numeric columns\n"
                "- Returns: one `DOUBLE` per group, `mean(|y_true - y_pred|)`\n"
                "- Linear weighting makes it more outlier-robust than MSE/RMSE; reported in the "
                "target's units, lower is better, `0.0` is perfect"
            ),
        }

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
        tags = {
            "vgi.doc_llm": (
                "Aggregate over a `(y_true, y_pred)` column pair returning the coefficient of determination "
                "R^2 — the fraction of the target's variance explained by the predictions. Pass actuals then "
                "predictions; `GROUP BY` for per-segment scores, NULL rows skipped. `1.0` is a perfect fit, "
                "`0.0` means the model does no better than always predicting the mean, and values can go "
                "negative when the fit is worse than that baseline. The standard scale-free summary of "
                "regression quality."
            ),
            "vgi.doc_md": (
                "**R^2 (coefficient of determination)** — share of target variance the model explains.\n\n"
                "- Inputs: `y_true` then `y_pred` numeric columns\n"
                "- Returns: one `DOUBLE` per group, unitless and typically in `(-inf, 1.0]`\n"
                "- `1.0` = perfect, `0.0` = no better than predicting the mean, **negative = worse "
                "than the mean baseline**; higher is better"
            ),
        }

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
        tags = {
            "vgi.doc_llm": (
                "Aggregate over a `(y_true, y_pred)` column pair returning the explained variance score: "
                "`1 - Var(y_true - y_pred) / Var(y_true)`. Pass actuals then predictions; `GROUP BY` for "
                "per-segment scores, NULL rows skipped. `1.0` is best; it resembles R^2 but ignores any "
                "constant bias in the residuals, so it can read higher than R^2 when the predictions are "
                "systematically offset. Use it to see how much variance the model captures apart from a "
                "constant shift."
            ),
            "vgi.doc_md": (
                "**Explained variance score** — variance of the target captured, ignoring constant bias.\n\n"
                "- Inputs: `y_true` then `y_pred` numeric columns\n"
                "- Returns: one `DOUBLE` per group, `1 - Var(residuals) / Var(y_true)`; higher is better\n"
                "- Differs from R^2 only when residuals have a non-zero mean (a systematic offset), "
                "which this metric forgives — compare the two to detect bias"
            ),
        }

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
        tags = {
            "vgi.doc_llm": (
                "Aggregate over a `(y_true, y_pred)` column pair returning the mean absolute percentage "
                "error (MAPE): the average of `abs((y_true - y_pred) / y_true)`. Pass actuals then "
                "predictions; `GROUP BY` for per-segment scores, NULL rows skipped. It expresses error as a "
                "scale-free fraction (e.g. `0.1` ~ 10% off on average), handy for comparing across targets "
                "of different magnitudes. Lower is better; beware that it explodes when `y_true` is near "
                "zero."
            ),
            "vgi.doc_md": (
                "**Mean absolute percentage error (MAPE)** — average relative error as a fraction.\n\n"
                "- Inputs: `y_true` then `y_pred` numeric columns\n"
                "- Returns: one `DOUBLE` per group, `mean(|(y_true - y_pred) / y_true|)` (a fraction, "
                "so `0.1` is roughly a 10% miss); lower is better\n"
                "- Scale-free, good for mixed-magnitude targets, but undefined/unstable when actuals "
                "are at or near zero"
            ),
        }

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
        tags = {
            "vgi.doc_llm": (
                "Aggregate over a `(y_true, y_pred)` column pair returning the maximum residual error: the "
                "single largest `abs(y_true - y_pred)` over the group. Pass actuals then predictions; "
                "`GROUP BY` for per-segment worst cases, NULL rows skipped. It captures the worst-case miss "
                "rather than an average, so use it when the largest deviation matters (tolerance/SLA "
                "checks). Lower is better; 0 means every prediction was exact."
            ),
            "vgi.doc_md": (
                "**Max error** — the worst single residual in the group.\n\n"
                "- Inputs: `y_true` then `y_pred` numeric columns\n"
                "- Returns: one `DOUBLE` per group, `max(|y_true - y_pred|)` in the target's units\n"
                "- A worst-case (not average) metric — use it for tolerance/SLA guarantees; lower is "
                "better, `0.0` means no prediction missed"
            ),
        }

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
        tags = {
            "vgi.doc_llm": (
                "Aggregate over a `(y_true, y_pred)` column pair returning the median absolute error: the "
                "median of `abs(y_true - y_pred)`. Pass actuals then predictions; `GROUP BY` for "
                "per-segment scores, NULL rows skipped. Because it uses the median rather than the mean, it "
                "is highly robust to outliers and reports the typical (50th-percentile) miss in the "
                "target's units. Lower is better; 0 means at least half the predictions were exact."
            ),
            "vgi.doc_md": (
                "**Median absolute error** — the typical (median) residual, outlier-robust.\n\n"
                "- Inputs: `y_true` then `y_pred` numeric columns\n"
                "- Returns: one `DOUBLE` per group, `median(|y_true - y_pred|)` in the target's units\n"
                "- Insensitive to a handful of large errors (unlike MAE/RMSE) — pair it with `max_error` "
                "to separate typical from worst-case behavior; lower is better"
            ),
        }

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
        tags = {
            "vgi.doc_llm": (
                "Aggregate over a `(y_true, y_pred)` column pair returning the mean squared logarithmic "
                "error (MSLE): the mean of `(log1p(y_true) - log1p(y_pred))**2`. Pass actuals then "
                "predictions; `GROUP BY` for per-segment scores, NULL rows skipped. Working in log space "
                "penalizes relative (ratio) error and weights under-prediction more than over-prediction, "
                "so it suits targets that span orders of magnitude (counts, prices). Requires non-negative "
                "values and lower is better."
            ),
            "vgi.doc_md": (
                "**Mean squared log error (MSLE)** — squared error in log space, for wide-range targets.\n\n"
                "- Inputs: `y_true` then `y_pred` numeric columns, both **non-negative**\n"
                "- Returns: one `DOUBLE` per group, `mean((log1p(y_true) - log1p(y_pred))^2)`\n"
                "- Penalizes relative rather than absolute error and is asymmetric (under-prediction "
                "costs more); good for counts/prices spanning orders of magnitude; lower is better"
            ),
        }

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
        tags = {
            "vgi.doc_llm": (
                "Aggregate over a `(y_true, y_pred)` column pair returning the mean pinball (quantile) loss "
                "evaluated at the median quantile (alpha=0.5). Pass actuals then predictions; `GROUP BY` for "
                "per-segment scores, NULL rows skipped. Pinball loss is the standard scoring rule for "
                "quantile regression; at alpha=0.5 it is half the mean absolute error, so it measures how "
                "well predictions track the conditional median. Lower is better."
            ),
            "vgi.doc_md": (
                "**Mean pinball loss (median quantile)** — quantile-regression scoring rule at alpha=0.5.\n\n"
                "- Inputs: `y_true` then `y_pred` numeric columns\n"
                "- Returns: one `DOUBLE` per group; lower is better\n"
                "- The proper score for a median (quantile) forecast; at alpha=0.5 it equals half the MAE, "
                "so it rewards predictions centered on the conditional median"
            ),
        }

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
        tags = {
            "vgi.doc_llm": (
                "Aggregate over a `(y_true, y_pred)` column pair returning classification accuracy: the "
                "fraction of rows where the predicted label equals the true label. Pass true labels then "
                "predicted labels (integer-coded; values are rounded to the nearest int); `GROUP BY` for "
                "per-segment scores, NULL rows skipped. Ranges 0..1, higher is better. Simple and "
                "intuitive, but misleading on imbalanced classes — prefer `balanced_accuracy_score` or "
                "`f1_score` there."
            ),
            "vgi.doc_md": (
                "**Accuracy** — share of correctly classified rows.\n\n"
                "- Inputs: `y_true` then `y_pred` integer-coded label columns\n"
                "- Returns: one `DOUBLE` in `[0, 1]` per group; higher is better\n"
                "- Can overstate quality on imbalanced data (a constant majority-class predictor scores "
                "high) — cross-check with balanced accuracy, F1, or MCC"
            ),
        }

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
        tags = {
            "vgi.doc_llm": (
                "Aggregate over a `(y_true, y_pred)` column pair returning precision: `TP / (TP + FP)`, the "
                "fraction of predicted positives that were actually positive. Pass true labels then "
                "predicted labels (integer-coded); multiclass is macro-averaged (unweighted mean over "
                "classes) and undefined classes score 0. `GROUP BY` for per-segment scores, NULL rows "
                "skipped. Ranges 0..1, higher is better — use it when false positives are costly."
            ),
            "vgi.doc_md": (
                "**Precision** — of the rows predicted positive, how many truly are.\n\n"
                "- Inputs: `y_true` then `y_pred` integer-coded label columns\n"
                "- Returns: one `DOUBLE` in `[0, 1]` per group; macro-averaged across classes for "
                "multiclass; higher is better\n"
                "- Penalizes **false positives**; pair with `recall_score` (their harmonic mean is "
                "`f1_score`) for the full trade-off"
            ),
        }

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
        tags = {
            "vgi.doc_llm": (
                "Aggregate over a `(y_true, y_pred)` column pair returning recall (sensitivity / true "
                "positive rate): `TP / (TP + FN)`, the fraction of actual positives the model caught. Pass "
                "true labels then predicted labels (integer-coded); multiclass is macro-averaged and "
                "undefined classes score 0. `GROUP BY` for per-segment scores, NULL rows skipped. Ranges "
                "0..1, higher is better — use it when missing positives (false negatives) is costly."
            ),
            "vgi.doc_md": (
                "**Recall (sensitivity)** — of the truly positive rows, how many were caught.\n\n"
                "- Inputs: `y_true` then `y_pred` integer-coded label columns\n"
                "- Returns: one `DOUBLE` in `[0, 1]` per group; macro-averaged across classes for "
                "multiclass; higher is better\n"
                "- Penalizes **false negatives**; trades off against precision, and `f1_score` is their "
                "harmonic mean"
            ),
        }

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
        tags = {
            "vgi.doc_llm": (
                "Aggregate over a `(y_true, y_pred)` column pair returning the F1 score: the harmonic mean "
                "of precision and recall, `2*P*R / (P + R)`. Pass true labels then predicted labels "
                "(integer-coded); multiclass is macro-averaged (unweighted mean over classes) and "
                "undefined classes score 0. `GROUP BY` for per-segment scores, NULL rows skipped. Ranges "
                "0..1, higher is better — the go-to single-number classifier metric for imbalanced data."
            ),
            "vgi.doc_md": (
                "**F1 score** — harmonic mean of precision and recall.\n\n"
                "- Inputs: `y_true` then `y_pred` integer-coded label columns\n"
                "- Returns: one `DOUBLE` in `[0, 1]` per group; macro-averaged across classes for "
                "multiclass; higher is better\n"
                "- Balances false positives and false negatives in one number and is harsh when either "
                "precision or recall is low, making it a robust default on imbalanced classes"
            ),
        }

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
        tags = {
            "vgi.doc_llm": (
                "Aggregate over a `(y_true, y_pred)` column pair returning balanced accuracy: the average "
                "of per-class recall (the mean true-positive rate across classes). Pass true labels then "
                "predicted labels (integer-coded); `GROUP BY` for per-segment scores, NULL rows skipped. "
                "Because each class counts equally regardless of size, it corrects plain accuracy's bias on "
                "imbalanced data — a majority-class-only predictor scores ~0.5 on a balanced two-class "
                "problem. Higher is better."
            ),
            "vgi.doc_md": (
                "**Balanced accuracy** — accuracy that weights every class equally.\n\n"
                "- Inputs: `y_true` then `y_pred` integer-coded label columns\n"
                "- Returns: one `DOUBLE` per group, the mean of per-class recall; higher is better\n"
                "- Unlike plain accuracy it is not fooled by class imbalance: a trivial majority-class "
                "predictor lands near chance (`0.5` for two equal classes)"
            ),
        }

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
        tags = {
            "vgi.doc_llm": (
                "Aggregate over a `(y_true, y_pred)` column pair returning the Matthews correlation "
                "coefficient (MCC): a correlation between true and predicted labels computed from all four "
                "confusion-matrix counts. Pass true labels then predicted labels (integer-coded); "
                "`GROUP BY` for per-segment scores, NULL rows skipped. Ranges -1..+1 (+1 perfect, 0 random, "
                "-1 total disagreement). Widely regarded as the most informative single classification "
                "metric because it stays meaningful even under heavy class imbalance."
            ),
            "vgi.doc_md": (
                "**Matthews correlation coefficient (MCC)** — balanced correlation of labels.\n\n"
                "- Inputs: `y_true` then `y_pred` integer-coded label columns\n"
                "- Returns: one `DOUBLE` in `[-1, 1]` per group: `+1` perfect, `0` no better than "
                "random, `-1` perfectly wrong\n"
                "- Uses all of TP/TN/FP/FN, so it remains reliable on imbalanced classes where accuracy "
                "and even F1 can mislead"
            ),
        }

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
        tags = {
            "vgi.doc_llm": (
                "Aggregate over a `(y_true, y_pred)` column pair returning Cohen's kappa: the agreement "
                "between the two label columns corrected for the agreement expected by chance. Pass the two "
                "integer-coded label columns (true and predicted, or any two raters); `GROUP BY` for "
                "per-segment scores, NULL rows skipped. Ranges up to +1 (perfect), 0 means agreement no "
                "better than chance, and it can go negative. Use it to score classifiers or inter-annotator "
                "reliability beyond raw accuracy."
            ),
            "vgi.doc_md": (
                "**Cohen's kappa** — chance-corrected agreement between two labelings.\n\n"
                "- Inputs: two integer-coded label columns (`y_true` then `y_pred`, or any two raters)\n"
                "- Returns: one `DOUBLE` (typically `[-1, 1]`) per group; `1` perfect, `0` chance-level, "
                "negative is worse than chance\n"
                "- Discounts the agreement you'd expect at random, so it is stricter than raw accuracy "
                "and standard for inter-rater reliability"
            ),
        }

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
        tags = {
            "vgi.doc_llm": (
                "Aggregate over a `(y_true, y_pred)` column pair returning the Jaccard similarity "
                "coefficient: intersection over union of the predicted and true positive sets, "
                "`TP / (TP + FP + FN)`. Pass true labels then predicted labels (integer-coded); multiclass "
                "is macro-averaged and undefined classes score 0. `GROUP BY` for per-segment scores, NULL "
                "rows skipped. Ranges 0..1, higher is better — common for set-overlap and "
                "segmentation-style tasks."
            ),
            "vgi.doc_md": (
                "**Jaccard score** — intersection-over-union of true and predicted positives.\n\n"
                "- Inputs: `y_true` then `y_pred` integer-coded label columns\n"
                "- Returns: one `DOUBLE` in `[0, 1]` per group, `TP / (TP + FP + FN)`; macro-averaged "
                "for multiclass; higher is better\n"
                "- Excludes true negatives (unlike accuracy), so it focuses on overlap of the positive "
                "sets — handy for tagging and segmentation evaluation"
            ),
        }

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
        tags = {
            "vgi.doc_llm": (
                "Aggregate over a `(y_true, y_pred)` column pair returning the Hamming loss: the fraction "
                "of rows whose predicted label differs from the true label. Pass true labels then predicted "
                "labels (integer-coded); `GROUP BY` for per-segment scores, NULL rows skipped. For "
                "single-label classification it is exactly `1 - accuracy`. Ranges 0..1, lower is better "
                "(0 = every label correct)."
            ),
            "vgi.doc_md": (
                "**Hamming loss** — fraction of labels predicted wrong.\n\n"
                "- Inputs: `y_true` then `y_pred` integer-coded label columns\n"
                "- Returns: one `DOUBLE` in `[0, 1]` per group; **lower is better**, `0.0` is perfect\n"
                "- The complement of accuracy for single-label tasks (`hamming_loss = 1 - accuracy`)"
            ),
        }

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
        tags = {
            "vgi.doc_llm": (
                "Aggregate over a `(y_true, y_pred)` column pair returning the zero-one loss: the fraction "
                "of misclassified rows (each row scores 0 if correct, 1 if wrong, then averaged). Pass true "
                "labels then predicted labels (integer-coded); `GROUP BY` for per-segment scores, NULL rows "
                "skipped. It equals `1 - accuracy`. Ranges 0..1, lower is better (0 = no misclassifications)."
            ),
            "vgi.doc_md": (
                "**Zero-one loss** — proportion of misclassified rows.\n\n"
                "- Inputs: `y_true` then `y_pred` integer-coded label columns\n"
                "- Returns: one `DOUBLE` in `[0, 1]` per group; **lower is better**, `0.0` is perfect\n"
                "- Equivalent to `1 - accuracy`; the error-rate framing of the same quantity"
            ),
        }

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
        tags = {
            "vgi.doc_llm": (
                "Aggregate over a `(y_true, y_pred)` column pair returning the area under the ROC curve "
                "(AUC) for a binary classifier. Pass the true 0/1 labels first and a continuous "
                "score/probability (not a hard label) second; `GROUP BY` for per-segment scores, NULL rows "
                "skipped. AUC equals the probability that a random positive is ranked above a random "
                "negative: 1.0 is perfect ranking, 0.5 is random, below 0.5 is inverted. Threshold-free, so "
                "it measures ranking quality independent of any cutoff."
            ),
            "vgi.doc_md": (
                "**ROC AUC** — threshold-free ranking quality of a binary scorer.\n\n"
                "- Inputs: `y_true` (0/1 labels) then `y_pred` (a **score or probability**, not a class)\n"
                "- Returns: one `DOUBLE` in `[0, 1]` per group; `0.5` is random, `1.0` is perfect, `<0.5` "
                "is worse-than-random ranking\n"
                "- Interpretable as P(a random positive outranks a random negative); independent of the "
                "decision threshold"
            ),
        }

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
        tags = {
            "vgi.doc_llm": (
                "Aggregate over a `(y_true, y_pred)` column pair returning average precision (AP): a "
                "summary of the precision-recall curve as the precision-weighted mean over recall "
                "thresholds. Pass the true 0/1 labels first and a continuous score/probability second; "
                "`GROUP BY` for per-segment scores, NULL rows skipped. Ranges 0..1, higher is better, with "
                "a no-skill baseline equal to the positive-class prevalence. Prefer it over ROC AUC when "
                "positives are rare and you care about the top-ranked results."
            ),
            "vgi.doc_md": (
                "**Average precision (AP)** — area under the precision-recall curve.\n\n"
                "- Inputs: `y_true` (0/1 labels) then `y_pred` (a **score or probability**)\n"
                "- Returns: one `DOUBLE` in `[0, 1]` per group; higher is better, baseline = positive-"
                "class fraction\n"
                "- More informative than ROC AUC under heavy class imbalance because it ignores true "
                "negatives and rewards ranking positives near the top"
            ),
        }

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
        tags = {
            "vgi.doc_llm": (
                "Aggregate over a `(y_true, y_pred)` column pair returning the logistic / cross-entropy "
                "loss for binary probabilities. Pass the true 0/1 labels first and the predicted "
                "probability of class 1 second (clipped away from 0/1 for numerical stability); `GROUP BY` "
                "for per-segment scores, NULL rows skipped. It is a proper scoring rule that punishes "
                "confident wrong predictions sharply (unbounded above). Lower is better (0 is perfect, "
                "well-calibrated certainty); use it to evaluate predicted probabilities, not hard labels."
            ),
            "vgi.doc_md": (
                "**Log loss (binary cross-entropy)** — penalty on predicted probabilities.\n\n"
                "- Inputs: `y_true` (0/1 labels) then `y_pred` = **P(class 1)**\n"
                "- Returns: one `DOUBLE` >= 0 per group; **lower is better**, unbounded above\n"
                "- A proper scoring rule: confident-and-wrong predictions are punished heavily, so it "
                "rewards calibrated probabilities rather than just correct rankings"
            ),
        }

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
        tags = {
            "vgi.doc_llm": (
                "Aggregate over a `(y_true, y_pred)` column pair returning the Brier score: the mean "
                "squared error between predicted probabilities and 0/1 outcomes, `mean((y_pred - "
                "y_true)**2)`. Pass the true 0/1 labels first and the predicted probability of class 1 "
                "second; `GROUP BY` for per-segment scores, NULL rows skipped. Ranges 0..1, lower is better "
                "(0 = perfectly calibrated and confident). It measures probability calibration and is "
                "gentler on confident errors than log loss."
            ),
            "vgi.doc_md": (
                "**Brier score** — squared error of probability forecasts.\n\n"
                "- Inputs: `y_true` (0/1 labels) then `y_pred` = **P(class 1)**\n"
                "- Returns: one `DOUBLE` in `[0, 1]` per group; **lower is better**, `0.0` is perfect\n"
                "- Like log loss it scores calibrated probabilities, but the squared (vs. logarithmic) "
                "penalty is bounded and less punishing of confident mistakes"
            ),
        }

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
        tags = {
            "vgi.doc_llm": (
                "Aggregate over a pair of integer cluster-label columns returning the Adjusted Rand Index "
                "(ARI): how well two clusterings (e.g. ground-truth labels vs. predicted clusters) agree on "
                "which point pairs are grouped together, corrected for chance. Pass the two label columns; "
                "`GROUP BY` for per-segment scores, NULL rows skipped. Ranges up to +1 (identical "
                "clusterings), ~0 for random labelings, and can be negative. Label-permutation invariant, "
                "so cluster id values need not match across the two columns."
            ),
            "vgi.doc_md": (
                "**Adjusted Rand Index (ARI)** — chance-corrected agreement of two clusterings.\n\n"
                "- Inputs: two integer cluster-label columns (e.g. true vs. predicted clusters)\n"
                "- Returns: one `DOUBLE` (typically `[-0.5, 1]`) per group; `1` identical, `~0` random, "
                "negative is worse than chance\n"
                "- Compares co-grouping of point pairs and is invariant to how the clusters are "
                "numbered, so the two columns need no shared labeling"
            ),
        }

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
        tags = {
            "vgi.doc_llm": (
                "Aggregate over a pair of integer cluster-label columns returning normalized mutual "
                "information (NMI): the mutual information between two clusterings rescaled to 0..1 by the "
                "label entropies. Pass the two label columns; `GROUP BY` for per-segment scores, NULL rows "
                "skipped. `1.0` means the two clusterings are informationally identical, `0.0` means "
                "independent. Label-permutation invariant; unlike the adjusted variant it is NOT corrected "
                "for chance, so it trends upward with more clusters."
            ),
            "vgi.doc_md": (
                "**Normalized mutual information (NMI)** — shared information between two clusterings.\n\n"
                "- Inputs: two integer cluster-label columns\n"
                "- Returns: one `DOUBLE` in `[0, 1]` per group; `1` informationally identical, `0` "
                "independent; higher is better\n"
                "- Not chance-corrected (it tends to rise with the number of clusters) — use "
                "`adjusted_mutual_info_score` when comparing clusterings of differing granularity"
            ),
        }

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
        tags = {
            "vgi.doc_llm": (
                "Aggregate over a pair of integer cluster-label columns returning adjusted mutual "
                "information (AMI): normalized mutual information corrected for the agreement expected by "
                "chance. Pass the two label columns; `GROUP BY` for per-segment scores, NULL rows skipped. "
                "`1.0` means identical clusterings, ~0 means agreement no better than random, and it can go "
                "slightly negative. Label-permutation invariant and, unlike NMI, fair to compare across "
                "clusterings with different numbers of clusters."
            ),
            "vgi.doc_md": (
                "**Adjusted mutual information (AMI)** — chance-corrected NMI.\n\n"
                "- Inputs: two integer cluster-label columns\n"
                "- Returns: one `DOUBLE` (max `1`) per group; `~0` is chance-level, higher is better\n"
                "- Removes the upward bias NMI has with more clusters, so it is the right choice when "
                "the two clusterings differ in granularity"
            ),
        }

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
        tags = {
            "vgi.doc_llm": (
                "Aggregate over `(y_true, y_pred)` cluster-label columns returning homogeneity: 1.0 when "
                "every predicted cluster contains only members of a single ground-truth class. Pass the "
                "true class labels first and the predicted cluster ids second; `GROUP BY` for per-segment "
                "scores, NULL rows skipped. Ranges 0..1, higher is better. It is the 'purity' half of the "
                "homogeneity/completeness pair (their harmonic mean is `v_measure_score`); it does not "
                "penalize splitting one class across many clusters."
            ),
            "vgi.doc_md": (
                "**Homogeneity** — are predicted clusters class-pure?\n\n"
                "- Inputs: `y_true` (ground-truth classes) then `y_pred` (predicted cluster ids)\n"
                "- Returns: one `DOUBLE` in `[0, 1]` per group; `1.0` = each cluster holds a single "
                "class; higher is better\n"
                "- Cares only about purity, not whether a class is spread across clusters — that is "
                "`completeness_score`; combine both via `v_measure_score`"
            ),
        }

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
        tags = {
            "vgi.doc_llm": (
                "Aggregate over `(y_true, y_pred)` cluster-label columns returning completeness: 1.0 when "
                "all members of each ground-truth class are assigned to the same predicted cluster. Pass "
                "the true class labels first and the predicted cluster ids second; `GROUP BY` for "
                "per-segment scores, NULL rows skipped. Ranges 0..1, higher is better. It is the dual of "
                "homogeneity — it does not penalize lumping several classes into one cluster — and the two "
                "combine into `v_measure_score`."
            ),
            "vgi.doc_md": (
                "**Completeness** — is each class kept together in one cluster?\n\n"
                "- Inputs: `y_true` (ground-truth classes) then `y_pred` (predicted cluster ids)\n"
                "- Returns: one `DOUBLE` in `[0, 1]` per group; `1.0` = no class is split across "
                "clusters; higher is better\n"
                "- The mirror image of `homogeneity_score` (it tolerates mixing classes within a "
                "cluster); their harmonic mean is `v_measure_score`"
            ),
        }

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
        tags = {
            "vgi.doc_llm": (
                "Aggregate over `(y_true, y_pred)` cluster-label columns returning the V-measure: the "
                "harmonic mean of homogeneity (clusters are class-pure) and completeness (each class stays "
                "in one cluster). Pass the true class labels first and the predicted cluster ids second; "
                "`GROUP BY` for per-segment scores, NULL rows skipped. Ranges 0..1, higher is better, "
                "symmetric in the two inputs. A single balanced score for how well a clustering matches "
                "known labels (equivalent to NMI with arithmetic averaging)."
            ),
            "vgi.doc_md": (
                "**V-measure** — balanced clustering-vs-truth score.\n\n"
                "- Inputs: `y_true` (ground-truth classes) then `y_pred` (predicted cluster ids)\n"
                "- Returns: one `DOUBLE` in `[0, 1]` per group; higher is better\n"
                "- Harmonic mean of `homogeneity_score` and `completeness_score`, so it is low unless "
                "both hold; symmetric and equal to NMI under arithmetic normalization"
            ),
        }

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
        tags = {
            "vgi.doc_llm": (
                "Aggregate over a pair of integer cluster-label columns returning the Fowlkes-Mallows index "
                "(FMI): the geometric mean of the pairwise precision and recall of co-grouped point pairs "
                "between two clusterings. Pass the two label columns; `GROUP BY` for per-segment scores, "
                "NULL rows skipped. Ranges 0..1, higher is better (1.0 = identical clusterings). "
                "Label-permutation invariant; useful as an alternative to ARI that stays non-negative."
            ),
            "vgi.doc_md": (
                "**Fowlkes-Mallows index (FMI)** — pairwise precision/recall agreement of two clusterings.\n\n"
                "- Inputs: two integer cluster-label columns\n"
                "- Returns: one `DOUBLE` in `[0, 1]` per group, the geometric mean of pair precision and "
                "recall; higher is better, `1.0` identical\n"
                "- Built on whether point pairs are co-grouped, label-invariant, and (unlike ARI) "
                "bounded below at 0"
            ),
        }

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
