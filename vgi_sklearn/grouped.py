"""Per-group modeling: fit one model per ``GROUP BY`` key, predict per row.

This pairs an **aggregate** with **scalar** functions to sidestep table-function
limits (no correlated/lateral args), and is the natural fit for "a model per
segment" research:

* ``fit_model`` -- an aggregate, so ``GROUP BY`` does the partitioning. Each
  group's rows are buffered, an estimator is fit on them, and one ``STRUCT``
  (the model as a BLOB plus diagnostics) is returned per group.
* ``predict_one`` / ``predict_class_one`` / ``predict_proba_one`` -- scalars that
  take a per-row model BLOB and a feature ``STRUCT``, so you can score each row
  with the model for *its* group (a plain join), or even a different model per
  row.

    -- a model per region
    CREATE TABLE models AS
      SELECT region, sklearn.models.fit_model({'tenure': tenure, 'spend': spend}, churned,
                                       estimator := 'gradient_boosting_classifier') AS m
      FROM customers GROUP BY region;

    -- score each customer with their region's model
    SELECT c.id, sklearn.models.predict_class_one(m.m.model, {'tenure': c.tenure, 'spend': c.spend})
    FROM customers c JOIN models m USING (region);

Features are passed as a ``STRUCT`` (named, so alignment is by name), the target
as any type (numeric for regression; numeric **or** string labels for
classification -- the label dtype is preserved). Hyperparameters are a JSON
string (aggregate args are constant scalars).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Annotated, Any

import numpy as np
import pyarrow as pa
import sklearn
from vgi.aggregate_function import AggregateFunction
from vgi.arguments import ConstParam, Param, Returns
from vgi.metadata import FunctionExample
from vgi.scalar_function import ScalarFunction
from vgi.table_function import ProcessParams
from vgi_rpc import ArrowSerializableDataclass

from .features import build_x, categorical_mask, wrap_estimator
from .models import CLASSIFICATION, build_estimator
from .registry import ModelMetadata, now_iso, pack_model, unpack_model

# ---------------------------------------------------------------------------
# Feature / target extraction
# ---------------------------------------------------------------------------


def _struct_rows(features: pa.Array) -> tuple[list[str], list[list[Any]], list[bool]]:
    """Return (ordered feature names, raw rows, categorical mask) from a struct.

    String fields are kept as strings (categorical); numeric/boolean fields are
    coerced to floats. ``fit_group`` one-hot-encodes the categorical columns.
    """
    if not pa.types.is_struct(features.type):
        raise ValueError("features must be a STRUCT, e.g. {'a': a, 'b': b}")
    names = [features.type.field(i).name for i in range(features.type.num_fields)]
    cat_mask = categorical_mask([features.type.field(i).type for i in range(features.type.num_fields)])
    rows: list[list[Any]] = []
    for rec in features.to_pylist():
        rec = rec or {}
        row: list[Any] = []
        for n, is_cat in zip(names, cat_mask, strict=True):
            v = rec.get(n)
            if is_cat:
                row.append(None if v is None else str(v))
            else:
                try:
                    row.append(float(v) if v is not None else float("nan"))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"feature {n!r} is not numeric (got {v!r})") from exc
        rows.append(row)
    return names, rows, cat_mask


def _matrix_for(model_features: list[str], features: pa.Array, cat_mask: list[bool]) -> np.ndarray:
    """Build the prediction feature matrix, aligning struct fields by name."""
    if not pa.types.is_struct(features.type):
        raise ValueError("features must be a STRUCT")
    present = {features.type.field(i).name for i in range(features.type.num_fields)}
    missing = [f for f in model_features if f not in present]
    if missing:
        raise ValueError(f"features struct is missing model column(s): {', '.join(missing)}")
    recs = features.to_pylist()
    rows = [[(rec or {}).get(name) for name in model_features] for rec in recs]
    return build_x(rows, cat_mask or [False] * len(model_features))


# ---------------------------------------------------------------------------
# fit_model (aggregate)
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class FitState(ArrowSerializableDataclass):
    """Per-group accumulation: each ``update`` appends one JSON chunk of its rows.

    Storing per-batch chunks (rather than one growing list) keeps the
    reassign-to-persist cost proportional to the number of batches, not rows.
    """

    chunks: list[bytes] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    categorical: list[bool] = field(default_factory=list)
    target_numeric: bool = True
    estimator: str = ""
    params: str = ""


_FIT_RESULT = pa.struct(
    [
        pa.field("model", pa.binary()),
        pa.field("estimator", pa.string()),
        pa.field("task", pa.string()),
        pa.field("n_samples", pa.int64()),
        pa.field("n_features", pa.int64()),
        pa.field("n_classes", pa.int64()),
        pa.field("train_score", pa.float64()),
    ]
)


class FitModel(AggregateFunction[FitState]):
    """Fit one estimator per ``GROUP BY`` group; emit the model + diagnostics."""

    class Meta:
        """VGI metadata for the fit_model aggregate."""

        name = "fit_model"
        description = "Fit one model per group (aggregate); returns the model BLOB + diagnostics"
        categories = ["models", "supervised", "grouped"]
        tags = {
            "vgi.doc_llm": (
                "Aggregate function that fits one estimator **per `GROUP BY` group** and returns a `STRUCT` "
                "(the model as a BLOB plus diagnostics: `estimator`, `task`, `n_samples`, `n_features`, "
                "`n_classes`, `train_score`) for each group. Pass a feature `STRUCT` (e.g. "
                "`{'tenure': tenure, 'spend': spend}` — features align by name), then the `target` column (any "
                "type: numeric for regression, numeric **or** string labels for classification, dtype "
                "preserved), then `estimator :=` and the required `hyperparams :=` JSON (`'{}'` for defaults). "
                "`GROUP BY` does the partitioning, so this is the way to build 'a model per segment'; score "
                "the rows with the `predict_one`/`predict_class_one`/`predict_proba_one` scalars on the "
                "returned `.model` BLOB."
            ),
            "vgi.doc_md": (
                "**fit_model** — fit a separate model for each `GROUP BY` group (an aggregate).\n\n"
                "Buffers each group's rows, fits an estimator, and emits one result `STRUCT` per group; "
                "`GROUP BY` supplies the partitioning that table functions cannot.\n\n"
                "- Args: feature `STRUCT` (by-name alignment); `target` (numeric or string labels, dtype "
                "preserved); `estimator :=` name; `hyperparams :=` JSON (**required**, pass `'{}'`)\n"
                "- Returns per group: a `STRUCT(model BLOB, estimator, task, n_samples, n_features, "
                "n_classes, train_score)`\n"
                "- The per-segment modeling entry point; feed the `.model` BLOB to `predict_one` / "
                "`predict_class_one` / `predict_proba_one` to score rows with their group's model"
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    "SELECT region, sklearn.models.fit_model("
                    "{'tenure': tenure, 'spend': spend}, churned, "
                    "estimator := 'gradient_boosting_classifier', hyperparams := '{}') AS m "
                    "FROM (VALUES ('east', 12, 30.0, 0), ('east', 3, 80.0, 1), "
                    "('west', 24, 20.0, 0), ('west', 1, 90.0, 1)) "
                    "AS customers(region, tenure, spend, churned) GROUP BY region"
                ),
                description="One churn model per region",
            )
        ]

    @classmethod
    def initial_state(cls, params: ProcessParams[None]) -> FitState:
        """Start each group with an empty accumulation state."""
        return FitState()

    @classmethod
    def update(
        cls,
        states: dict[int, FitState],
        group_ids: pa.Int64Array,
        features: Annotated[pa.Array, Param(doc="Feature values, one field per feature, e.g. {'a': a, 'b': b}")],
        target: Annotated[pa.Array, Param(doc="Target column (continuous values, or string class labels)")],
        estimator: Annotated[str, ConstParam(doc="Estimator name, e.g. 'random_forest_classifier'")],
        hyperparams: Annotated[str, ConstParam(doc="JSON hyperparameters; '{}' for defaults")],
    ) -> None:
        """Buffer this batch's rows into each touched group's state."""
        names, rows, cat_mask = _struct_rows(features)
        target_numeric = pa.types.is_floating(target.type) or pa.types.is_integer(target.type)
        tvals = target.to_pylist()

        # Group this batch, then reassign each touched group's state (the
        # framework only persists groups you assign — see CLAUDE.md edge #1).
        per_group: dict[int, tuple[list[list[Any]], list[Any]]] = {}
        for g, row, t in zip(group_ids.to_pylist(), rows, tvals, strict=True):
            fr, tg = per_group.setdefault(g, ([], []))
            fr.append(row)
            tg.append(t)
        for g, (fr, tg) in per_group.items():
            s = states[g]
            chunk = json.dumps({"f": fr, "t": tg}).encode("utf-8")
            states[g] = FitState(
                chunks=[*s.chunks, chunk],
                feature_names=names,
                categorical=cat_mask,
                target_numeric=target_numeric,
                estimator=estimator,
                params=hyperparams or "",
            )

    @classmethod
    def combine(cls, source: FitState, target: FitState, params: ProcessParams[None]) -> FitState:
        """Merge two partial states for the same group (concatenate chunks)."""
        return FitState(
            chunks=source.chunks + target.chunks,
            feature_names=source.feature_names or target.feature_names,
            categorical=source.categorical or target.categorical,
            target_numeric=source.target_numeric and target.target_numeric,
            estimator=source.estimator or target.estimator,
            params=source.params or target.params,
        )

    @classmethod
    def finalize(
        cls,
        group_ids: pa.Int64Array,
        states: dict[int, FitState],
        params: ProcessParams[None],
    ) -> Annotated[pa.RecordBatch, Returns(_FIT_RESULT)]:
        """Fit each group's model and emit one result struct per group."""
        results: list[dict[str, Any] | None] = []
        for gid in group_ids:
            s = states.get(gid.as_py())
            results.append(None if s is None or not s.chunks else cls._fit_group(s))
        return pa.record_batch({"result": pa.array(results, type=_FIT_RESULT)})

    @classmethod
    def _fit_group(cls, s: FitState) -> dict[str, Any]:
        rows: list[list[float]] = []
        targets: list[Any] = []
        for chunk in s.chunks:
            payload = json.loads(chunk)
            rows.extend(payload["f"])
            targets.extend(payload["t"])

        task, estimator = build_estimator(s.estimator, _parse(s.params))
        x = build_x(rows, s.categorical)
        estimator = wrap_estimator(estimator, s.categorical)
        if task == CLASSIFICATION:
            y = (
                np.asarray([str(t) for t in targets])
                if not s.target_numeric
                else np.rint(np.asarray(targets, dtype=float)).astype(int)
            )
        else:
            y = np.asarray(targets, dtype=float)

        estimator.fit(x, y)
        classes = (
            [c.item() if hasattr(c, "item") else c for c in estimator.classes_] if task == CLASSIFICATION else None
        )
        meta = ModelMetadata(
            name="",
            estimator=s.estimator,
            task=task,
            target="",
            feature_names=s.feature_names,
            categorical=s.categorical,
            params=_parse(s.params),
            classes=classes,
            n_samples=int(x.shape[0]),
            n_features=int(x.shape[1]),
            train_score=float(estimator.score(x, y)),
            sklearn_version=sklearn.__version__,
            created_at=now_iso(),
        )
        return {
            "model": pack_model(estimator, meta),
            "estimator": s.estimator,
            "task": task,
            "n_samples": meta.n_samples,
            "n_features": meta.n_features,
            "n_classes": len(classes) if classes is not None else None,
            "train_score": meta.train_score,
        }


def _parse(params: str) -> dict[str, Any]:
    params = (params or "").strip()
    return json.loads(params) if params else {}


# ---------------------------------------------------------------------------
# predict scalars (per-row, model BLOB + feature STRUCT)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=128)
def _load(blob: bytes) -> tuple[Any, ModelMetadata]:
    """Deserialize a model BLOB, cached by identity so a per-group model loads once."""
    return unpack_model(blob)


def _predict_values(model: pa.Array, features: pa.Array, *, proba: bool = False) -> list[Any]:
    # rows often share a model (per-group join), so group by blob to predict in batches
    blobs = model.to_pylist()
    by_blob: dict[bytes, list[int]] = {}
    for i, b in enumerate(blobs):
        by_blob.setdefault(b, []).append(i)
    results: list[Any] = [None] * len(blobs)
    for blob, idxs in by_blob.items():
        if blob is None:
            continue
        est, meta = _load(blob)
        sub = features.take(pa.array(idxs))
        x = _matrix_for(meta.feature_names, sub, meta.categorical)
        preds = est.predict_proba(x).tolist() if proba else est.predict(x).tolist()
        for k, i in enumerate(idxs):
            results[i] = preds[k]
    return results


class PredictOne(ScalarFunction):
    """Predict one numeric value per row (regression, or numeric class labels)."""

    class Meta:
        """VGI metadata for the predict_one scalar."""

        name = "predict_one"
        description = "Score a row through a model BLOB; returns a numeric prediction"
        categories = ["models", "inference", "grouped"]
        tags = {
            "vgi.doc_llm": (
                "Scalar function that scores one row through a per-row model BLOB and returns a `DOUBLE` "
                "prediction — for regression, or numeric class labels. Pass the `model` BLOB (column 1, "
                "typically `m.m.model` from a `fit_model` join, so each row uses its own group's model) and a "
                "feature `STRUCT` (column 2; fields align to the model's features by name, extras ignored). "
                "Because it is a scalar it composes in any `SELECT`/join — the usual pattern joins rows to a "
                "per-group models table and scores each with its group's model. Loading is cached by BLOB "
                "identity, so a shared model deserializes once. Use `predict_class_one` for string labels or "
                "`predict_proba_one` for probabilities."
            ),
            "vgi.doc_md": (
                "**predict_one** — score a row with a model BLOB, returning a numeric prediction.\n\n"
                "- Args: `model` BLOB (per-row, e.g. from a `fit_model` join); feature `STRUCT` (by-name "
                "alignment, extra fields ignored)\n"
                "- Returns: a `DOUBLE` prediction per row (regression or numeric class labels)\n"
                "- A scalar, so it joins each row to its group's model and scores it inline; rows sharing a "
                "BLOB deserialize it once — use `predict_class_one`/`predict_proba_one` for text labels or "
                "class probabilities"
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    "SELECT sklearn.models.predict_one(m.m.model, {'tenure': c.tenure, 'spend': c.spend}) "
                    "FROM customers c JOIN models m USING (region)"
                ),
                description="Score each customer with their group's regression model",
            )
        ]

    @classmethod
    def compute(
        cls,
        model: Annotated[pa.BinaryArray, Param(doc="A fitted model (from fit_model / fit)")],
        features: Annotated[pa.Array, Param(doc="Feature values, one field per feature")],
    ) -> Annotated[pa.DoubleArray, Returns(pa.float64())]:
        """Score each row through its model BLOB and return numeric predictions."""
        vals = _predict_values(model, features)
        return pa.array([None if v is None else float(v) for v in vals], type=pa.float64())


class PredictClassOne(ScalarFunction):
    """Predict the class label per row as text (supports string labels)."""

    class Meta:
        """VGI metadata for the predict_class_one scalar."""

        name = "predict_class_one"
        description = "Score a row through a classifier BLOB; returns the class label as text"
        categories = ["models", "inference", "grouped"]
        tags = {
            "vgi.doc_llm": (
                "Scalar function that scores one row through a per-row classifier model BLOB and returns the "
                "predicted class label **as text** (`VARCHAR`) — so it works for string class labels that "
                "`predict_one` (numeric only) cannot represent. Pass the `model` BLOB (column 1, usually "
                "`m.m.model` from a `fit_model` join) and a feature `STRUCT` (column 2; fields align by name). "
                "Returns one label string per row; loading is cached by BLOB identity. Use it in the "
                "per-group join pattern when each segment's classifier emits string labels; for raw "
                "per-class probabilities use `predict_proba_one`."
            ),
            "vgi.doc_md": (
                "**predict_class_one** — score a row with a classifier BLOB, returning the label as text.\n\n"
                "- Args: `model` BLOB (per-row, e.g. from a `fit_model` join); feature `STRUCT` (by-name "
                "alignment)\n"
                "- Returns: the predicted class label as `VARCHAR` per row\n"
                "- The string-label counterpart to `predict_one` (which is numeric-only), so it preserves "
                "text classes; pair with `predict_proba_one` when you need per-class probabilities"
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    "SELECT sklearn.models.predict_class_one(m.m.model, {'tenure': c.tenure, 'spend': c.spend}) "
                    "FROM customers c JOIN models m USING (region)"
                ),
                description="Predict each customer's class label with their group's classifier",
            )
        ]

    @classmethod
    def compute(
        cls,
        model: Annotated[pa.BinaryArray, Param(doc="A fitted classifier model")],
        features: Annotated[pa.Array, Param(doc="Feature values, one field per feature")],
    ) -> Annotated[pa.StringArray, Returns(pa.string())]:
        """Score each row through its classifier BLOB and return text class labels."""
        vals = _predict_values(model, features)
        return pa.array([None if v is None else str(v) for v in vals], type=pa.string())


class PredictProbaOne(ScalarFunction):
    """Predict per-class probabilities per row, in the model's class order."""

    class Meta:
        """VGI metadata for the predict_proba_one scalar."""

        name = "predict_proba_one"
        description = "Class probabilities for a row through a classifier BLOB (DOUBLE[])"
        categories = ["models", "inference", "grouped"]
        tags = {
            "vgi.doc_llm": (
                "Scalar function that scores one row through a per-row classifier model BLOB and returns the "
                "per-class probabilities as a `DOUBLE[]` (a list), ordered by the model's own class order. "
                "Pass the `model` BLOB (column 1, e.g. `m.m.model` from a `fit_model` join) and a feature "
                "`STRUCT` (column 2; fields align by name). Use it when you need confidences rather than a "
                "single label — index or `unnest` the array to get individual `P(class)` values; loading is "
                "cached by BLOB identity. Use `predict_class_one` if you only need the hard label."
            ),
            "vgi.doc_md": (
                "**predict_proba_one** — per-class probabilities for a row through a classifier BLOB.\n\n"
                "- Args: `model` BLOB (per-row, e.g. from a `fit_model` join); feature `STRUCT` (by-name "
                "alignment)\n"
                "- Returns: a `DOUBLE[]` of probabilities in the model's class order, one per row\n"
                "- For confidence/soft outputs rather than a hard label (`predict_class_one`); `unnest` or "
                "index the array to pull out individual `P(class)` values"
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    "SELECT sklearn.models.predict_proba_one(m.m.model, {'tenure': c.tenure, 'spend': c.spend}) "
                    "FROM customers c JOIN models m USING (region)"
                ),
                description="Per-class probabilities from each customer's group classifier",
            )
        ]

    @classmethod
    def compute(
        cls,
        model: Annotated[pa.BinaryArray, Param(doc="A fitted classifier model")],
        features: Annotated[pa.Array, Param(doc="Feature values, one field per feature")],
    ) -> Annotated[pa.ListArray, Returns(pa.list_(pa.float64()))]:
        """Return per-class probabilities for each row from its classifier BLOB."""
        vals = _predict_values(model, features, proba=True)
        return pa.array(
            [None if v is None else [float(p) for p in v] for v in vals],
            type=pa.list_(pa.float64()),
        )


GROUPED_FUNCTIONS: list[type] = [FitModel, PredictOne, PredictClassOne, PredictProbaOne]
