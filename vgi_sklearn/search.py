"""Hyperparameter search exposed as a single discriminated-union SQL function.

``sklearn.grid_search`` runs scikit-learn's ``GridSearchCV`` over a table and
returns the cross-validation leaderboard (one row per parameter combination)
plus the refit best model as a BLOB. The estimator and its search grid are a
single **tagged-union** argument: the union *tag* is the estimator name and the
*value* is a struct of hyperparameter value-lists. Each member therefore only
exposes the hyperparameters that estimator actually has — a discriminated union,
realized with DuckDB's ``UNION`` type:

    SELECT params, mean_test_score, rank
    FROM sklearn.grid_search(
      (SELECT tenure, monthly_spend, support_tickets, churned FROM churn),
      target := 'churned',
      estimator := union_value(random_forest_classifier := {
        'n_estimators': [100, 300], 'max_depth': [3, 5, 8]}))
    ORDER BY rank;

Only the hyperparameters you list are searched; the rest stay at the estimator's
defaults. Requires a vgi-python with union-tag-preserving argument decoding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import pyarrow as pa
import sklearn
from sklearn.model_selection import GridSearchCV
from vgi import TaggedUnion
from vgi.arguments import Arg, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_buffering_function import OutputCollector, TableBufferingParams
from vgi.table_function import BindParams

from .buffering import DrainState, SinkBuffer, input_schema_of
from .features import PIPELINE_PARAM_PREFIX, prefix_grid, wrap_estimator
from .models import _ESTIMATORS, CLASSIFICATION, _features_excluding, _xy
from .registry import ModelMetadata, get_store, now_iso, pack_model, validate_name
from .schema_utils import field as sfield
from .typed_models import _HPARAMS, _UNSET

_PYTYPE_TO_ARROW: dict[type, pa.DataType] = {
    int: pa.int64(),
    float: pa.float64(),
    str: pa.string(),
    bool: pa.bool_(),
}


def _member_struct(spec: list) -> pa.DataType:
    """Struct type for one estimator's grid: each hyperparameter as a list of its scalar type."""
    return pa.struct([pa.field(hp.name, pa.list_(_PYTYPE_TO_ARROW[hp.type])) for hp in spec])


# One sparse-union member per estimator, tagged by the estimator name. This is
# the discriminated union surfaced to SQL via union_value(<estimator> := {...}).
_GRID_UNION = pa.sparse_union([pa.field(name, _member_struct(spec)) for name, spec in _HPARAMS.items()])


def _param_grid(tag: str, value: dict[str, Any] | None) -> dict[str, list[Any]]:
    """Translate a union member value (``{param: [values]}``) into a scikit-learn param grid.

    Applies the same per-hyperparameter translations as the typed ``fit_<estimator>``
    functions, element-wise (e.g. ``max_depth`` 0 -> ``None``; mlp ``hidden_units``
    -> ``hidden_layer_sizes`` tuples). Hyperparameters left unset (NULL) are
    omitted, so they stay at the estimator default rather than being searched.
    """
    grid: dict[str, list[Any]] = {}
    for hp in _HPARAMS[tag]:
        vals = (value or {}).get(hp.name)
        if vals is None:
            continue
        items = list(vals)
        if hp.none_if is not _UNSET:
            items = [None if v == hp.none_if else v for v in items]
        if hp.wrap_tuple:
            items = [(int(v),) for v in items]
        grid[hp.kwarg or hp.name] = items
    return grid


@dataclass(slots=True, frozen=True)
class GridSearchArgs:
    data: Annotated[TableInput, Arg(0, doc="Training table (features + target [+ id]).")]
    estimator: Annotated[
        TaggedUnion,
        Arg(
            "estimator",
            arrow_type=_GRID_UNION,
            doc="union_value(<estimator> := {param: [values], ...}); the tag picks the estimator.",
        ),
    ]
    target: Annotated[str, Arg("target", default="", doc="Name of the target/label column (required).")]
    id: Annotated[str, Arg("id", default="", doc="Optional id column to exclude from features.")]
    cv: Annotated[int, Arg("cv", default=5, doc="Number of cross-validation folds.")]
    scoring: Annotated[str, Arg("scoring", default="", doc="Scorer name (default: the estimator's own scorer).")]
    model_name: Annotated[str, Arg("model_name", default="", doc="Optional registry name for the refit best model.")]


_SEARCH_SCHEMA = pa.schema(
    [
        sfield("estimator", pa.string(), "Estimator that was searched.", nullable=False),
        sfield("params", pa.string(), "This combination's hyperparameters (JSON).", nullable=False),
        sfield("mean_test_score", pa.float64(), "Mean cross-validated score.", nullable=False),
        sfield("std_test_score", pa.float64(), "Std-dev of the cross-validated score.", nullable=False),
        sfield("rank", pa.int64(), "Rank by mean score (1 = best).", nullable=False),
        sfield("model", pa.binary(), "The refit best model as a BLOB (only on the rank-1 row)."),
    ]
)


class GridSearch(SinkBuffer[GridSearchArgs, DrainState]):
    FunctionArguments: ClassVar[type] = GridSearchArgs

    class Meta:
        name = "grid_search"
        description = "Cross-validated grid search over an estimator's hyperparameters"
        categories = ["models", "supervised", "tuning"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT params, mean_test_score, rank FROM sklearn.grid_search("
                    "(SELECT * FROM sklearn.iris()), target := 'target', id := 'sample_id', "
                    "estimator := union_value(random_forest_classifier := "
                    "{'n_estimators': [100, 300], 'max_depth': [3, 5]})) ORDER BY rank"
                ),
                description="Grid-search a random forest on iris",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[GridSearchArgs]) -> BindResponse:
        a = params.args
        if not a.target:
            raise ValueError("grid_search requires 'target' (the label column name, e.g. target := 'label')")
        tag = getattr(a.estimator, "tag", None)
        if tag is not None and tag not in _ESTIMATORS:
            raise ValueError(f"unknown estimator {tag!r}; choose one of: {', '.join(sorted(_ESTIMATORS))}")
        if a.model_name:
            validate_name(a.model_name)
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        if a.target not in input_schema.names:
            raise ValueError(f"target column {a.target!r} not found in input; columns: {', '.join(input_schema.names)}")
        return BindResponse(output_schema=_SEARCH_SCHEMA)

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[GridSearchArgs]
    ) -> DrainState:
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[GridSearchArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        if state.done:
            out.finish()
            return
        state.done = True

        a = params.args
        tag = a.estimator.tag
        if tag not in _ESTIMATORS:
            raise ValueError(f"unknown estimator {tag!r}")
        task, est_cls, defaults = _ESTIMATORS[tag]
        grid = _param_grid(tag, a.estimator.value)

        input_schema = input_schema_of(params)
        feats = _features_excluding(input_schema, a.target, a.id)
        table = cls.buffered_table(params, input_schema)
        if table is None or table.num_rows == 0:
            raise ValueError("grid_search received no training rows")

        x, y, cat_mask = _xy(table, feats, a.target, task)
        # When features are categorical the estimator becomes a one-hot Pipeline,
        # so grid keys must address the estimator step (est__<param>).
        wrapped = any(cat_mask)
        estimator = wrap_estimator(est_cls(**defaults), cat_mask)
        search = GridSearchCV(estimator, prefix_grid(grid, wrapped), cv=a.cv, scoring=(a.scoring or None), refit=True)
        search.fit(x, y)

        results = search.cv_results_
        n = len(results["params"])
        best_idx = int(search.best_index_)

        classes = [int(c) for c in search.best_estimator_.classes_] if task == CLASSIFICATION else None
        meta = ModelMetadata(
            name=a.model_name,
            estimator=tag,
            task=task,
            target=a.target,
            feature_names=feats,
            categorical=cat_mask,
            params={k: _json_safe(v) for k, v in _strip_prefix(search.best_params_).items()},
            classes=classes,
            n_samples=int(table.num_rows),
            n_features=len(feats),
            train_score=float(search.best_score_),
            sklearn_version=sklearn.__version__,
            created_at=now_iso(),
        )
        if a.model_name:
            get_store().save(search.best_estimator_, meta)
        best_blob = pack_model(search.best_estimator_, meta)

        out.emit(
            pa.RecordBatch.from_pydict(
                {
                    "estimator": [tag] * n,
                    "params": [
                        json.dumps({k: _json_safe(v) for k, v in _strip_prefix(p).items()}) for p in results["params"]
                    ],
                    "mean_test_score": [float(s) for s in results["mean_test_score"]],
                    "std_test_score": [float(s) for s in results["std_test_score"]],
                    "rank": [int(r) for r in results["rank_test_score"]],
                    "model": [best_blob if i == best_idx else None for i in range(n)],
                },
                schema=params.output_schema,
            )
        )


def _strip_prefix(d: dict[str, Any]) -> dict[str, Any]:
    """Drop the pipeline ``est__`` prefix from param names for display."""
    n = len(PIPELINE_PARAM_PREFIX)
    return {(k[n:] if k.startswith(PIPELINE_PARAM_PREFIX) else k): v for k, v in d.items()}


def _json_safe(v: Any) -> Any:
    """Make a hyperparameter value JSON-serializable (tuples -> lists)."""
    if isinstance(v, tuple):
        return list(v)
    return v


SEARCH_FUNCTIONS: list[type] = [GridSearch]
