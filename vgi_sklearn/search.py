"""Hyperparameter search exposed as a single discriminated-union SQL function.

``sklearn.models.grid_search`` runs scikit-learn's ``GridSearchCV`` over a table and
returns the cross-validation leaderboard (one row per parameter combination)
plus the refit best model as a BLOB. The estimator and its search grid are a
single **tagged-union** argument: the union *tag* is the estimator name and the
*value* is a struct of hyperparameter value-lists. Each member therefore only
exposes the hyperparameters that estimator actually has — a discriminated union,
realized with DuckDB's ``UNION`` type:

    SELECT params, mean_test_score, rank
    FROM sklearn.models.grid_search(
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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from vgi import TaggedUnion
from vgi.arguments import Arg, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_buffering_function import TableBufferingParams
from vgi.table_function import BindParams
from vgi_rpc.rpc import OutputCollector

from .buffering import DrainState, SinkBuffer, input_schema_of
from .features import PIPELINE_PARAM_PREFIX, prefix_grid, wrap_estimator
from .models import _ESTIMATORS, CLASSIFICATION, _features_excluding, _xy
from .registry import ModelMetadata, get_store, now_iso, pack_model, validate_name
from .schema_utils import columns_md
from .schema_utils import field as sfield
from .typed_models import _HP, _HPARAMS, _UNSET

_PYTYPE_TO_ARROW: dict[type, pa.DataType] = {
    int: pa.int64(),
    float: pa.float64(),
    str: pa.string(),
    bool: pa.bool_(),
}


def _member_struct(spec: list[_HP]) -> pa.DataType:
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
    """Arguments for the grid_search function."""

    data: Annotated[TableInput, Arg(0, doc="Training rows: features + target [+ id].")]
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
    """Cross-validated grid search over an estimator's hyperparameters."""

    FunctionArguments: ClassVar[type] = GridSearchArgs

    class Meta:
        """VGI metadata for the grid_search function."""

        name = "grid_search"
        description = "Cross-validated grid search over an estimator's hyperparameters"
        categories = ["models", "supervised", "tuning"]
        tags = {
            "vgi.result_columns_md": columns_md(_SEARCH_SCHEMA),
            "vgi.doc_llm": (
                "Table function that runs scikit-learn `GridSearchCV` — an exhaustive cross-validated "
                "sweep of every combination in a hyperparameter grid — over a buffered training relation "
                "`(SELECT ...)` (Arg(0)) and returns the CV leaderboard, one row per combination. The "
                "estimator and grid are a single discriminated-union argument: call "
                "`estimator := union_value(<estimator> := {param: [v1, v2, ...], ...})`, where the union "
                "*tag* picks the algorithm (any name from `fit_<estimator>`) and each field is the list of "
                "values to try; hyperparameters you omit stay at their defaults rather than being searched. "
                "Name the `target :=` label column (required), optionally an `id :=` to exclude, set `cv :=` "
                "folds and an optional `scoring :=`. String features auto one-hot-encode. Output columns: "
                "`estimator`, `params` (JSON of the combo), `mean_test_score`, `std_test_score`, `rank` "
                "(1 = best), and `model` — the refit best estimator as a self-contained BLOB carried on the "
                "single `best_index_` row, so grab it with `WHERE model IS NOT NULL` (rank 1 can tie). Pass "
                "`model_name :=` to also persist the best model to the registry. Cost is the full Cartesian "
                "product of the grid times `cv` fits — use `randomized_search` to cap it."
            ),
            "vgi.doc_md": (
                "**grid_search** — exhaustive cross-validated hyperparameter search; returns a leaderboard "
                "plus the refit best model BLOB.\n\n"
                "Runs `GridSearchCV` over every combination in the grid and emits one row per combination, "
                "ranked by mean CV score.\n\n"
                "- Estimator + grid are one tagged-union arg: "
                "`estimator := union_value(<estimator> := {param: [values], ...})` — the tag picks the "
                "algorithm and each field lists the values to try; omitted params stay at their defaults\n"
                "- Input: `(SELECT ...)` training table; `target :=` label (**required**); `id :=` excluded "
                "passthrough; `cv :=` folds; `scoring :=` scorer; `model_name :=` to persist the best model\n"
                "- Output: `estimator`, `params` (JSON), `mean_test_score`, `std_test_score`, `rank` "
                "(1 = best), and `model`\n"
                "- The refit best model BLOB sits only on the best row — select it with "
                "`WHERE model IS NOT NULL` (rank 1 can tie). Cost grows with the full grid size; prefer "
                "`randomized_search` for large spaces"
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    "SELECT params, mean_test_score, rank FROM sklearn.models.grid_search("
                    "(SELECT * FROM sklearn.datasets.iris()), target := 'target', id := 'sample_id', "
                    "estimator := union_value(random_forest_classifier := "
                    "{'n_estimators': [100, 300], 'max_depth': [3, 5]})) ORDER BY rank"
                ),
                description="Grid-search a random forest on iris",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[GridSearchArgs]) -> BindResponse:
        """Validate the estimator/target and declare the leaderboard output schema."""
        return _validate_search_bind(cls.Meta.name, params)

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[GridSearchArgs]
    ) -> DrainState:
        """Start with an unfinished drain state."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[GridSearchArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Run GridSearchCV on the buffered table and emit the CV leaderboard."""
        _run_search(
            cls,
            params,
            state,
            out,
            lambda est, space, a: GridSearchCV(est, space, cv=a.cv, scoring=(a.scoring or None), refit=True),
        )


def _validate_search_bind(name: str, params: BindParams[Any]) -> BindResponse:
    """Shared bind validation for grid_search / randomized_search."""
    a = params.args
    if not a.target:
        raise ValueError(f"{name} requires 'target' (the label column name, e.g. target := 'label')")
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


def _grid_size(space: dict[str, Any]) -> int:
    """Total number of combinations in a (list-valued) parameter grid."""
    total = 1
    for values in space.values():
        total *= max(1, len(values))
    return total


def _run_search(cls: type, params: Any, state: DrainState, out: OutputCollector, build_search: Any) -> None:
    """Shared finalize for the search functions; ``build_search(est, space, args)`` makes the CV object."""
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
    table = cls.buffered_table(params, input_schema)  # type: ignore[attr-defined]  # SinkBuffer subclass
    if table is None or table.num_rows == 0:
        raise ValueError(f"{cls.Meta.name} received no training rows")  # type: ignore[attr-defined]  # VGI function class

    x, y, cat_mask = _xy(table, feats, a.target, task)
    # When features are categorical the estimator becomes a one-hot Pipeline,
    # so grid keys must address the estimator step (est__<param>).
    wrapped = any(cat_mask)
    estimator = wrap_estimator(est_cls(**defaults), cat_mask)
    search = build_search(estimator, prefix_grid(grid, wrapped), a)
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


@dataclass(slots=True, frozen=True)
class RandomizedSearchArgs:
    """Arguments for the randomized_search function."""

    data: Annotated[TableInput, Arg(0, doc="Training rows: features + target [+ id].")]
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
    n_iter: Annotated[int, Arg("n_iter", default=10, doc="Number of random combinations to sample.")]
    cv: Annotated[int, Arg("cv", default=5, doc="Number of cross-validation folds.")]
    scoring: Annotated[str, Arg("scoring", default="", doc="Scorer name (default: the estimator's own scorer).")]
    random_state: Annotated[int, Arg("random_state", default=0, doc="Random seed for the sampler.")]
    model_name: Annotated[str, Arg("model_name", default="", doc="Optional registry name for the refit best model.")]


class RandomizedSearch(SinkBuffer[RandomizedSearchArgs, DrainState]):
    """Cross-validated randomized search over an estimator's hyperparameters."""

    FunctionArguments: ClassVar[type] = RandomizedSearchArgs

    class Meta:
        """VGI metadata for the randomized_search function."""

        name = "randomized_search"
        description = "Cross-validated randomized search: sample n_iter hyperparameter combinations"
        categories = ["models", "supervised", "tuning"]
        tags = {
            "vgi.result_columns_md": columns_md(_SEARCH_SCHEMA),
            "vgi.doc_llm": (
                "Table function that runs scikit-learn `RandomizedSearchCV` — it samples `n_iter :=` "
                "random combinations from a hyperparameter grid (instead of the full Cartesian product) "
                "and cross-validates each — over a buffered training relation `(SELECT ...)` (Arg(0)), "
                "returning the CV leaderboard one row per sampled combination. The estimator and grid are a "
                "single discriminated-union argument: "
                "`estimator := union_value(<estimator> := {param: [v1, v2, ...], ...})`, where the union "
                "*tag* picks the algorithm (any name from `fit_<estimator>`) and each field lists candidate "
                "values; omitted hyperparameters stay at their defaults. Use `n_iter :=` to cap the budget "
                "(it is clamped to the grid size) and `random_state :=` to make the sampling reproducible. "
                "Name the `target :=` label column (required), optionally an `id :=` to exclude, set `cv :=` "
                "folds and an optional `scoring :=`. String features auto one-hot-encode. Output columns: "
                "`estimator`, `params` (JSON), `mean_test_score`, `std_test_score`, `rank` (1 = best), and "
                "`model` — the refit best estimator as a self-contained BLOB on the single best row, found "
                "with `WHERE model IS NOT NULL` (rank 1 can tie). Pass `model_name :=` to also persist it. "
                "Prefer this over `grid_search` when the grid is large and an exhaustive sweep is too costly."
            ),
            "vgi.doc_md": (
                "**randomized_search** — sampled cross-validated hyperparameter search; returns a "
                "leaderboard plus the refit best model BLOB.\n\n"
                "Runs `RandomizedSearchCV`, drawing `n_iter` random combinations from the grid (capped at "
                "the grid size) and ranking them by mean CV score — cheaper than an exhaustive "
                "`grid_search` on large spaces.\n\n"
                "- Estimator + grid are one tagged-union arg: "
                "`estimator := union_value(<estimator> := {param: [values], ...})` — the tag picks the "
                "algorithm and each field lists candidate values; omitted params stay at their defaults\n"
                "- Input: `(SELECT ...)` training table; `target :=` label (**required**); `id :=` excluded "
                "passthrough; `n_iter :=` combinations to sample; `random_state :=` sampler seed; `cv :=` "
                "folds; `scoring :=` scorer; `model_name :=` to persist the best model\n"
                "- Output: `estimator`, `params` (JSON), `mean_test_score`, `std_test_score`, `rank` "
                "(1 = best), and `model`\n"
                "- The refit best model BLOB sits only on the best row — select it with "
                "`WHERE model IS NOT NULL` (rank 1 can tie)"
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    "SELECT params, mean_test_score, rank FROM sklearn.models.randomized_search("
                    "(SELECT * FROM sklearn.datasets.iris()), target := 'target', id := 'sample_id', n_iter := 4, "
                    "estimator := union_value(random_forest_classifier := "
                    "{'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 8]})) ORDER BY rank"
                ),
                description="Randomized-search a random forest on iris",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[RandomizedSearchArgs]) -> BindResponse:
        """Validate the estimator/target and declare the leaderboard output schema."""
        return _validate_search_bind(cls.Meta.name, params)

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[RandomizedSearchArgs]
    ) -> DrainState:
        """Start with an unfinished drain state."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[RandomizedSearchArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Run RandomizedSearchCV on the buffered table and emit the CV leaderboard."""
        # n_iter can't exceed the number of distinct combinations (the grid is discrete).
        _run_search(
            cls,
            params,
            state,
            out,
            lambda est, space, a: RandomizedSearchCV(
                est,
                space,
                n_iter=min(a.n_iter, _grid_size(space)),
                cv=a.cv,
                scoring=(a.scoring or None),
                random_state=a.random_state,
                refit=True,
            ),
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


SEARCH_FUNCTIONS: list[type] = [GridSearch, RandomizedSearch]
