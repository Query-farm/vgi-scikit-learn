"""``fit_pipeline``: chain preprocessing steps + an estimator into one model.

Composes a sequence of transformer steps (the same ``kind`` registry as
``fit_transformer``) followed by a supervised estimator, fits the whole thing at
once, and stores it as an ordinary model BLOB. Because the stored model *is* the
fitted ``Pipeline``, everything that consumes a model — ``predict``,
``cross_val_predict``, ``permutation_importance``, the registry — works on it
unchanged; there is no separate ``apply_pipeline``.

    SELECT * FROM sklearn.models.fit_pipeline((SELECT * FROM train),
      steps := '[{"kind": "simple_imputer", "params": {"strategy": "median"}},
                 {"kind": "standard_scaler"}, {"kind": "pca", "params": {"n_components": 3}}]',
      estimator := 'logistic_regression', target := 'label', model_name := 'clf');

    SELECT * FROM sklearn.models.predict((SELECT * FROM test), model_name := 'clf', id := 'id');

String features are one-hot-encoded ahead of the steps (same auto-encoding as
``fit``), so a pipeline of ``scaler -> pca -> model`` also works on mixed data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

from sklearn.pipeline import Pipeline
from vgi.arguments import Arg, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_buffering_function import TableBufferingParams
from vgi.table_function import BindParams
from vgi_rpc.rpc import OutputCollector

from .buffering import DrainState, SinkBuffer, input_schema_of
from .models import _ESTIMATORS, _FIT_SCHEMA, _fit_and_emit, _parse_params, build_estimator
from .registry import validate_name
from .schema_utils import columns_md
from .stored_transforms import TRANSFORMER_KINDS, _build


def _parse_steps(steps: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse the JSON ``steps`` spec into ``[(kind, params), ...]``."""
    steps = (steps or "").strip()
    if not steps:
        return []
    parsed = json.loads(steps)
    if not isinstance(parsed, list):
        raise ValueError('steps must be a JSON array, e.g. \'[{"kind": "standard_scaler"}]\'')
    out: list[tuple[str, dict[str, Any]]] = []
    for i, step in enumerate(parsed):
        if not isinstance(step, dict) or "kind" not in step:
            raise ValueError(f'step {i} must be an object with a \'kind\', e.g. {{"kind": "pca"}}')
        kind = step["kind"]
        if kind not in TRANSFORMER_KINDS:
            raise ValueError(f"unknown step kind {kind!r}; choose one of: {', '.join(TRANSFORMER_KINDS)}")
        params = step.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"step {i} ({kind!r}) params must be a JSON object")
        out.append((kind, params))
    return out


@dataclass(slots=True, frozen=True)
class FitPipelineArgs:
    """Arguments for the fit_pipeline function."""

    data: Annotated[TableInput, Arg(0, doc="Training table (features + target [+ id]).")]
    steps: Annotated[
        str,
        Arg("steps", default="", doc='JSON array of preprocessing steps, e.g. [{"kind":"standard_scaler"}].'),
    ]
    estimator: Annotated[str, Arg("estimator", default="random_forest_classifier", doc="Final estimator name.")]
    target: Annotated[str, Arg("target", default="", doc="Name of the target/label column (required).")]
    id: Annotated[str, Arg("id", default="", doc="Optional id column to exclude from features.")]
    params: Annotated[str, Arg("params", default="", doc="JSON object of estimator hyperparameters.")]
    model_name: Annotated[str, Arg("model_name", default="", doc="Name to store the fitted pipeline under (optional).")]


class FitPipeline(SinkBuffer[FitPipelineArgs, DrainState]):
    """Fit preprocessing steps plus an estimator as one stored model."""

    FunctionArguments: ClassVar[type] = FitPipelineArgs

    class Meta:
        """VGI metadata for the fit_pipeline function."""

        name = "fit_pipeline"
        description = "Fit preprocessing steps + an estimator as one model; predict with the usual 'predict'"
        categories = ["models", "supervised", "pipeline"]
        tags = {
            "vgi.result_columns_md": columns_md(_FIT_SCHEMA),
            "vgi.doc_llm": (
                "Table function that chains preprocessing transformers and a final estimator into one "
                "scikit-learn `Pipeline`, fits the whole thing on the buffered training relation "
                "`(SELECT ...)` (Arg(0)), and stores it as an ordinary model BLOB. Pass `steps :=` a JSON "
                'array of `{"kind": ..., "params": {...}}` transformers (same kinds as `fit_transformer`, '
                "e.g. simple_imputer -> standard_scaler -> pca), the final `estimator :=` name, the required "
                "`target :=` label column, an optional `id :=` to exclude, estimator `params :=` JSON, and "
                "`model_name :=` to persist. String features are one-hot-encoded ahead of the steps. It "
                "returns the same one-row fit summary + `model` BLOB as `fit`, and the stored artifact is a "
                "normal model — score it with the usual `predict` (no `apply_pipeline` exists)."
            ),
            "vgi.doc_md": (
                "**fit_pipeline** — fit preprocessing steps + an estimator as a single stored model.\n\n"
                "Builds a `Pipeline([steps..., estimator])`, fits it on the buffered training table, and "
                "saves it as a normal model BLOB; everything that consumes a model (`predict`, "
                "`cross_val_predict`, `permutation_importance`) then works unchanged.\n\n"
                "- Input: `(SELECT ...)` training table (features + target [+ id])\n"
                "- `steps :=` JSON array of `{kind, params}` transformers; `estimator :=` final estimator; "
                "`target :=` label column (**required**); `id :=` excluded passthrough; `params :=` estimator "
                "hyperparams JSON; `model_name :=` to persist to the registry\n"
                "- Output: the `fit` summary row (`model_name`, `estimator`, `task`, `n_features`, "
                "`train_score`, ...) plus the `model` BLOB\n"
                "- String columns one-hot-encode before the steps; predict with the standard `predict` — "
                "there is intentionally no `apply_pipeline`"
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    "SELECT model_name, estimator, n_features FROM sklearn.models.fit_pipeline("
                    "(SELECT * FROM sklearn.datasets.iris()), "
                    'steps := \'[{"kind": "standard_scaler"}, {"kind": "pca", "params": {"n_components": 3}}]\', '
                    "estimator := 'logistic_regression', target := 'target', id := 'sample_id', "
                    "model_name := 'iris_pipe')"
                ),
                description="Scale -> PCA -> logistic regression, stored as one model",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[FitPipelineArgs]) -> BindResponse:
        """Validate the estimator, steps, and target and declare the fit summary schema."""
        a = params.args
        if not a.target:
            raise ValueError("fit_pipeline requires 'target' (the label column name, e.g. target := 'label')")
        if a.estimator not in _ESTIMATORS:
            raise ValueError(f"unknown estimator {a.estimator!r}; choose one of: {', '.join(sorted(_ESTIMATORS))}")
        _parse_steps(a.steps)  # validate the step spec early
        if a.model_name:
            validate_name(a.model_name)
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        if a.target not in input_schema.names:
            raise ValueError(f"target column {a.target!r} not found in input; columns: {', '.join(input_schema.names)}")
        return BindResponse(output_schema=_FIT_SCHEMA)

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[FitPipelineArgs]
    ) -> DrainState:
        """Start with an unfinished drain state."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[FitPipelineArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Assemble the pipeline, fit it on the buffered table, and emit the model."""
        if state.done:
            out.finish()
            return
        state.done = True

        a = params.args
        input_schema = input_schema_of(params)
        steps = _parse_steps(a.steps)
        est_params = _parse_params(a.params)
        task, estimator = build_estimator(a.estimator, est_params)

        pipeline = Pipeline([(f"step{i}", _build(kind, p)) for i, (kind, p) in enumerate(steps)] + [("est", estimator)])

        table = cls.buffered_table(params, input_schema)
        _fit_and_emit(
            out,
            params.output_schema,
            table=table,
            input_schema=input_schema,
            estimator_label=a.estimator,
            task=task,
            estimator=pipeline,
            model_name=a.model_name,
            target=a.target,
            id_col=a.id,
            params_dict={"steps": [{"kind": k, "params": p} for k, p in steps], "estimator": est_params},
        )


PIPELINE_FUNCTIONS: list[type] = [FitPipeline]
