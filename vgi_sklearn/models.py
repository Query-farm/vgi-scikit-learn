"""Supervised learning: fit estimators into the registry and predict from it.

* ``fit``       -- TableBufferingFunction: buffer the training table, fit an
  estimator, persist it to the registry, return a one-row training summary.
* ``predict``   -- TableInOutGenerator: stream a table through a stored model.
* ``cross_val_predict`` -- buffering: out-of-fold predictions, no persistence.
* ``list_models`` / ``model_info`` / ``drop_model`` -- registry management.

Column roles follow the project convention: name the ``target`` column (for
fit / cross_val_predict) and optionally an ``id`` column to carry through; every
other column is a numeric feature. Hyperparameters are passed as a JSON string.

    SELECT * FROM sklearn.fit((SELECT * FROM training), model_name => 'iris_rf',
                              estimator => 'random_forest_classifier', target => 'species', id => 'id');
    SELECT * FROM sklearn.predict((SELECT * FROM new_data), model_name => 'iris_rf', id => 'id');
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import numpy as np
import pyarrow as pa
import sklearn
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import cross_val_predict as sk_cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from vgi.arguments import Arg, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_buffering_function import OutputCollector, TableBufferingParams
from vgi.table_function import (
    BindParams,
    ProcessParams,
    TableCardinality,
    TableFunctionGenerator,
    bind_fixed_schema,
    init_single_worker,
)
from vgi.table_in_out_function import OutputCollector as InOutCollector
from vgi.table_in_out_function import TableInOutGenerator

from .buffering import DrainState, SinkBuffer, input_schema_of, matrix
from .registry import (
    ModelMetadata,
    ModelNotFoundError,
    get_store,
    now_iso,
    pack_model,
    unpack_meta,
    unpack_model,
    validate_name,
)
from .schema_utils import field as sfield

CLASSIFICATION = "classification"
REGRESSION = "regression"

# name -> (task, estimator class, default kwargs)
_ESTIMATORS: dict[str, tuple[str, type, dict[str, Any]]] = {
    "logistic_regression": (CLASSIFICATION, LogisticRegression, {"max_iter": 1000}),
    "random_forest_classifier": (CLASSIFICATION, RandomForestClassifier, {"random_state": 0}),
    "random_forest_regressor": (REGRESSION, RandomForestRegressor, {"random_state": 0}),
    "gradient_boosting_classifier": (CLASSIFICATION, GradientBoostingClassifier, {"random_state": 0}),
    "gradient_boosting_regressor": (REGRESSION, GradientBoostingRegressor, {"random_state": 0}),
    "hist_gradient_boosting_classifier": (CLASSIFICATION, HistGradientBoostingClassifier, {"random_state": 0}),
    "hist_gradient_boosting_regressor": (REGRESSION, HistGradientBoostingRegressor, {"random_state": 0}),
    "linear_regression": (REGRESSION, LinearRegression, {}),
    "ridge": (REGRESSION, Ridge, {}),
    "lasso": (REGRESSION, Lasso, {}),
    "svc": (CLASSIFICATION, SVC, {}),
    "svr": (REGRESSION, SVR, {}),
    "knn_classifier": (CLASSIFICATION, KNeighborsClassifier, {}),
    "knn_regressor": (REGRESSION, KNeighborsRegressor, {}),
    "decision_tree_classifier": (CLASSIFICATION, DecisionTreeClassifier, {"random_state": 0}),
    "decision_tree_regressor": (REGRESSION, DecisionTreeRegressor, {"random_state": 0}),
    "mlp_classifier": (CLASSIFICATION, MLPClassifier, {"max_iter": 500, "random_state": 0}),
    "mlp_regressor": (REGRESSION, MLPRegressor, {"max_iter": 500, "random_state": 0}),
    "gaussian_nb": (CLASSIFICATION, GaussianNB, {}),
}


def _parse_params(params: str) -> dict[str, Any]:
    params = (params or "").strip()
    if not params:
        return {}
    parsed = json.loads(params)
    if not isinstance(parsed, dict):
        raise ValueError("params must be a JSON object, e.g. '{\"n_estimators\": 200}'")
    return parsed


def estimator_param_names(name: str) -> list[str]:
    """Sorted list of hyperparameters accepted by an estimator (for discovery/errors)."""
    _task, cls, _defaults = _ESTIMATORS[name]
    return sorted(cls().get_params().keys())


def build_estimator(name: str, params: dict[str, Any]) -> tuple[str, Any]:
    """Return ``(task, estimator)`` for a registered estimator name + hyperparams."""
    if name not in _ESTIMATORS:
        raise ValueError(f"unknown estimator {name!r}; choose one of: {', '.join(sorted(_ESTIMATORS))}")
    task, cls, defaults = _ESTIMATORS[name]
    # Reject unknown hyperparameters up front with the valid set, rather than
    # surfacing sklearn's opaque "unexpected keyword argument" TypeError.
    valid = set(cls().get_params().keys())
    unknown = [k for k in params if k not in valid]
    if unknown:
        raise ValueError(
            f"unknown hyperparameter(s) for {name!r}: {', '.join(sorted(unknown))}. "
            f"valid params: {', '.join(sorted(valid))}"
        )
    try:
        return task, cls(**{**defaults, **params})
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid hyperparameters for {name!r}: {exc}") from exc


def _xy(table: pa.Table, feature_names: list[str], target: str, task: str) -> tuple[np.ndarray, np.ndarray]:
    x = matrix(table, feature_names)
    y = np.asarray(table.column(target).to_numpy(zero_copy_only=False))
    if task == CLASSIFICATION:
        y = np.rint(y.astype(float)).astype(int)
    else:
        y = y.astype(float)
    return x, y


def _features_excluding(input_schema: pa.Schema, *exclude: str) -> list[str]:
    drop = {e for e in exclude if e}
    return [n for n in input_schema.names if n not in drop]


def _prediction_field(task: str) -> pa.Field:
    if task == CLASSIFICATION:
        return sfield("prediction", pa.int64(), "Predicted class label.", nullable=False)
    return sfield("prediction", pa.float64(), "Predicted value.", nullable=False)


# ===========================================================================
# fit
# ===========================================================================


# Required string args carry a "" default so an omitted value reaches on_bind as
# "" and we can raise a friendly error, instead of the framework's raw KeyError
# during argument parsing.
@dataclass(slots=True, frozen=True)
class FitArgs:
    data: Annotated[TableInput, Arg(0, doc="Training table (features + target [+ id]).")]
    model_name: Annotated[str, Arg("model_name", default="", doc="Name to store the fitted model under (required).")]
    estimator: Annotated[str, Arg("estimator", default="random_forest_classifier", doc="Estimator name.")]
    target: Annotated[str, Arg("target", default="", doc="Name of the target/label column (required).")]
    id: Annotated[str, Arg("id", default="", doc="Optional id column to exclude from features.")]
    params: Annotated[str, Arg("params", default="", doc="JSON object of hyperparameters.")]


_FIT_SCHEMA = pa.schema(
    [
        sfield("model_name", pa.string(), "Name the model was stored under ('' if not persisted).", nullable=False),
        sfield("estimator", pa.string(), "Estimator type used.", nullable=False),
        sfield("task", pa.string(), "classification or regression.", nullable=False),
        sfield("n_samples", pa.int64(), "Number of training rows.", nullable=False),
        sfield("n_features", pa.int64(), "Number of features.", nullable=False),
        sfield("n_classes", pa.int64(), "Number of classes (NULL for regression)."),
        sfield("train_score", pa.float64(), "In-sample score (accuracy or R^2)."),
        sfield("features", pa.list_(pa.string()), "Ordered feature column names.", nullable=False),
        sfield("model", pa.binary(), "The fitted model as a self-contained BLOB (estimator + metadata).", nullable=False),
    ]
)


def _fit_and_emit(
    out: OutputCollector,
    output_schema: pa.Schema,
    *,
    table: pa.Table | None,
    input_schema: pa.Schema,
    estimator_label: str,
    task: str,
    estimator: Any,
    model_name: str,
    target: str,
    id_col: str,
    params_dict: dict[str, Any],
) -> None:
    """Fit ``estimator`` on the buffered table, persist if named, emit summary + BLOB.

    Shared by the generic ``fit`` and the typed ``fit_<estimator>`` functions.
    """
    if table is None or table.num_rows == 0:
        raise ValueError("fit received no training rows")
    feats = _features_excluding(input_schema, target, id_col)
    x, y = _xy(table, feats, target, task)
    estimator.fit(x, y)
    train_score = float(estimator.score(x, y))
    classes = [int(c) for c in estimator.classes_] if task == CLASSIFICATION else None

    meta = ModelMetadata(
        name=model_name,
        estimator=estimator_label,
        task=task,
        target=target,
        feature_names=feats,
        params=params_dict,
        classes=classes,
        n_samples=int(table.num_rows),
        n_features=len(feats),
        train_score=train_score,
        sklearn_version=sklearn.__version__,
        created_at=now_iso(),
    )
    if model_name:
        get_store().save(estimator, meta)

    out.emit(
        pa.RecordBatch.from_pydict(
            {
                "model_name": [model_name],
                "estimator": [estimator_label],
                "task": [task],
                "n_samples": [meta.n_samples],
                "n_features": [meta.n_features],
                "n_classes": [len(classes) if classes is not None else None],
                "train_score": [train_score],
                "features": [feats],
                "model": [pack_model(estimator, meta)],
            },
            schema=output_schema,
        )
    )


class FitModel(SinkBuffer[FitArgs, DrainState]):
    FunctionArguments: ClassVar[type] = FitArgs

    class Meta:
        name = "fit"
        description = "Fit a supervised estimator and store it in the model registry"
        categories = ["models", "supervised"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM sklearn.fit("
                    "(SELECT sample_id, sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm, target "
                    "FROM sklearn.iris()), model_name => 'iris_rf', "
                    "estimator => 'random_forest_classifier', target => 'target', id => 'sample_id')"
                ),
                description="Train a random forest on iris and store it as 'iris_rf'",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[FitArgs]) -> BindResponse:
        a = params.args
        # model_name is optional: the model is always returned as a BLOB; when a
        # name is given it is also persisted to the registry.
        if a.model_name:
            validate_name(a.model_name)
        if not a.target:
            raise ValueError("fit requires 'target' (the label column name, e.g. target := 'label')")
        # Validate estimator + hyperparameters now so errors surface at bind.
        build_estimator(a.estimator, _parse_params(a.params))
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        if a.target not in input_schema.names:
            raise ValueError(
                f"target column {a.target!r} not found in input; columns: {', '.join(input_schema.names)}"
            )
        return BindResponse(output_schema=_FIT_SCHEMA)

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[FitArgs]) -> DrainState:
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[FitArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        if state.done:
            out.finish()
            return
        state.done = True
        a = params.args
        task, estimator = build_estimator(a.estimator, _parse_params(a.params))
        table = cls.buffered_table(params, input_schema_of(params))
        _fit_and_emit(
            out,
            params.output_schema,
            table=table,
            input_schema=input_schema_of(params),
            estimator_label=a.estimator,
            task=task,
            estimator=estimator,
            model_name=a.model_name,
            target=a.target,
            id_col=a.id,
            params_dict=_parse_params(a.params),
        )


# ===========================================================================
# predict
# ===========================================================================


@dataclass(slots=True, frozen=True)
class PredictArgs:
    data: Annotated[TableInput, Arg(0, doc="Table to score (must contain the model's feature columns).")]
    model_name: Annotated[str, Arg("model_name", default="", doc="Name of a model in the registry. Provide this OR model.")]
    model: Annotated[bytes, Arg("model", default=b"", doc="A model BLOB (as returned by fit). Provide this OR model_name.")]
    id: Annotated[str, Arg("id", default="", doc="Optional id column to carry through.")]
    with_proba: Annotated[bool, Arg("with_proba", default=False, doc="Also emit per-class probabilities (classifiers).")]


# Loaded estimators cached per query execution to avoid reloading each batch.
_PREDICT_CACHE: dict[bytes, tuple[Any, ModelMetadata]] = {}
# Execution ids for which a version-mismatch warning was already emitted.
_VERSION_WARNED: set[bytes] = set()


class PredictModel(TableInOutGenerator[PredictArgs]):
    FunctionArguments: ClassVar[type] = PredictArgs

    class Meta:
        name = "predict"
        description = "Score a table through a stored model, emitting predictions"
        categories = ["models", "supervised", "inference"]
        examples = [
            FunctionExample(
                sql="SELECT * FROM sklearn.predict((SELECT * FROM sklearn.iris()), model_name => 'iris_rf', id => 'sample_id')",
                description="Predict with the stored 'iris_rf' model",
            )
        ]

    @classmethod
    def _proba_classes(cls, meta: ModelMetadata) -> list[Any]:
        return meta.classes or []

    @classmethod
    def on_bind(cls, params: BindParams[PredictArgs]) -> BindResponse:
        a = params.args
        if not a.model_name and not a.model:
            raise ValueError("predict requires either 'model_name' (a registry name) or 'model' (a model BLOB)")
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        if a.model_name:
            try:
                meta = get_store().load_meta(a.model_name)
            except ModelNotFoundError as exc:
                raise ValueError(f"model {a.model_name!r} not found in the registry") from exc
        else:
            meta = unpack_meta(a.model)

        # Fail fast at bind if the input is missing any feature the model needs.
        # (predict selects features by name, so order doesn't matter and extra
        # columns are ignored — only missing ones are an error.)
        missing = [f for f in meta.feature_names if f not in input_schema.names]
        if missing:
            raise ValueError(
                f"model {a.model_name!r} requires feature column(s) {', '.join(missing)} "
                f"not present in the input; model features: {', '.join(meta.feature_names)}; "
                f"input columns: {', '.join(input_schema.names)}"
            )

        fields: list[pa.Field] = []
        if a.id:
            fields.append(input_schema.field(a.id))
        fields.append(_prediction_field(meta.task))
        if a.with_proba and meta.task == CLASSIFICATION:
            for c in cls._proba_classes(meta):
                fields.append(sfield(f"proba_{c}", pa.float64(), f"P(class = {c}).", nullable=False))
        return BindResponse(output_schema=pa.schema(fields))

    @classmethod
    def _model(cls, params: ProcessParams[PredictArgs]) -> tuple[Any, ModelMetadata]:
        key = params.init_response.execution_id
        cached = _PREDICT_CACHE.get(key)
        if cached is None:
            a = params.args
            cached = get_store().load(a.model_name) if a.model_name else unpack_model(a.model)
            _PREDICT_CACHE[key] = cached
        return cached

    @classmethod
    def process(
        cls,
        params: ProcessParams[PredictArgs],
        state: None,
        batch: pa.RecordBatch,
        out: InOutCollector,
    ) -> None:
        a = params.args
        estimator, meta = cls._model(params)

        key = params.init_response.execution_id
        if meta.sklearn_version and meta.sklearn_version != sklearn.__version__ and key not in _VERSION_WARNED:
            _VERSION_WARNED.add(key)
            with contextlib.suppress(Exception):
                out.client_log(
                    "warning",
                    f"model {(a.model_name or '<blob>')!r} was fitted with scikit-learn {meta.sklearn_version}, "
                    f"worker has {sklearn.__version__}; predictions may differ",
                )

        x = matrix(pa.Table.from_batches([batch]), meta.feature_names)

        columns: dict[str, list[Any]] = {}
        if a.id:
            columns[a.id] = batch.column(a.id).to_pylist()

        preds = estimator.predict(x)
        if meta.task == CLASSIFICATION:
            columns["prediction"] = [int(v) for v in preds]
        else:
            columns["prediction"] = [float(v) for v in preds]

        if a.with_proba and meta.task == CLASSIFICATION:
            proba = estimator.predict_proba(x)
            for j, c in enumerate(cls._proba_classes(meta)):
                columns[f"proba_{c}"] = [float(v) for v in proba[:, j]]

        out.emit(pa.RecordBatch.from_pydict(columns, schema=params.output_schema))


# ===========================================================================
# cross_val_predict (no persistence)
# ===========================================================================


@dataclass(slots=True, frozen=True)
class CrossValArgs:
    data: Annotated[TableInput, Arg(0, doc="Training table (features + target [+ id]).")]
    estimator: Annotated[str, Arg("estimator", default="random_forest_classifier", doc="Estimator name.")]
    target: Annotated[str, Arg("target", default="", doc="Name of the target/label column (required).")]
    id: Annotated[str, Arg("id", default="", doc="Optional id column to carry through.")]
    params: Annotated[str, Arg("params", default="", doc="JSON object of hyperparameters.")]
    cv: Annotated[int, Arg("cv", default=5, doc="Number of cross-validation folds.")]


class CrossValPredict(SinkBuffer[CrossValArgs, DrainState]):
    FunctionArguments: ClassVar[type] = CrossValArgs

    class Meta:
        name = "cross_val_predict"
        description = "Out-of-fold cross-validated predictions (no model is stored)"
        categories = ["models", "supervised", "evaluation"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM sklearn.cross_val_predict("
                    "(SELECT sample_id, sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm, target "
                    "FROM sklearn.iris()), estimator => 'logistic_regression', target => 'target', id => 'sample_id')"
                ),
                description="5-fold out-of-fold predictions on iris",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[CrossValArgs]) -> BindResponse:
        a = params.args
        if not a.target:
            raise ValueError("cross_val_predict requires 'target' (the label column name, e.g. target := 'label')")
        task, _ = build_estimator(a.estimator, _parse_params(a.params))
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        if a.target not in input_schema.names:
            raise ValueError(
                f"target column {a.target!r} not found in input; columns: {', '.join(input_schema.names)}"
            )
        fields: list[pa.Field] = []
        if a.id:
            fields.append(input_schema.field(a.id))
        fields.append(_prediction_field(task))
        return BindResponse(output_schema=pa.schema(fields))

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[CrossValArgs]) -> DrainState:
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[CrossValArgs],
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
        feats = _features_excluding(input_schema, a.target, a.id)
        task, estimator = build_estimator(a.estimator, _parse_params(a.params))

        table = cls.buffered_table(params, input_schema)
        if table is None or table.num_rows == 0:
            out.emit(pa.RecordBatch.from_pydict({n: [] for n in params.output_schema.names}, schema=params.output_schema))
            return

        x, y = _xy(table, feats, a.target, task)
        preds = sk_cross_val_predict(estimator, x, y, cv=a.cv)

        columns: dict[str, list[Any]] = {}
        if a.id:
            columns[a.id] = table.column(a.id).to_pylist()
        columns["prediction"] = [int(v) for v in preds] if task == CLASSIFICATION else [float(v) for v in preds]
        out.emit(pa.RecordBatch.from_pydict(columns, schema=params.output_schema))


# ===========================================================================
# Registry management: list_models / model_info / drop_model
# ===========================================================================

_MODEL_INFO_SCHEMA = pa.schema(
    [
        sfield("model_name", pa.string(), "Stored model name.", nullable=False),
        sfield("estimator", pa.string(), "Estimator type.", nullable=False),
        sfield("task", pa.dictionary(pa.int8(), pa.string()), "classification or regression.", nullable=False),
        sfield("target", pa.string(), "Target column the model was trained on.", nullable=False),
        sfield("n_features", pa.int32(), "Number of features.", nullable=False),
        sfield("n_samples", pa.int32(), "Number of training rows.", nullable=False),
        sfield("n_classes", pa.int32(), "Number of classes (NULL for regression)."),
        sfield("train_score", pa.float64(), "In-sample training score."),
        sfield("sklearn_version", pa.string(), "scikit-learn version used to fit."),
        sfield("created_at", pa.string(), "UTC timestamp the model was stored."),
        sfield("features", pa.list_(pa.string()), "Ordered feature column names.", nullable=False),
    ]
)


def _meta_rows(metas: list[ModelMetadata]) -> dict[str, list[Any]]:
    return {
        "model_name": [m.name for m in metas],
        "estimator": [m.estimator for m in metas],
        "task": [m.task for m in metas],
        "target": [m.target for m in metas],
        "n_features": [m.n_features for m in metas],
        "n_samples": [m.n_samples for m in metas],
        "n_classes": [len(m.classes) if m.classes is not None else None for m in metas],
        "train_score": [m.train_score for m in metas],
        "sklearn_version": [m.sklearn_version for m in metas],
        "created_at": [m.created_at for m in metas],
        "features": [m.feature_names for m in metas],
    }


@dataclass(slots=True, frozen=True)
class NoArgs:
    pass


@init_single_worker
@bind_fixed_schema
class ListModels(TableFunctionGenerator[NoArgs]):
    FIXED_SCHEMA: ClassVar[pa.Schema] = _MODEL_INFO_SCHEMA

    class Meta:
        name = "list_models"
        description = "List all models in the registry"
        categories = ["models", "registry"]
        examples = [FunctionExample(sql="SELECT * FROM sklearn.list_models()", description="List stored models")]

    @classmethod
    def cardinality(cls, params: BindParams[NoArgs]) -> TableCardinality:
        return TableCardinality(estimate=10, max=10000)

    @classmethod
    def process(cls, params: ProcessParams[NoArgs], state: None, out: OutputCollector) -> None:
        out.emit(pa.RecordBatch.from_pydict(_meta_rows(get_store().list()), schema=params.output_schema))
        out.finish()


@dataclass(slots=True, frozen=True)
class ModelInfoArgs:
    model_name: Annotated[str, Arg(0, doc="Name of a stored model.")]


@init_single_worker
@bind_fixed_schema
class ModelInfo(TableFunctionGenerator[ModelInfoArgs]):
    FIXED_SCHEMA: ClassVar[pa.Schema] = _MODEL_INFO_SCHEMA

    class Meta:
        name = "model_info"
        description = "Describe a single stored model (one row, empty if absent)"
        categories = ["models", "registry"]
        examples = [
            FunctionExample(sql="SELECT * FROM sklearn.model_info('iris_rf')", description="Show one model's metadata")
        ]

    @classmethod
    def cardinality(cls, params: BindParams[ModelInfoArgs]) -> TableCardinality:
        return TableCardinality(estimate=1, max=1)

    @classmethod
    def process(cls, params: ProcessParams[ModelInfoArgs], state: None, out: OutputCollector) -> None:
        try:
            metas = [get_store().load_meta(params.args.model_name)]
        except ModelNotFoundError:
            metas = []
        out.emit(pa.RecordBatch.from_pydict(_meta_rows(metas), schema=params.output_schema))
        out.finish()


@dataclass(slots=True, frozen=True)
class DropModelArgs:
    model_name: Annotated[str, Arg(0, doc="Name of the model to delete.")]


_DROP_SCHEMA = pa.schema(
    [
        sfield("model_name", pa.string(), "Name of the model.", nullable=False),
        sfield("dropped", pa.bool_(), "True if a model was deleted, False if it did not exist.", nullable=False),
    ]
)


@init_single_worker
@bind_fixed_schema
class DropModel(TableFunctionGenerator[DropModelArgs]):
    FIXED_SCHEMA: ClassVar[pa.Schema] = _DROP_SCHEMA

    class Meta:
        name = "drop_model"
        description = "Delete a model from the registry"
        categories = ["models", "registry"]
        examples = [
            FunctionExample(sql="SELECT * FROM sklearn.drop_model('iris_rf')", description="Delete a stored model")
        ]

    @classmethod
    def cardinality(cls, params: BindParams[DropModelArgs]) -> TableCardinality:
        return TableCardinality(estimate=1, max=1)

    @classmethod
    def process(cls, params: ProcessParams[DropModelArgs], state: None, out: OutputCollector) -> None:
        name = params.args.model_name
        dropped = get_store().delete(name)
        out.emit(pa.RecordBatch.from_pydict({"model_name": [name], "dropped": [dropped]}, schema=params.output_schema))
        out.finish()


MODEL_FUNCTIONS: list[type] = [
    FitModel,
    PredictModel,
    CrossValPredict,
    ListModels,
    ModelInfo,
    DropModel,
]
