"""Typed per-estimator fit functions: ``sklearn.fit_<estimator>(...)``.

These wrap the generic ``fit`` with each estimator's common hyperparameters
exposed as **native, typed SQL named arguments** — so they show up in the
catalog and DuckDB's autocomplete, are type-checked, and are discoverable
without consulting docs:

    SELECT * FROM sklearn.fit_random_forest_classifier(
      (SELECT * FROM training), model_name := 'm', target := 'y',
      n_estimators := 300, max_depth := 8);

Each function behaves exactly like ``fit``: it returns the training summary plus
the model as a BLOB, and persists to the registry when ``model_name`` is given.
The generic ``fit`` (JSON ``params``) remains the escape hatch for hyperparameters
not surfaced here. The curated parameter set per estimator is intentionally the
common, scalar ones — see ``_HPARAMS`` below.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, make_dataclass
from dataclasses import field as dc_field
from typing import Annotated, Any

from vgi.arguments import Arg, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_buffering_function import OutputCollector, TableBufferingParams
from vgi.table_function import BindParams

from .buffering import DrainState, SinkBuffer, input_schema_of
from .models import _ESTIMATORS, _FIT_SCHEMA, _fit_and_emit
from .registry import validate_name

_UNSET: Any = object()


@dataclass(frozen=True)
class _HP:
    """One typed hyperparameter exposed as a SQL named argument."""

    name: str
    type: type
    default: Any
    doc: str
    none_if: Any = _UNSET  # if the SQL value equals this, pass None to sklearn
    kwarg: str | None = None  # sklearn kwarg name, if it differs from ``name``
    wrap_tuple: bool = False  # wrap a scalar int as a 1-tuple (hidden_layer_sizes)


# Curated common hyperparameters per estimator. Defaults match scikit-learn's,
# except where a SQL-friendly sentinel is needed (e.g. max_depth 0 => None).
_RF = [
    _HP("n_estimators", int, 100, "Number of trees."),
    _HP("max_depth", int, 0, "Max tree depth; 0 = unlimited.", none_if=0),
    _HP("min_samples_split", int, 2, "Min samples to split an internal node."),
    _HP("max_features", str, "sqrt", "Features considered per split ('sqrt', 'log2', 'all')."),
    _HP("random_state", int, 0, "Random seed."),
]
_GB = [
    _HP("n_estimators", int, 100, "Number of boosting stages."),
    _HP("learning_rate", float, 0.1, "Shrinkage applied to each tree."),
    _HP("max_depth", int, 3, "Max depth of each tree."),
    _HP("subsample", float, 1.0, "Fraction of samples per stage."),
    _HP("random_state", int, 0, "Random seed."),
]
_HGB = [
    _HP("max_iter", int, 100, "Number of boosting iterations."),
    _HP("learning_rate", float, 0.1, "Shrinkage applied to each iteration."),
    _HP("max_depth", int, 0, "Max tree depth; 0 = unlimited.", none_if=0),
    _HP("l2_regularization", float, 0.0, "L2 regularization."),
    _HP("random_state", int, 0, "Random seed."),
]
_TREE = lambda crit: [  # noqa: E731
    _HP("max_depth", int, 0, "Max tree depth; 0 = unlimited.", none_if=0),
    _HP("min_samples_split", int, 2, "Min samples to split an internal node."),
    _HP("criterion", str, crit, "Split quality criterion."),
    _HP("random_state", int, 0, "Random seed."),
]
_KNN = [
    _HP("n_neighbors", int, 5, "Number of neighbours."),
    _HP("weights", str, "uniform", "Neighbour weighting ('uniform' or 'distance')."),
    _HP("p", int, 2, "Minkowski power (1=manhattan, 2=euclidean)."),
]
_MLP = [
    _HP("hidden_units", int, 100, "Units in the single hidden layer.", kwarg="hidden_layer_sizes", wrap_tuple=True),
    _HP("alpha", float, 0.0001, "L2 penalty."),
    _HP("max_iter", int, 500, "Max training iterations."),
    _HP("learning_rate_init", float, 0.001, "Initial learning rate."),
    _HP("random_state", int, 0, "Random seed."),
]

_HPARAMS: dict[str, list[_HP]] = {
    "logistic_regression": [
        _HP("C", float, 1.0, "Inverse regularization strength."),
        _HP("max_iter", int, 1000, "Max solver iterations."),
        _HP("penalty", str, "l2", "Regularization penalty ('l2', 'l1', 'elasticnet', 'none')."),
        _HP("solver", str, "lbfgs", "Optimization solver."),
    ],
    "random_forest_classifier": _RF,
    "random_forest_regressor": _RF,
    "gradient_boosting_classifier": _GB,
    "gradient_boosting_regressor": _GB,
    "hist_gradient_boosting_classifier": _HGB,
    "hist_gradient_boosting_regressor": _HGB,
    "linear_regression": [_HP("fit_intercept", bool, True, "Whether to fit an intercept.")],
    "ridge": [
        _HP("alpha", float, 1.0, "Regularization strength."),
        _HP("fit_intercept", bool, True, "Whether to fit an intercept."),
        _HP("solver", str, "auto", "Solver to use."),
    ],
    "lasso": [
        _HP("alpha", float, 1.0, "Regularization strength."),
        _HP("max_iter", int, 1000, "Max iterations."),
        _HP("fit_intercept", bool, True, "Whether to fit an intercept."),
    ],
    "svc": [
        _HP("C", float, 1.0, "Regularization strength."),
        _HP("kernel", str, "rbf", "Kernel ('rbf', 'linear', 'poly', 'sigmoid')."),
        _HP("gamma", str, "scale", "Kernel coefficient ('scale', 'auto')."),
        _HP("degree", int, 3, "Degree for the 'poly' kernel."),
    ],
    "svr": [
        _HP("C", float, 1.0, "Regularization strength."),
        _HP("kernel", str, "rbf", "Kernel ('rbf', 'linear', 'poly', 'sigmoid')."),
        _HP("gamma", str, "scale", "Kernel coefficient ('scale', 'auto')."),
        _HP("epsilon", float, 0.1, "Epsilon-tube within which no penalty is given."),
    ],
    "knn_classifier": _KNN,
    "knn_regressor": _KNN,
    "decision_tree_classifier": _TREE("gini"),
    "decision_tree_regressor": _TREE("squared_error"),
    "mlp_classifier": _MLP,
    "mlp_regressor": _MLP,
    "gaussian_nb": [_HP("var_smoothing", float, 1e-9, "Variance smoothing added for stability.")],
    # --- additional classifiers ---
    "sgd_classifier": [
        _HP("loss", str, "hinge", "Loss function ('hinge', 'log_loss', 'modified_huber')."),
        _HP("alpha", float, 0.0001, "Regularization strength."),
        _HP("penalty", str, "l2", "Penalty ('l2', 'l1', 'elasticnet')."),
        _HP("max_iter", int, 1000, "Max passes over the training data."),
        _HP("random_state", int, 0, "Random seed."),
    ],
    "ridge_classifier": [
        _HP("alpha", float, 1.0, "Regularization strength."),
        _HP("fit_intercept", bool, True, "Whether to fit an intercept."),
        _HP("solver", str, "auto", "Solver to use."),
    ],
    "extra_trees_classifier": _RF,
    "extra_trees_regressor": _RF,
    "ada_boost_classifier": [
        _HP("n_estimators", int, 50, "Number of boosting stages."),
        _HP("learning_rate", float, 1.0, "Weight applied to each classifier."),
        _HP("random_state", int, 0, "Random seed."),
    ],
    "ada_boost_regressor": [
        _HP("n_estimators", int, 50, "Number of boosting stages."),
        _HP("learning_rate", float, 1.0, "Weight applied to each regressor."),
        _HP("random_state", int, 0, "Random seed."),
    ],
    "bagging_classifier": [
        _HP("n_estimators", int, 10, "Number of base estimators."),
        _HP("max_samples", float, 1.0, "Fraction of samples per base estimator."),
        _HP("max_features", float, 1.0, "Fraction of features per base estimator."),
        _HP("random_state", int, 0, "Random seed."),
    ],
    "bagging_regressor": [
        _HP("n_estimators", int, 10, "Number of base estimators."),
        _HP("max_samples", float, 1.0, "Fraction of samples per base estimator."),
        _HP("max_features", float, 1.0, "Fraction of features per base estimator."),
        _HP("random_state", int, 0, "Random seed."),
    ],
    "linear_svc": [
        _HP("C", float, 1.0, "Regularization strength."),
        _HP("loss", str, "squared_hinge", "Loss ('hinge' or 'squared_hinge')."),
        _HP("max_iter", int, 1000, "Max iterations."),
        _HP("random_state", int, 0, "Random seed."),
    ],
    "multinomial_nb": [
        _HP("alpha", float, 1.0, "Additive (Laplace) smoothing."),
        _HP("fit_prior", bool, True, "Whether to learn class prior probabilities."),
    ],
    "bernoulli_nb": [
        _HP("alpha", float, 1.0, "Additive (Laplace) smoothing."),
        _HP("binarize", float, 0.0, "Threshold for binarizing features."),
        _HP("fit_prior", bool, True, "Whether to learn class prior probabilities."),
    ],
    "complement_nb": [
        _HP("alpha", float, 1.0, "Additive (Laplace) smoothing."),
        _HP("norm", bool, False, "Whether to normalize the weights."),
        _HP("fit_prior", bool, True, "Whether to learn class prior probabilities."),
    ],
    "lda": [
        _HP("solver", str, "svd", "Solver ('svd', 'lsqr', 'eigen')."),
        _HP("tol", float, 1e-4, "Threshold for rank estimation (svd)."),
    ],
    "qda": [
        _HP("reg_param", float, 0.0, "Regularization of the per-class covariance."),
        _HP("tol", float, 1e-4, "Threshold for rank estimation."),
    ],
    # --- additional regressors ---
    "elastic_net": [
        _HP("alpha", float, 1.0, "Regularization strength."),
        _HP("l1_ratio", float, 0.5, "Mix of L1/L2 (0=ridge, 1=lasso)."),
        _HP("max_iter", int, 1000, "Max iterations."),
        _HP("fit_intercept", bool, True, "Whether to fit an intercept."),
    ],
    "sgd_regressor": [
        _HP("loss", str, "squared_error", "Loss ('squared_error', 'huber', 'epsilon_insensitive')."),
        _HP("alpha", float, 0.0001, "Regularization strength."),
        _HP("penalty", str, "l2", "Penalty ('l2', 'l1', 'elasticnet')."),
        _HP("max_iter", int, 1000, "Max passes over the training data."),
        _HP("random_state", int, 0, "Random seed."),
    ],
    "linear_svr": [
        _HP("C", float, 1.0, "Regularization strength."),
        _HP("epsilon", float, 0.0, "Epsilon in the epsilon-insensitive loss."),
        _HP("max_iter", int, 1000, "Max iterations."),
        _HP("random_state", int, 0, "Random seed."),
    ],
    "poisson_regressor": [
        _HP("alpha", float, 1.0, "L2 regularization strength."),
        _HP("max_iter", int, 100, "Max solver iterations."),
        _HP("fit_intercept", bool, True, "Whether to fit an intercept."),
    ],
    "gamma_regressor": [
        _HP("alpha", float, 1.0, "L2 regularization strength."),
        _HP("max_iter", int, 100, "Max solver iterations."),
        _HP("fit_intercept", bool, True, "Whether to fit an intercept."),
    ],
    "tweedie_regressor": [
        _HP("power", float, 0.0, "Tweedie power (0=normal, 1=Poisson, 2=Gamma, 3=inverse-Gaussian)."),
        _HP("alpha", float, 1.0, "L2 regularization strength."),
        _HP("max_iter", int, 100, "Max solver iterations."),
    ],
    "bayesian_ridge": [
        _HP("alpha_1", float, 1e-6, "Shape parameter for the alpha Gamma prior."),
        _HP("lambda_1", float, 1e-6, "Shape parameter for the lambda Gamma prior."),
        _HP("fit_intercept", bool, True, "Whether to fit an intercept."),
    ],
    "huber_regressor": [
        _HP("epsilon", float, 1.35, "Threshold above which samples are treated as outliers."),
        _HP("alpha", float, 0.0001, "L2 regularization strength."),
        _HP("max_iter", int, 100, "Max iterations."),
    ],
    "quantile_regressor": [
        _HP("quantile", float, 0.5, "Quantile to predict (0-1)."),
        _HP("alpha", float, 1.0, "L1 regularization strength."),
        _HP("fit_intercept", bool, True, "Whether to fit an intercept."),
    ],
}


def _estimator_kwargs(spec: list[_HP], args: Any) -> dict[str, Any]:
    """Translate the typed SQL args into scikit-learn estimator kwargs."""
    kw: dict[str, Any] = {}
    for hp in spec:
        v = getattr(args, hp.name)
        if hp.none_if is not _UNSET and v == hp.none_if:
            v = None
        elif hp.wrap_tuple:
            v = (int(v),)
        kw[hp.kwarg or hp.name] = v
    return kw


def _make_args_class(est_name: str, spec: list[_HP]) -> type:
    fields: list[Any] = [
        ("data", Annotated[TableInput, Arg(0, doc="Training table (features + target [+ id]).")]),
        (
            "model_name",
            Annotated[
                str,
                Arg("model_name", default="", doc="Optional registry name; the model is always returned as a BLOB."),
            ],
            dc_field(default=""),
        ),
        (
            "target",
            Annotated[str, Arg("target", default="", doc="Label column name (required).")],
            dc_field(default=""),
        ),
        ("id", Annotated[str, Arg("id", default="", doc="Optional id passthrough column.")], dc_field(default="")),
    ]
    for hp in spec:
        fields.append(
            (hp.name, Annotated[hp.type, Arg(hp.name, default=hp.default, doc=hp.doc)], dc_field(default=hp.default))
        )
    cls_name = "Fit" + "".join(p.title() for p in est_name.split("_")) + "Args"
    return make_dataclass(cls_name, fields, frozen=True, slots=True)


def _make_fit_function(est_name: str) -> type:
    task, est_cls, defaults = _ESTIMATORS[est_name]
    spec = _HPARAMS[est_name]
    args_cls = _make_args_class(est_name, spec)
    fn_name = f"fit_{est_name}"
    param_hint = ", ".join(f"{hp.name} := {hp.default!r}" for hp in spec[:2])

    def on_bind(cls: type, params: BindParams[Any]) -> BindResponse:
        a = params.args
        if not a.target:
            raise ValueError(f"{fn_name} requires 'target' (the label column name, e.g. target := 'label')")
        if a.model_name:
            validate_name(a.model_name)
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        if a.target not in input_schema.names:
            raise ValueError(f"target column {a.target!r} not found in input; columns: {', '.join(input_schema.names)}")
        return BindResponse(output_schema=_FIT_SCHEMA)

    def initial_finalize_state(cls: type, finalize_state_id: bytes, params: TableBufferingParams[Any]) -> DrainState:
        return DrainState()

    def finalize(
        cls: type,
        params: TableBufferingParams[Any],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        if state.done:
            out.finish()
            return
        state.done = True
        a = params.args
        kwargs = _estimator_kwargs(spec, a)
        estimator = est_cls(**{**defaults, **kwargs})
        table = cls.buffered_table(params, input_schema_of(params))
        _fit_and_emit(
            out,
            params.output_schema,
            table=table,
            input_schema=input_schema_of(params),
            estimator_label=est_name,
            task=task,
            estimator=estimator,
            model_name=a.model_name,
            target=a.target,
            id_col=a.id,
            params_dict=kwargs,
        )

    meta = type(
        "Meta",
        (),
        {
            "name": fn_name,
            "description": f"Fit a {est_name} with typed hyperparameters; returns/stores the model",
            "categories": ["models", "supervised", "typed"],
            "examples": [
                FunctionExample(
                    sql=(
                        f"SELECT * FROM sklearn.{fn_name}((SELECT * FROM training), "
                        f"model_name := 'm', target := 'y'" + (f", {param_hint}" if param_hint else "") + ")"
                    ),
                    description=f"Train a {est_name} with named hyperparameters",
                )
            ],
        },
    )
    namespace = {
        "FunctionArguments": args_cls,
        "Meta": meta,
        "on_bind": classmethod(on_bind),
        "initial_finalize_state": classmethod(initial_finalize_state),
        "finalize": classmethod(finalize),
    }
    cls_name = "Fit" + "".join(p.title() for p in est_name.split("_"))
    return types.new_class(cls_name, (SinkBuffer[args_cls, DrainState],), {}, lambda ns: ns.update(namespace))


TYPED_FIT_FUNCTIONS: list[type] = [_make_fit_function(name) for name in _HPARAMS]
