"""Typed per-estimator fit functions: ``sklearn.fit_<estimator>(...)``.

These wrap the generic ``fit`` with each estimator's common hyperparameters
exposed as **native, typed SQL named arguments** — so they show up in the
catalog and DuckDB's autocomplete, are type-checked, and are discoverable
without consulting docs:

    SELECT * FROM sklearn.estimators.fit_random_forest_classifier(
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
from vgi.table_buffering_function import TableBufferingParams
from vgi.table_function import BindParams
from vgi_rpc.rpc import OutputCollector

from .buffering import DrainState, SinkBuffer, input_schema_of
from .models import _ESTIMATORS, _FIT_SCHEMA, _fit_and_emit
from .registry import validate_name
from .schema_utils import columns_md

_FIT_COLUMNS_MD = columns_md(_FIT_SCHEMA)

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


def _TREE(crit: str) -> list[_HP]:
    """Common decision-tree hyperparameters with the given default split criterion."""
    return [
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


# Per-estimator prose: (task word, one-sentence "what it is / how it works",
# "best for" guidance). Used to synthesize the rich description tags so each
# generated fit_<estimator> reads specifically, not from a template.
_DESC: dict[str, tuple[str, str, str]] = {
    "logistic_regression": (
        "classifier",
        "fits a linear decision boundary and squashes it through a logistic (sigmoid/softmax) "
        "link to produce calibrated class probabilities",
        "a fast, interpretable linear baseline on roughly linearly-separable, well-scaled data",
    ),
    "random_forest_classifier": (
        "classifier",
        "averages a bagged ensemble of decision trees, each grown on a bootstrap sample with a "
        "random feature subset per split, to cut the variance of a single tree",
        "a strong, low-tuning default on tabular data with mixed feature types and non-linear interactions",
    ),
    "random_forest_regressor": (
        "regressor",
        "averages a bagged ensemble of regression trees grown on bootstrap samples with random "
        "per-split feature subsets to reduce variance",
        "a robust, low-tuning default for non-linear tabular regression",
    ),
    "gradient_boosting_classifier": (
        "classifier",
        "builds shallow trees sequentially, each correcting the residual errors of the ensemble so "
        "far via gradient descent on the loss",
        "squeezing accuracy out of small-to-medium tabular data when you can afford to tune learning rate and depth",
    ),
    "gradient_boosting_regressor": (
        "regressor",
        "fits shallow regression trees stage by stage, each one descending the gradient of the "
        "squared-error loss to correct prior residuals",
        "accurate non-linear regression on small-to-medium data with careful learning-rate/depth tuning",
    ),
    "hist_gradient_boosting_classifier": (
        "classifier",
        "is a histogram-binned gradient-boosting classifier (LightGBM-style) that bins features for "
        "fast split-finding and natively handles missing values",
        "large tabular datasets where plain gradient boosting is too slow",
    ),
    "hist_gradient_boosting_regressor": (
        "regressor",
        "is a histogram-binned gradient-boosting regressor (LightGBM-style) that bins features for "
        "fast split-finding and natively handles missing values",
        "fast, accurate regression on large tabular datasets",
    ),
    "linear_regression": (
        "regressor",
        "fits an ordinary-least-squares linear model, minimizing the sum of squared residuals with a "
        "closed-form solution",
        "an interpretable baseline when the target is roughly a linear function of the features",
    ),
    "ridge": (
        "regressor",
        "is linear regression with an L2 penalty that shrinks coefficients toward zero to tame "
        "multicollinearity and overfitting",
        "linear regression with many correlated features; tune `alpha` for the shrinkage strength",
    ),
    "lasso": (
        "regressor",
        "is linear regression with an L1 penalty that drives some coefficients exactly to zero, "
        "performing built-in feature selection",
        "sparse linear models where you want automatic feature selection",
    ),
    "svc": (
        "classifier",
        "is a kernel support-vector classifier that finds the maximum-margin boundary, using the "
        "kernel trick to model non-linear separations",
        "small-to-medium datasets with clear margins; try the `rbf` kernel and tune `C`/`gamma`",
    ),
    "svr": (
        "regressor",
        "is kernel support-vector regression fitting a tube of width `epsilon` around the data and "
        "penalizing only points outside it",
        "small-to-medium non-linear regression robust to mild noise within the epsilon tube",
    ),
    "knn_classifier": (
        "classifier",
        "is a lazy k-nearest-neighbours classifier that labels a point by majority vote of its "
        "closest training examples (no explicit training)",
        "low-dimensional, well-scaled data with smooth local class structure",
    ),
    "knn_regressor": (
        "regressor",
        "is a lazy k-nearest-neighbours regressor that predicts the (optionally distance-weighted) "
        "mean target of its closest training examples",
        "low-dimensional, well-scaled regression with smooth local structure",
    ),
    "decision_tree_classifier": (
        "classifier",
        "grows a single axis-aligned decision tree, recursively splitting on the feature that most "
        "improves the gini/entropy criterion",
        "an interpretable, white-box model or a quick non-linear baseline (prone to overfit unbounded)",
    ),
    "decision_tree_regressor": (
        "regressor",
        "grows a single regression tree, recursively splitting to minimize within-leaf error and "
        "predicting the leaf mean",
        "an interpretable, piecewise-constant non-linear baseline (bound `max_depth` to avoid overfit)",
    ),
    "mlp_classifier": (
        "classifier",
        "is a feed-forward neural network (multi-layer perceptron) trained by backpropagation, "
        "learning non-linear decision boundaries through a hidden layer",
        "complex non-linear patterns on well-scaled data when you can tune size and regularization",
    ),
    "mlp_regressor": (
        "regressor",
        "is a feed-forward neural network (multi-layer perceptron) trained by backpropagation to "
        "approximate a non-linear target function",
        "complex non-linear regression on well-scaled data with enough samples to train a network",
    ),
    "gaussian_nb": (
        "classifier",
        "is a Gaussian naive-Bayes classifier assuming each feature is class-conditionally normal "
        "and independent, giving an extremely fast closed-form fit",
        "a very fast probabilistic baseline on continuous features",
    ),
    "sgd_classifier": (
        "classifier",
        "fits a linear classifier (SVM/logistic, set by `loss`) by stochastic gradient descent, "
        "scaling to very large or streaming datasets",
        "large-scale linear classification where batch solvers are too slow",
    ),
    "ridge_classifier": (
        "classifier",
        "casts classification as L2-penalized least-squares regression on the class labels, picking "
        "the nearest class target",
        "a fast linear classifier on many correlated features",
    ),
    "extra_trees_classifier": (
        "classifier",
        "is an extremely-randomized-trees ensemble that, unlike a random forest, also picks split "
        "thresholds at random — trading a little bias for lower variance and faster fits",
        "a fast, low-variance alternative to a random forest on noisy tabular data",
    ),
    "extra_trees_regressor": (
        "regressor",
        "is an extremely-randomized-trees regression ensemble that randomizes split thresholds for "
        "lower variance and faster training than a random forest",
        "fast, low-variance tabular regression",
    ),
    "ada_boost_classifier": (
        "classifier",
        "builds an AdaBoost ensemble of weak learners, reweighting misclassified samples each round "
        "so later learners focus on the hard cases",
        "boosting simple base learners on clean, low-noise data (sensitive to outliers)",
    ),
    "ada_boost_regressor": (
        "regressor",
        "builds an AdaBoost regression ensemble, reweighting high-residual samples each round so "
        "later learners focus on the hardest targets",
        "boosting simple regressors on clean, low-noise data",
    ),
    "bagging_classifier": (
        "classifier",
        "trains an ensemble of base classifiers on bootstrap samples and votes their predictions, "
        "reducing variance through bagging",
        "stabilizing a high-variance base learner via bootstrap aggregation",
    ),
    "bagging_regressor": (
        "regressor",
        "trains an ensemble of base regressors on bootstrap samples and averages them, reducing "
        "variance through bagging",
        "stabilizing a high-variance base regressor via bootstrap aggregation",
    ),
    "linear_svc": (
        "classifier",
        "is a linear support-vector classifier solved in the primal (liblinear), scaling to far more "
        "samples and features than kernel `svc`",
        "large, high-dimensional, linearly-separable problems such as text classification",
    ),
    "multinomial_nb": (
        "classifier",
        "is a multinomial naive-Bayes classifier modelling feature counts, the classic choice for "
        "bag-of-words text with non-negative features",
        "text/count features such as TF or term-frequency vectors",
    ),
    "bernoulli_nb": (
        "classifier",
        "is a Bernoulli naive-Bayes classifier over binary (present/absent) features, binarizing "
        "inputs at a threshold first",
        "binary or boolean-ized features such as word-presence indicators",
    ),
    "complement_nb": (
        "classifier",
        "is the complement naive-Bayes variant that estimates each class from the complement of its "
        "data, correcting multinomial NB's bias on imbalanced text",
        "imbalanced text-classification problems",
    ),
    "lda": (
        "classifier",
        "is linear discriminant analysis: it fits class-conditional Gaussians with a shared "
        "covariance to derive a linear boundary and can double as a supervised dimensionality reducer",
        "linearly-separable classes and as a supervised projection",
    ),
    "qda": (
        "classifier",
        "is quadratic discriminant analysis: per-class Gaussians with their own covariance, yielding "
        "a curved (quadratic) decision boundary",
        "classes with differing covariance structure and enough samples per class",
    ),
    "elastic_net": (
        "regressor",
        "is linear regression with a blend of L1 and L2 penalties (`l1_ratio`), combining lasso's "
        "sparsity with ridge's stability on correlated features",
        "high-dimensional regression with groups of correlated, partly-redundant features",
    ),
    "sgd_regressor": (
        "regressor",
        "fits a linear regressor (set by `loss`) by stochastic gradient descent, scaling to very "
        "large or streaming datasets",
        "large-scale linear regression where batch solvers are too slow",
    ),
    "linear_svr": (
        "regressor",
        "is a linear support-vector regressor solved in the primal (liblinear), scaling to many more "
        "samples and features than kernel `svr`",
        "large, high-dimensional linear regression",
    ),
    "poisson_regressor": (
        "regressor",
        "is a generalized linear model with a Poisson likelihood and log link, modelling non-negative count targets",
        "count-valued targets such as event or claim frequencies",
    ),
    "gamma_regressor": (
        "regressor",
        "is a generalized linear model with a Gamma likelihood and log link, modelling positive, "
        "right-skewed continuous targets",
        "positive skewed targets such as costs, durations, or claim severities",
    ),
    "tweedie_regressor": (
        "regressor",
        "is a generalized linear model spanning the Tweedie family (`power` selects normal, Poisson, "
        "Gamma, or compound-Poisson) for flexible exponential-family targets",
        "targets with a mixed mass-at-zero and continuous part such as insurance pure premiums",
    ),
    "bayesian_ridge": (
        "regressor",
        "is Bayesian ridge regression that infers the regularization strength from the data via "
        "Gamma priors and returns predictive uncertainty",
        "linear regression when you want automatic regularization and uncertainty estimates",
    ),
    "huber_regressor": (
        "regressor",
        "is a robust linear regressor using the Huber loss (quadratic for small residuals, linear "
        "beyond `epsilon`) so outliers pull the fit less",
        "linear regression on data with outliers in the target",
    ),
    "quantile_regressor": (
        "regressor",
        "fits a linear model to a chosen target `quantile` via the pinball loss, predicting "
        "conditional quantiles rather than the mean",
        "estimating prediction intervals or modelling a specific quantile (e.g. the median)",
    ),
}


def _description_tags(est_name: str, spec: list[_HP]) -> dict[str, str]:
    """Build estimator-specific ``vgi.description_llm`` / ``vgi.description_md`` tags."""
    task, how, best_for = _DESC[est_name]
    hp_names = [hp.name for hp in spec]
    hp_inline = ", ".join(f"`{n}`" for n in hp_names)
    llm = (
        f"Table function that trains a `{est_name}` {task} on a buffered training relation "
        f"`(SELECT ...)` (Arg(0)), with its common hyperparameters exposed as typed, named SQL "
        f"arguments ({hp_inline}). A `{est_name}` {how}; it is well suited to {best_for}. Name the "
        f"`target :=` label column (required) and optionally an `id :=` passthrough to exclude; every "
        f"other column is a feature and string columns auto one-hot-encode. Returns a one-row summary "
        f"(estimator, task, sample/feature counts, train score, feature list) plus the fitted `model` "
        f"as a self-contained BLOB, and also persists it to the registry when `model_name :=` is given. "
        f"Behaves exactly like generic `fit` but with discoverable, type-checked hyperparameters; score "
        f"new rows with `predict` (features align by name)."
    )
    hp_bullets = "".join(f"  - {hp.name}: {hp.doc}\n" for hp in spec)
    md = (
        f"**fit_{est_name}** — train a {task} with typed hyperparameters; return a summary + a reusable "
        f"model BLOB.\n\n"
        f"A `{est_name}` {how}, making it a good fit for {best_for}.\n\n"
        f"- Input: `(SELECT ...)` training table; `target :=` label (**required**); `id :=` excluded "
        f"passthrough; `model_name :=` to also save to the registry\n"
        f"- Typed hyperparameters (named SQL args):\n"
        f"{hp_bullets}"
        f"- Output: one row with the estimator, task, sample/feature counts, train score, the feature "
        f"list, and the `model` BLOB\n"
        f"- String features one-hot-encode automatically; `predict` aligns features by name "
        f"(order-insensitive, extra columns ignored)"
    )
    return {"vgi.doc_llm": llm, "vgi.doc_md": md}


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
        ("data", Annotated[TableInput, Arg(0, doc="Training rows: features + target [+ id].")]),
        (
            "model_name",
            Annotated[
                str,
                Arg("model_name", default="", doc="Optional registry name; the model is always returned regardless."),
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
        table = cls.buffered_table(params, input_schema_of(params))  # type: ignore[attr-defined]  # dynamically-generated SinkBuffer subclass
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
                        f"SELECT * FROM sklearn.estimators.{fn_name}((SELECT * FROM sklearn.datasets.iris()), "
                        f"target := 'target'" + (f", {param_hint}" if param_hint else "") + ")"
                    ),
                    description=f"Train a {est_name} with named hyperparameters",
                )
            ],
            "tags": {"vgi.result_columns_md": _FIT_COLUMNS_MD, **_description_tags(est_name, spec)},
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
    base = SinkBuffer[args_cls, DrainState]  # type: ignore[valid-type]  # args_cls is a runtime-built dataclass
    return types.new_class(cls_name, (base,), {}, lambda ns: ns.update(namespace))


TYPED_FIT_FUNCTIONS: list[type] = [_make_fit_function(name) for name in _HPARAMS]
