"""Every registered estimator must actually fit + predict on real data.

The typed-param test only checks that exposed hyperparameters are *valid*; it
never calls ``.fit()``. This fits every estimator on appropriate toy data
(classification on iris, regression on the positive-valued diabetes target — so
the Poisson/Gamma/Tweedie GLMs are happy) and scores a few rows, catching
per-estimator runtime failures that param validation can't.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.datasets import load_diabetes, load_iris

from vgi_sklearn.features import wrap_estimator
from vgi_sklearn.models import _ESTIMATORS, CLASSIFICATION, build_estimator

_XC, _YC = load_iris(return_X_y=True)
_XR, _YR = load_diabetes(return_X_y=True)


@pytest.mark.parametrize("name", sorted(_ESTIMATORS))
def test_estimator_fits_and_predicts(name: str) -> None:
    task, est = build_estimator(name, {})
    x, y = (_XC, _YC) if task == CLASSIFICATION else (_XR, _YR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(x, y)
        preds = est.predict(x[:5])
    assert len(preds) == 5


# QDA computes a per-class covariance matrix, which is singular when features
# include collinear one-hot dummy columns (hi + lo == 1) — an inherent QDA
# limitation, not a worker bug. Everything else must fit through the pipeline.
_CATEGORICAL_INCOMPATIBLE = {"qda"}


@pytest.mark.parametrize("name", sorted(_ESTIMATORS))
def test_estimator_fits_with_categorical_pipeline(name: str) -> None:
    if name in _CATEGORICAL_INCOMPATIBLE:
        pytest.skip(f"{name} cannot fit collinear one-hot dummies (singular covariance)")
    # Prepend a string column so wrap_estimator builds the one-hot Pipeline; every
    # estimator must still fit through it (mirrors the auto-encode path in fit).
    task, est = build_estimator(name, {})
    base_x, y = (_XC, _YC) if task == CLASSIFICATION else (_XR, _YR)
    cats = np.where(base_x[:, 0] > np.median(base_x[:, 0]), "hi", "lo")
    x = np.column_stack([cats, base_x.astype(object)])
    cat_mask = [True] + [False] * base_x.shape[1]
    wrapped = wrap_estimator(est, cat_mask)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wrapped.fit(x, y)
        assert len(wrapped.predict(x[:5])) == 5
