"""Unit tests for the fit_pipeline step-spec parsing.

The full fit/predict lifecycle is covered by test/sql/sklearn_pipeline.test.
"""

from __future__ import annotations

import pytest

from vgi_sklearn.pipeline import _parse_steps


def test_empty_steps() -> None:
    assert _parse_steps("") == []
    assert _parse_steps("  ") == []


def test_valid_steps_with_and_without_params() -> None:
    steps = _parse_steps('[{"kind": "standard_scaler"}, {"kind": "pca", "params": {"n_components": 2}}]')
    assert steps == [("standard_scaler", {}), ("pca", {"n_components": 2})]


def test_non_array_rejected() -> None:
    with pytest.raises(ValueError, match="must be a JSON array"):
        _parse_steps('{"kind": "pca"}')


def test_step_without_kind_rejected() -> None:
    with pytest.raises(ValueError, match="must be an object with a 'kind'"):
        _parse_steps('[{"params": {}}]')


def test_unknown_kind_rejected() -> None:
    with pytest.raises(ValueError, match="unknown step kind"):
        _parse_steps('[{"kind": "not_a_transform"}]')


def test_non_object_params_rejected() -> None:
    with pytest.raises(ValueError, match="params must be a JSON object"):
        _parse_steps('[{"kind": "pca", "params": [1, 2]}]')
