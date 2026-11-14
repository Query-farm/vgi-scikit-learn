"""Composed (multi-function) examples are placed where agents and docs see them.

The vgi-lint agent simulator only shows each object's first 5 per-function
examples (never schema-level lists), so every composed example must sit within
its driving function's first 5 and also appear in a schema-level
``vgi.example_queries`` list. Results are asserted in test/sql/sklearn_composed.test.
"""

from __future__ import annotations

import json

import pytest
from vgi.metadata import FunctionExample

from vgi_sklearn import composed_examples as ce
from vgi_sklearn import worker
from vgi_sklearn.models import CrossValPredict
from vgi_sklearn.table_metrics import ConfusionMatrix, SilhouetteScore
from vgi_sklearn.transforms import KMeansFn, PcaFn

_PLACEMENT: list[tuple[FunctionExample, type, str]] = [
    (ce.KMEANS_VS_SPECIES, KMeansFn, "preprocessing"),
    (ce.SCALED_PCA_BY_SPECIES, PcaFn, "preprocessing"),
    (ce.OOF_CONFUSION_MATRIX, CrossValPredict, "models"),
    (ce.OOF_CONFUSION_MATRIX, ConfusionMatrix, "metrics"),
    (ce.KMEANS_SILHOUETTE, SilhouetteScore, "metrics"),
]


@pytest.mark.parametrize(("example", "fn", "schema"), _PLACEMENT)
def test_composed_example_placement(example: FunctionExample, fn: type, schema: str) -> None:
    assert example in fn.Meta.examples[:5]  # type: ignore[attr-defined]
    queries = json.loads(worker._SCHEMA_META[schema]["example_queries"])
    assert ce.as_query(example) in queries
