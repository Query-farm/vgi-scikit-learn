"""VGI worker exposing scikit-learn to DuckDB/SQL.

Assembles the per-area implementation modules in ``vgi_sklearn`` into a single
``sklearn`` catalog and provides the process entry points. The repo-root
``sklearn_worker.py`` / ``serve.py`` are thin shims over this module for
``uv run`` and the Fly.io container; installed users get the ``vgi-sklearn`` and
``vgi-sklearn-http`` console scripts, which call ``main`` / ``main_http`` here.

    ATTACH 'sklearn' (TYPE vgi, LOCATION 'vgi-sklearn');
    SELECT * FROM sklearn.iris();
"""

from __future__ import annotations

import dataclasses
import logging
import os
import sys
from typing import Any

from vgi import Worker
from vgi.catalog import Catalog, ReadOnlyCatalogInterface, Schema
from vgi.catalog.catalog_interface import CatalogAttachResult, CatalogInfo

from vgi_sklearn import __version__
from vgi_sklearn.datasets import DATASET_FUNCTIONS
from vgi_sklearn.feature_selection import FEATURE_SELECTION_FUNCTIONS
from vgi_sklearn.grouped import GROUPED_FUNCTIONS
from vgi_sklearn.metrics import METRIC_FUNCTIONS
from vgi_sklearn.models import MODEL_FUNCTIONS
from vgi_sklearn.stored_transforms import STORED_TRANSFORM_FUNCTIONS
from vgi_sklearn.table_metrics import TABLE_METRIC_FUNCTIONS
from vgi_sklearn.text import TEXT_FUNCTIONS
from vgi_sklearn.transforms import TRANSFORM_FUNCTIONS
from vgi_sklearn.typed_models import TYPED_FIT_FUNCTIONS

try:
    # grid_search needs union-typed arguments with tag preservation, added in a
    # newer vgi-python. Against an older vgi-python the import fails and the
    # function is simply not registered (everything else still works).
    from vgi_sklearn.search import SEARCH_FUNCTIONS
except ImportError:  # pragma: no cover - depends on the installed vgi-python
    SEARCH_FUNCTIONS = []

log = logging.getLogger(__name__)

DATA_VERSION = __version__
GIT_COMMIT = os.environ.get("VGI_SKLEARN_GIT_COMMIT") or "unknown"

# Every callable the worker exposes, grouped by scikit-learn area.
_FUNCTIONS: list[type] = [
    *DATASET_FUNCTIONS,
    *METRIC_FUNCTIONS,
    *TABLE_METRIC_FUNCTIONS,
    *TRANSFORM_FUNCTIONS,
    *STORED_TRANSFORM_FUNCTIONS,
    *TEXT_FUNCTIONS,
    *FEATURE_SELECTION_FUNCTIONS,
    *MODEL_FUNCTIONS,
    *TYPED_FIT_FUNCTIONS,
    *GROUPED_FUNCTIONS,
    *SEARCH_FUNCTIONS,
]

_SKLEARN_CATALOG = Catalog(
    name="sklearn",
    default_schema="main",
    schemas=[
        Schema(
            name="main",
            comment="scikit-learn datasets, metrics, transforms, and models for SQL",
            functions=list(_FUNCTIONS),
        ),
    ],
)


class SklearnCatalog(ReadOnlyCatalogInterface):
    """Advertises the worker's data + implementation version on ATTACH."""

    catalog = _SKLEARN_CATALOG
    catalog_name = _SKLEARN_CATALOG.name

    def catalogs(self) -> list[CatalogInfo]:
        return [
            CatalogInfo(
                name=self._effective_catalog_name,
                implementation_version=GIT_COMMIT,
                data_version_spec=DATA_VERSION,
                attach_option_specs=[spec.serialize() for spec in self.attach_option_specs],
            )
        ]

    def catalog_attach(self, **kwargs: Any) -> CatalogAttachResult:
        result = super().catalog_attach(**kwargs)
        return dataclasses.replace(
            result,
            resolved_data_version=DATA_VERSION,
            resolved_implementation_version=GIT_COMMIT,
        )


class SklearnWorker(Worker):
    """Worker process hosting the scikit-learn catalog."""

    catalog = _SKLEARN_CATALOG
    catalog_interface = SklearnCatalog


def main() -> None:
    """Run the worker (stdio by default; pass ``--http`` for the HTTP server)."""
    SklearnWorker.main()


def main_http() -> None:
    """Run the worker over HTTP (injects ``--http`` into the worker CLI)."""
    argv = sys.argv[1:]
    if "--http" not in argv:
        argv = ["--http", *argv]
    sys.argv = [sys.argv[0], *argv]
    SklearnWorker.main()
