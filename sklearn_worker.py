# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http,oauth]",
#     "vgi-rpc[sentry]",
#     "scikit-learn>=1.5",
#     "numpy",
#     "skops>=0.11",
# ]
#
# [tool.uv.sources]
# vgi-python = { path = "../vgi-python" }
# vgi-rpc = { path = "../vgi-rpc" }
#
# [tool.uv]
# # Use the local vgi-rpc checkout even if it lags vgi-python's pinned lower bound.
# override-dependencies = ["vgi-rpc>=0.20.3"]
# ///
"""VGI worker exposing scikit-learn to DuckDB/SQL.

Assembles the per-area implementation modules in ``vgi_sklearn`` into a single
``sklearn`` catalog and runs the worker over stdio (local) or HTTP (Fly.io).

Usage:
    uv run sklearn_worker.py            # serve over stdio (DuckDB subprocess)
    python serve.py --port 8000         # serve over HTTP

    ATTACH 'sklearn' (TYPE vgi, LOCATION 'uv run sklearn_worker.py');
    SELECT * FROM sklearn.iris();
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any

from vgi import Worker
from vgi.catalog import Catalog, ReadOnlyCatalogInterface, Schema
from vgi.catalog.catalog_interface import CatalogAttachResult, CatalogInfo

from vgi_sklearn import __version__
from vgi_sklearn.datasets import DATASET_FUNCTIONS
from vgi_sklearn.metrics import METRIC_FUNCTIONS
from vgi_sklearn.models import MODEL_FUNCTIONS
from vgi_sklearn.table_metrics import TABLE_METRIC_FUNCTIONS
from vgi_sklearn.transforms import TRANSFORM_FUNCTIONS
from vgi_sklearn.typed_models import TYPED_FIT_FUNCTIONS

log = logging.getLogger(__name__)

DATA_VERSION = __version__
GIT_COMMIT = os.environ.get("VGI_SKLEARN_GIT_COMMIT") or "unknown"

# Every callable the worker exposes, grouped by scikit-learn area. Phases 2-4
# append metrics / transforms / models here.
_FUNCTIONS: list[type] = [
    *DATASET_FUNCTIONS,
    *METRIC_FUNCTIONS,
    *TABLE_METRIC_FUNCTIONS,
    *TRANSFORM_FUNCTIONS,
    *MODEL_FUNCTIONS,
    *TYPED_FIT_FUNCTIONS,
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
    """Run the scikit-learn worker process (stdio or, via flags, HTTP)."""
    SklearnWorker.main()


if __name__ == "__main__":
    main()
