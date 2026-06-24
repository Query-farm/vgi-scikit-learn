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
from vgi_sklearn.pipeline import PIPELINE_FUNCTIONS
from vgi_sklearn.search import SEARCH_FUNCTIONS
from vgi_sklearn.splitters import SPLITTER_FUNCTIONS
from vgi_sklearn.stored_transforms import STORED_TRANSFORM_FUNCTIONS
from vgi_sklearn.table_metrics import TABLE_METRIC_FUNCTIONS
from vgi_sklearn.text import TEXT_FUNCTIONS
from vgi_sklearn.transforms import TRANSFORM_FUNCTIONS
from vgi_sklearn.typed_models import TYPED_FIT_FUNCTIONS

log = logging.getLogger(__name__)

# The version the worker advertises over VGI. `implementation_version` is the
# worker *software* version (a semver per the VGI protocol), so it must be the
# released package version — not a build/commit id. Both it and the data version
# track __version__, which is the single source bumped per release.
IMPLEMENTATION_VERSION = __version__
DATA_VERSION = __version__
# Build provenance only (Sentry release / diagnostics) — NOT the advertised
# implementation version, which must stay a semver.
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
    *PIPELINE_FUNCTIONS,
    *SPLITTER_FUNCTIONS,
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
                implementation_version=IMPLEMENTATION_VERSION,
                data_version_spec=DATA_VERSION,
                attach_option_specs=[spec.serialize() for spec in self.attach_option_specs],
            )
        ]

    def catalog_attach(self, **kwargs: Any) -> CatalogAttachResult:
        result = super().catalog_attach(**kwargs)
        return dataclasses.replace(
            result,
            resolved_data_version=DATA_VERSION,
            resolved_implementation_version=IMPLEMENTATION_VERSION,
        )


class SklearnWorker(Worker):
    """Worker process hosting the scikit-learn catalog."""

    catalog = _SKLEARN_CATALOG
    catalog_interface = SklearnCatalog


def _warn_if_ephemeral_state() -> None:
    """Warn when the worker's state dirs look container-local (no volume mounted).

    The published image declares a ``/data`` volume (advertised via the
    ``farm.query.vgi.volumes`` image label) that holds the model registry and the
    shared ``BoundStorage`` SQLite. If the worker runs with those defaults but
    ``/data`` is not actually a mounted volume, models and shared state live on
    the container's writable layer and vanish on ``docker run --rm`` — and are not
    shared across instances. Surface that loudly instead of silently losing data.

    A no-op outside that container shape: it only fires when the state dirs are
    rooted under ``/data`` and ``/proc/mounts`` is readable (a Linux container).
    Never raises — an unmounted run is still valid for ephemeral use.
    """
    sqlite_dir = os.path.dirname(os.environ.get("VGI_WORKER_SQLITE_PATH", ""))
    roots = [p for p in (os.environ.get("SKLEARN_MODELS_DIR", ""), sqlite_dir) if p.startswith("/data")]
    if not roots:
        return
    try:
        with open("/proc/mounts", encoding="utf-8") as fh:  # Linux container only
            mountpoints = {parts[1] for line in fh if len(parts := line.split()) > 1}
    except OSError:
        return
    if "/data" not in mountpoints and not any(r in mountpoints for r in roots):
        log.warning(
            "state directory /data is not a mounted volume: the model registry and "
            "shared BoundStorage are container-local and will NOT persist across "
            "restarts or be shared across worker instances. Mount a volume at /data "
            "(the image advertises this via the 'farm.query.vgi.volumes' label)."
        )


def main() -> None:
    """Run the worker (stdio by default; pass ``--http`` for the HTTP server)."""
    _warn_if_ephemeral_state()
    SklearnWorker.main()


def main_http() -> None:
    """Run the worker over HTTP (injects ``--http`` into the worker CLI)."""
    _warn_if_ephemeral_state()
    argv = sys.argv[1:]
    if "--http" not in argv:
        argv = ["--http", *argv]
    sys.argv = [sys.argv[0], *argv]
    SklearnWorker.main()
