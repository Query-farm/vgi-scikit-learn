"""The worker must advertise the released package version over VGI.

`implementation_version` is a semver per the VGI protocol — the worker *software*
version — so it has to equal the released package version (and the PyPI wheel and
container tag), not a build/commit id. This locks that contract.
"""

from __future__ import annotations

import vgi_sklearn
from vgi_sklearn import worker


def test_implementation_version_is_release_version() -> None:
    assert vgi_sklearn.__version__ == worker.IMPLEMENTATION_VERSION


def test_data_version_is_release_version() -> None:
    assert vgi_sklearn.__version__ == worker.DATA_VERSION


def test_catalog_info_advertises_release_version() -> None:
    """The CatalogInfo emitted on ATTACH carries the release version, not the sha."""
    info = worker.SklearnCatalog().catalogs()[0]
    assert info.implementation_version == vgi_sklearn.__version__
    assert info.data_version_spec == vgi_sklearn.__version__
