"""scikit-learn as a VGI worker: datasets, metrics, transforms, and models for DuckDB/SQL.

The implementation is split by scikit-learn area so each module stays focused:

- ``datasets``    -- toy datasets and synthetic generators as table functions
- ``metrics``     -- scoring functions as SQL aggregates (added in phase 2)
- ``transforms``  -- unsupervised fit_transform as buffering functions (phase 3)
- ``models``      -- supervised fit/predict + model registry (phase 4)
- ``registry``    -- pluggable model store (local disk now, S3/R2 later)

``sklearn_worker.py`` at the repo root assembles these into the ``sklearn``
catalog and runs the worker.
"""

from __future__ import annotations

__version__ = "0.1.2"
