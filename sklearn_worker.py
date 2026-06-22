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
"""Stdio entry shim for the scikit-learn VGI worker.

Lets the worker run straight from a source checkout (``uv run
sklearn_worker.py``) and from the Fly.io container, and keeps ``import
sklearn_worker`` working for tests. The implementation lives in
``vgi_sklearn.worker``; installed users invoke the ``vgi-sklearn`` console
script (which points at ``vgi_sklearn.worker:main``) instead.

    ATTACH 'sklearn' (TYPE vgi, LOCATION 'uv run sklearn_worker.py');
    SELECT * FROM sklearn.iris();
"""

from vgi_sklearn.worker import SklearnWorker, main

__all__ = ["SklearnWorker", "main"]

if __name__ == "__main__":
    main()
