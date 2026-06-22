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
"""HTTP entry shim for the scikit-learn VGI worker (used by the Fly.io container).

Forces the worker CLI into HTTP mode. The implementation lives in
``vgi_sklearn.worker``; installed users invoke the ``vgi-sklearn-http`` console
script (which points at ``vgi_sklearn.worker:main_http``) instead.
"""

from vgi_sklearn.worker import SklearnWorker, main_http

__all__ = ["SklearnWorker", "main_http"]


def main() -> None:
    """Run the worker over HTTP."""
    main_http()


if __name__ == "__main__":
    main()
