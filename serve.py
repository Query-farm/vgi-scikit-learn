# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "vgi-python[http,oauth]>=0.52.2,<0.53",
#     "vgi-rpc[sentry]",
#     "scikit-learn>=1.5",
#     "numpy",
#     "skops>=0.11",
# ]
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
