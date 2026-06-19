# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http,oauth]",
#     "vgi-rpc[sentry]",
#     "scikit-learn>=1.5",
#     "numpy",
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
"""HTTP entrypoint for the scikit-learn worker (used by Fly.io).

Forces the worker's CLI into HTTP mode (``Worker.main()`` serves stdio by
default) so callers only pass ``--host``/``--port``.
"""

import sys

from sklearn_worker import SklearnWorker

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--http" not in argv:
        argv = ["--http", *argv]
    sys.argv = [sys.argv[0], *argv]
    SklearnWorker.main()
