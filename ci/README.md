# CI: the scikit-learn worker integration suite

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs the unit tests
and this repo's sqllogictest suite (`test/sql/*.test`) against the scikit-learn
VGI worker through the **real DuckDB `vgi` extension** on every push / PR —
across **Linux, macOS, and Windows**.

## How it works (no C++ build)

Rather than building the vgi DuckDB extension from source, CI drives a
**prebuilt** standalone `haybarn-unittest` (the DuckDB/Haybarn sqllogictest
runner, published in Haybarn's releases) and installs the **signed** `vgi`
extension from the Haybarn community channel:

1. **Install the worker** — `uv sync --frozen` into a venv. The `vgi-sklearn`
   console-script (with its venv shebang) is a self-contained stdio worker the
   extension can spawn.
2. **Download the runner** — the matching `haybarn_unittest-*` asset per
   platform from the latest Haybarn release.
3. **Preprocess** — the standalone runner links none of the extensions the
   tests gate on, so [`preprocess-require.awk`](preprocess-require.awk) rewrites
   each `require <ext>` into an explicit signed `INSTALL <ext> FROM
   {community,core}; LOAD <ext>;`. `require-env` and everything else pass
   through untouched.
4. **Run** — [`run-integration.sh`](run-integration.sh) stages the preprocessed
   tree, points `VGI_SKLEARN_WORKER` at the installed `vgi-sklearn` command,
   isolates the model registry under a temp `SKLEARN_MODELS_DIR`, warms the
   extension cache once, then runs the suite in a single `haybarn-unittest`
   invocation. Any failed assertion exits non-zero and fails the job.

## Run it locally

```bash
uv sync --python 3.13                       # install the worker + deps
# point HAYBARN_UNITTEST at a haybarn-unittest binary (or a local DuckDB
# `unittest` built with the vgi extension), and the worker at the console script:
HAYBARN_UNITTEST=/path/to/haybarn-unittest \
VGI_SKLEARN_WORKER="$PWD/.venv/bin/vgi-sklearn" \
  ci/run-integration.sh
```

Or, against the local source checkouts without installing, use the Makefile
targets (`make test-stdio` / `make test-http`), which point the worker at
`uv run sklearn_worker.py`.
