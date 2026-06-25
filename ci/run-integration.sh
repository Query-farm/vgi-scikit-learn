#!/usr/bin/env bash
# Copyright 2026 Query Farm LLC - https://query.farm
#
# Run this repo's sqllogictest suite (test/sql/*.test) against the scikit-learn
# VGI worker, using a prebuilt standalone `haybarn-unittest` and the signed
# community `vgi` extension — no C++ build from source. See ci/README.md.
#
# Required environment:
#   HAYBARN_UNITTEST    path to the haybarn-unittest binary
#   VGI_SKLEARN_WORKER  worker LOCATION the .test files attach. One of:
#                         - a stdio command (e.g. the installed `vgi-sklearn`)
#                         - `launch:<command>` for the warm Unix-socket launcher
#                           transport (worker spawned once, reused per ATTACH —
#                           avoids per-attach Python/sklearn import startup)
#                         - an `http://` URL for a running HTTP server
# Optional:
#   STAGE               scratch dir for the preprocessed test tree (default: mktemp)
set -euo pipefail

: "${HAYBARN_UNITTEST:?path to the haybarn-unittest binary}"
: "${VGI_SKLEARN_WORKER:?worker LOCATION (stdio command or http:// URL)}"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
STAGE="${STAGE:-$(mktemp -d)}"

# Isolate the model registry so fit/predict tests don't touch a real ./models.
export SKLEARN_MODELS_DIR="${SKLEARN_MODELS_DIR:-$STAGE/models}"
mkdir -p "$SKLEARN_MODELS_DIR"

echo "Staging preprocessed tests into $STAGE ..."
mkdir -p "$STAGE/test/sql"
for f in "$REPO"/test/sql/*.test; do
  awk -f "$HERE/preprocess-require.awk" "$f" > "$STAGE/test/sql/$(basename "$f")"
done

cd "$STAGE"

# Warm the extension cache once: vgi from the signed community channel, httpfs
# from signed core. A miss here is only a warning — the per-test INSTALL/LOAD
# (injected by preprocess-require.awk) is what actually gates each file.
echo "Warming the extension cache (vgi from community, httpfs from core) ..."
mkdir -p "$STAGE/test"
cat > "$STAGE/test/_warm.test" <<'EOF'
# name: test/_warm.test
# group: [warm]
statement ok
INSTALL vgi FROM community;

statement ok
INSTALL httpfs FROM core;
EOF
"$HAYBARN_UNITTEST" "test/_warm.test" >/dev/null 2>&1 || echo "::warning::extension warm step did not fully succeed"
rm -f "$STAGE/test/_warm.test"

# Run the suite in one invocation, streaming the runner's native sqllogictest
# report. Any failed assertion exits non-zero and fails the job. TEST_PATTERN
# defaults to the whole suite; override it (e.g. one file) for a quick smoke.
TEST_PATTERN="${TEST_PATTERN:-test/sql/*}"
echo "Running suite (pattern: $TEST_PATTERN, worker: $VGI_SKLEARN_WORKER) ..."
"$HAYBARN_UNITTEST" "$TEST_PATTERN"
