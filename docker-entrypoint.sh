#!/bin/sh
# Copyright 2026 Query Farm LLC - https://query.farm
#
# Dispatch the single vgi-sklearn image into one of its transports:
#   http   (default) the HTTP server on $PORT (Fly.io / local HTTP)
#   tcp              the native vgi-rpc protocol over TCP on $PORT_TCP (default
#                    8001), bound to 0.0.0.0 so a published host port reaches it.
#                    Used by the VGI extension's transparently-shared container.
#   stdio            a worker DuckDB spawns over stdio (on-host execution)
# Any other first argument is exec'd verbatim (escape hatch for debugging).
#
# Both modes share one /data volume (see the farm.query.vgi.volumes image label):
#   /data/models  -> SKLEARN_MODELS_DIR    (model registry)
#   /data/state   -> VGI_WORKER_SQLITE_PATH (shared BoundStorage, WAL SQLite)
set -e

# A freshly-mounted (empty) volume has no subdirs; create them so the non-root
# user can write the registry and state DB. Harmless when /data is unmounted.
mkdir -p "${SKLEARN_MODELS_DIR:-/data/models}" \
         "$(dirname "${VGI_WORKER_SQLITE_PATH:-/data/state/vgi_storage.db}")"

case "${1:-http}" in
  http)
    shift 2>/dev/null || true
    exec vgi-sklearn-http --host 0.0.0.0 --port "${PORT:-8000}" "$@"
    ;;
  tcp)
    shift 2>/dev/null || true
    # Bind 0.0.0.0 so docker's published-port forwarding reaches the listener;
    # remaining args (e.g. --idle-timeout) are forwarded to the worker.
    exec vgi-sklearn --tcp "0.0.0.0:${PORT_TCP:-8001}" "$@"
    ;;
  stdio)
    shift 2>/dev/null || true
    exec vgi-sklearn "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
