# Copyright 2026 Query Farm LLC - https://query.farm
#
# Single image that serves BOTH transports of the scikit-learn VGI worker:
#   docker run ... IMG            -> HTTP server on $PORT (default, Fly.io / local)
#   docker run -i ... IMG stdio   -> stdio worker DuckDB spawns on-host
# See docker-entrypoint.sh. Deps install from PyPI (vgi-python / vgi-rpc / sklearn
# are all published) — no vendored checkouts, no pin rewriting.
# syntax=docker/dockerfile:1
FROM python:3.13-slim

# Build metadata, wired from docker/metadata-action outputs in CI.
ARG VERSION=0.0.0
ARG GIT_COMMIT=unknown
ARG SOURCE_URL=https://github.com/Query-farm/vgi-scikit-learn

# Standard OCI labels + the VGI mount-discovery label. The VGI extension reads
# `farm.query.vgi.volumes` from the image config and injects the matching `-v`
# mount when it spawns the container, so the ATTACH LOCATION stays clean.
#   path    container mountpoint that must be persisted
#   name    suggested default volume name (the extension may override)
#   purpose state | scratch
#   shared  true => workers in one execution mount the SAME source (shared
#           registry + WAL-SQLite BoundStorage)
LABEL org.opencontainers.image.title="vgi-sklearn" \
      org.opencontainers.image.description="scikit-learn as a VGI worker for DuckDB/SQL (stdio + HTTP)" \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.licenses="MIT" \
      farm.query.vgi.volumes='[{"path":"/data","name":"vgi_sklearn_state","purpose":"state","shared":true}]' \
      farm.query.vgi.transports='["tcp","http"]'

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Both state mechanisms default under one /data volume (see the label above).
    SKLEARN_MODELS_DIR=/data/models \
    VGI_WORKER_SQLITE_PATH=/data/state/vgi_storage.db \
    # Build provenance only (Sentry release / diagnostics) — the version the
    # worker advertises over VGI comes from the installed package, not this.
    VGI_SKLEARN_GIT_COMMIT=${GIT_COMMIT} \
    SENTRY_RELEASE=${GIT_COMMIT}

WORKDIR /app

# curl backs the HEALTHCHECK below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install the worker + HTTP-serving extras (Sentry/OAuth/authlib) from PyPI. The
# version is read from vgi_sklearn/__init__.py by hatchling, so the wheel and the
# advertised implementation_version match the release with no .git present.
COPY pyproject.toml README.md LICENSE ./
COPY vgi_sklearn ./vgi_sklearn
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN pip install '.[serve]' \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

# Run unprivileged. No `VOLUME /data` on purpose: a VOLUME would make every
# `docker run` create an anonymous volume (deleted by --rm, and masking the
# worker's "is /data mounted?" check). The orchestrator/extension mounts /data
# explicitly. We pre-create + own the dirs so an unmounted run still works and a
# named volume inherits writable ownership.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /data/models /data/state \
    && chown -R app:app /data
USER app

EXPOSE 8000

# Readiness probe for HTTP mode (mirrors the Fly.io /health check). Inert for a
# short-lived stdio container, which has no HTTP server.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-8000}/health" || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["http"]
