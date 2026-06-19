FROM python:3.13-slim AS builder

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY vendor/vgi-rpc /app/vendor/vgi-rpc
COPY vendor/vgi-python /app/vendor/vgi-python

# Normalize the vgi-rpc dependency: strip any local file:// pin and relax the
# lower bound so the vendored vgi-rpc checkout satisfies it (the local checkout
# may lag vgi-python's pinned version).
RUN sed -i -E 's|"vgi-rpc @ file://[^"]*"|"vgi-rpc"|; s|"vgi-rpc>=[0-9.]+"|"vgi-rpc>=0.20.3"|' \
        /app/vendor/vgi-python/pyproject.toml \
    && pip wheel --no-deps --wheel-dir /wheels "/app/vendor/vgi-rpc" \
    && pip wheel --no-deps --wheel-dir /wheels "/app/vendor/vgi-python"

FROM python:3.13-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY --from=builder /wheels /wheels

# Install vendored wheels (--no-deps) with their extras' actual packages, then
# scikit-learn (pulls numpy/scipy/joblib/threadpoolctl).
RUN VGI_RPC_WHL=$(ls /wheels/vgi_rpc-*.whl) \
    && VGI_WHL=$(ls /wheels/vgi_python-*.whl) \
    && pip install --no-cache-dir "${VGI_RPC_WHL}[http,oauth,sentry]" "${VGI_WHL}" \
    && pip install --no-cache-dir authlib "scikit-learn>=1.5" numpy \
    && pip uninstall -y pip \
    && rm -rf /wheels

COPY vgi_sklearn /app/vgi_sklearn
COPY sklearn_worker.py /app/sklearn_worker.py
COPY serve.py /app/serve.py

ARG GIT_COMMIT=unknown
ENV VGI_SKLEARN_GIT_COMMIT=${GIT_COMMIT}
ENV SENTRY_RELEASE=${GIT_COMMIT}

# Where the local-disk model registry persists (mount a Fly volume here in prod).
ENV SKLEARN_MODELS_DIR=/data/models

EXPOSE 8000
CMD ["sh", "-c", "python /app/serve.py --host 0.0.0.0 --port ${PORT:-8000}"]
