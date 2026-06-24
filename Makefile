# scikit-learn VGI worker — dev, test, and deploy targets.
#
# Usage:
#   make test              # pytest unit/integration + SQL (stdio/http)
#   make test-stdio        # SQL tests with the worker as a subprocess
#   make test-http         # start a local HTTP server, run SQL tests, stop it
#   make test-docker-stdio # SQL tests against the built image, stdio transport
#   make test-docker-http  # SQL tests against the built image, HTTP transport
#   make test-cloud        # SQL tests against the deployed Fly.io service
#   make deploy            # deploy the published ghcr image to Fly.io
#
# The container image is built and published to ghcr.io by CI
# (.github/workflows/docker-publish.yml); `make deploy` just points Fly at it.

VGI_PYTHON_SRC ?= $(HOME)/Development/vgi-python
VGI_RPC_SRC    ?= $(HOME)/Development/vgi-rpc

VGI_BUILD_DIR  ?= $(HOME)/Development/vgi/build/release
TEST_RUNNER     = $(VGI_BUILD_DIR)/test/unittest
TEST_DIR        = .
TEST_PATTERN    = test/sql/*

# Worker paths (overridable)
WORKER_STDIO   ?= uv run --python 3.13 sklearn_worker.py
WORKER_HTTP    ?= http://localhost:8000
WORKER_CLOUD   ?= https://$(FLY_APP).fly.dev
HTTP_PORT      ?= 8000

# Fly.io config
FLY_APP        ?= vgi-sklearn

# Isolated model registry for local SQL tests (stdio/http workers inherit this).
TEST_MODELS_DIR ?= $(CURDIR)/.test-models

.PHONY: test pytest test-stdio test-http test-docker-build test-docker-stdio \
        test-docker-http test-cloud image smoke-test deploy venv

venv:
	uv venv --python 3.13
	uv pip install --python .venv \
		"vgi-python[http,oauth] @ $(VGI_PYTHON_SRC)" \
		"vgi-rpc[sentry] @ $(VGI_RPC_SRC)" \
		"scikit-learn>=1.5" numpy "skops>=0.11" pytest

pytest:
	.venv/bin/pytest tests/ --rootdir=. -o "addopts=" -q

test: pytest test-stdio test-http

test-stdio:
	rm -rf "$(TEST_MODELS_DIR)"
	# WORKER_STDIO runs against the local vgi-python checkout (PEP 723 sources).
	# Rebuild it in the script env first so uv doesn't reuse a stale cached build
	# (same version, changed source) that predates a local edit.
	uv run --reinstall-package vgi-python --python 3.13 sklearn_worker.py --help >/dev/null 2>&1 || true
	SKLEARN_MODELS_DIR="$(TEST_MODELS_DIR)" VGI_SKLEARN_WORKER="$(WORKER_STDIO)" \
		$(TEST_RUNNER) --test-dir "$(TEST_DIR)" "$(TEST_PATTERN)"

test-http:
	@if lsof -iTCP:$(HTTP_PORT) -sTCP:LISTEN -t >/dev/null 2>&1; then \
		echo "ERROR: port $(HTTP_PORT) is already in use" >&2; \
		echo "  Kill the existing process: kill $$(lsof -iTCP:$(HTTP_PORT) -sTCP:LISTEN -t)" >&2; \
		exit 1; \
	fi
	@rm -rf "$(TEST_MODELS_DIR)"
	@SKLEARN_MODELS_DIR="$(TEST_MODELS_DIR)" VGI_SIGNING_KEY=dev .venv/bin/python serve.py --port $(HTTP_PORT) & \
		SERVER_PID=$$!; \
		for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do \
			curl -fsS -o /dev/null "http://localhost:$(HTTP_PORT)/health" 2>/dev/null && break; \
			sleep 1; \
		done; \
		VGI_SKLEARN_WORKER="$(WORKER_HTTP)" $(TEST_RUNNER) --test-dir "$(TEST_DIR)" "$(TEST_PATTERN)"; \
		TEST_EXIT=$$?; \
		kill $$SERVER_PID 2>/dev/null; \
		wait $$SERVER_PID 2>/dev/null; \
		exit $$TEST_EXIT

test-cloud:
	VGI_SKLEARN_WORKER="$(WORKER_CLOUD)" $(TEST_RUNNER) --test-dir "$(TEST_DIR)" "$(TEST_PATTERN)"

# ---------------------------------------------------------------------------
# Container image
#
# CI builds and publishes the multi-arch image to ghcr.io
# (.github/workflows/docker-publish.yml). These targets build it locally for
# image testing and point Fly.io at the published image.
GIT_COMMIT     := $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
# Single source of truth: the __version__ literal the package advertises over VGI.
VERSION        := $(shell sed -nE 's/^__version__ = "([^"]+)".*/\1/p' vgi_sklearn/__init__.py)

GHCR_IMAGE     ?= ghcr.io/query-farm/vgi-sklearn
# Tag Fly.io pulls. Defaults to the released version; override for edge/sha tags.
TAG            ?= $(VERSION)

# Locally-built image for the image test targets below.
DOCKER_IMAGE     ?= vgi-sklearn:dev
DOCKER_STATE_VOL ?= vgi_sklearn_state_test

image:
	docker build --build-arg VERSION=$(VERSION) --build-arg GIT_COMMIT=$(GIT_COMMIT) \
		-t $(DOCKER_IMAGE) .

# Run the authoritative SQL suite against the built image, stdio transport: the
# extension spawns the container per ATTACH, sharing a throwaway named volume.
test-docker-stdio: image
	-docker volume rm $(DOCKER_STATE_VOL) >/dev/null 2>&1
	VGI_SKLEARN_WORKER="docker run -i --rm -v $(DOCKER_STATE_VOL):/data $(DOCKER_IMAGE) stdio" \
		$(TEST_RUNNER) --test-dir "$(TEST_DIR)" "$(TEST_PATTERN)"
	-docker volume rm $(DOCKER_STATE_VOL) >/dev/null 2>&1

# Same suite against the built image over HTTP: start one container, run, stop.
test-docker-http: image
	-docker volume rm $(DOCKER_STATE_VOL) >/dev/null 2>&1
	@CID=$$(docker run -d -p $(HTTP_PORT):8000 -v $(DOCKER_STATE_VOL):/data \
			-e VGI_SIGNING_KEY=dev $(DOCKER_IMAGE)); \
		trap "docker rm -f $$CID >/dev/null 2>&1; docker volume rm $(DOCKER_STATE_VOL) >/dev/null 2>&1" EXIT; \
		for i in $$(seq 1 30); do \
			curl -fsS -o /dev/null "http://localhost:$(HTTP_PORT)/health" 2>/dev/null && break; \
			sleep 1; \
		done; \
		VGI_SKLEARN_WORKER="$(WORKER_HTTP)" $(TEST_RUNNER) --test-dir "$(TEST_DIR)" "$(TEST_PATTERN)"

# Quick local probe of the built image's HTTP mode (CI does the full §4b suite).
smoke-test: image
	@CID=$$(docker run -d -e VGI_SIGNING_KEY=dev -p 18000:8000 $(DOCKER_IMAGE)); \
		trap "docker rm -f $$CID >/dev/null 2>&1" EXIT; \
		for i in 1 2 3 4 5 6 7 8 9 10; do \
			if curl -fsS -o /dev/null http://localhost:18000/health 2>/dev/null; then \
				echo "HTTP server responding"; exit 0; \
			fi; \
			sleep 1; \
		done; \
		echo "ERROR: container did not respond on /health within 10s" >&2; \
		docker logs $$CID >&2; \
		exit 1

# Deploy the published ghcr image to Fly.io (CI built + signed it on release).
deploy:
	fly deploy --image $(GHCR_IMAGE):$(TAG) --app $(FLY_APP)
