# scikit-learn VGI worker — dev, test, and deploy targets.
#
# Usage:
#   make vendor-sync  # re-sync vendor/ from ~/Development/vgi-{python,rpc}/
#   make test         # pytest unit/integration + SQL (stdio/http)
#   make test-stdio   # SQL tests with the worker as a subprocess
#   make test-http    # start a local HTTP server, run SQL tests, stop it
#   make test-cloud   # SQL tests against the deployed Fly.io service
#   make deploy       # build locally, push, deploy to Fly.io

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

.PHONY: test pytest test-stdio test-http test-cloud build push smoke-test deploy vendor-sync venv

venv:
	uv venv --python 3.13
	uv pip install --python .venv \
		"vgi-python[http,oauth] @ $(VGI_PYTHON_SRC)" \
		"vgi-rpc[sentry] @ $(VGI_RPC_SRC)" \
		"scikit-learn>=1.5" numpy "skops>=0.11" pytest

vendor-sync:
	@for src in "$(VGI_PYTHON_SRC)" "$(VGI_RPC_SRC)"; do \
		if [ ! -d "$$src" ]; then echo "ERROR: source missing: $$src" >&2; exit 1; fi; \
	done
	rm -rf vendor/vgi-python vendor/vgi-rpc
	mkdir -p vendor/vgi-python vendor/vgi-rpc
	rsync -a --exclude='__pycache__' --exclude='.mypy_cache' --exclude='.ruff_cache' \
		--exclude='.pytest_cache' "$(VGI_PYTHON_SRC)/vgi/" vendor/vgi-python/vgi/
	cp "$(VGI_PYTHON_SRC)/pyproject.toml" "$(VGI_PYTHON_SRC)/README.md" vendor/vgi-python/
	@cp "$(VGI_PYTHON_SRC)/LICENSE.md" vendor/vgi-python/ 2>/dev/null || \
		cp "$(VGI_PYTHON_SRC)/LICENSE" vendor/vgi-python/
	rsync -a --exclude='__pycache__' --exclude='.mypy_cache' --exclude='.ruff_cache' \
		--exclude='.pytest_cache' "$(VGI_RPC_SRC)/vgi_rpc/" vendor/vgi-rpc/vgi_rpc/
	cp "$(VGI_RPC_SRC)/pyproject.toml" "$(VGI_RPC_SRC)/README.md" vendor/vgi-rpc/
	@cp "$(VGI_RPC_SRC)/LICENSE.md" vendor/vgi-rpc/ 2>/dev/null || \
		cp "$(VGI_RPC_SRC)/LICENSE" vendor/vgi-rpc/
	@echo "vendor/ synced from $(VGI_PYTHON_SRC) and $(VGI_RPC_SRC)"

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

GIT_COMMIT     := $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
IMAGE_TAG      := $(GIT_COMMIT)-$(shell date +%Y%m%d%H%M%S)
IMAGE          := registry.fly.io/$(FLY_APP):$(IMAGE_TAG)

build:
	docker build --platform linux/amd64 --build-arg GIT_COMMIT=$(GIT_COMMIT) -t $(IMAGE) .

smoke-test: build
	@echo "Smoke-testing $(IMAGE)..."
	@docker run --rm --platform linux/amd64 -e VGI_SIGNING_KEY=dev $(IMAGE) \
		python -c "from sklearn_worker import SklearnWorker; import serve; print('imports OK')"
	@CID=$$(docker run -d --platform linux/amd64 -e VGI_SIGNING_KEY=dev -p 18000:8000 $(IMAGE)); \
		trap "docker rm -f $$CID >/dev/null" EXIT; \
		for i in 1 2 3 4 5 6 7 8 9 10; do \
			if curl -fsS -o /dev/null -w "%{http_code}\n" http://localhost:18000/health 2>/dev/null | grep -qE '^(200|401|403|404)$$'; then \
				echo "HTTP server responding"; exit 0; \
			fi; \
			sleep 1; \
		done; \
		echo "ERROR: container did not respond on /health within 10s" >&2; \
		docker logs $$CID >&2; \
		exit 1

push: smoke-test
	fly auth docker
	docker push $(IMAGE)

deploy: push
	fly deploy --image $(IMAGE) --app $(FLY_APP)
