# CLAUDE.md — vgi-sklearn

Contributor/agent notes for this repo. User-facing docs live in `README.md`;
this file is the "how it's built and where the sharp edges are" companion.

## What this is

A [VGI](https://github.com/query-farm/vgi-python) worker exposing scikit-learn
to DuckDB/SQL. `sklearn_worker.py` assembles every function into one `sklearn`
catalog (single `main` schema) and runs it over stdio (local) or HTTP (Fly.io).
Built on the local `~/Development/vgi-python` + `~/Development/vgi-rpc`
checkouts; modeled on `~/Development/vgi-trains-python-fly`.

## Layout

```
vgi_sklearn/
  worker.py           builds the `sklearn` Catalog + SklearnWorker; main()/main_http() entry points
  datasets.py         dataset table functions (toy, generators, california_housing)
  metrics.py          metric aggregates over (y_true, y_pred)
  table_metrics.py    confusion_matrix / silhouette_score (buffering, table input)
  transforms.py       unsupervised fit_transform + ordinal/one_hot encoders (buffering)
  features.py         categorical (string) detection + auto one-hot Pipeline wrapping
  text.py             count/tfidf vectorizers over a text column (long-format output)
  feature_selection.py select_k_best / variance_threshold (per-feature scores + selected flag)
  stored_transforms.py fit_transformer / apply_transform (persisted, reusable transformers)
  models.py           fit / predict / cross_val_predict / cross_val_score / permutation_importance + registry mgmt
  typed_models.py     generated fit_<estimator> functions with typed hyperparams
  search.py           grid_search — discriminated-union (sparse) hyperparameter search
  grouped.py          per-group modeling: fit_model (aggregate) + predict_* (scalars)
  registry.py         ModelStore + LocalDiskStore (S3/R2 seam) + model-BLOB pack/unpack
  buffering.py        shared sink/combine/serialize/matrix helpers (numeric validation)
  schema_utils.py     pa.Field comment helper, name sanitisation, NoArgs
sklearn_worker.py     repo-root stdio shim over vgi_sklearn.worker (uv run / Fly / tests)
serve.py              repo-root HTTP shim over vgi_sklearn.worker (Fly / tests)
tests/                pytest (in-process harness in tests/harness.py)
test/sql/*.test       DuckDB sqllogictest — the authoritative integration tests
```

To add functions: implement in the relevant `vgi_sklearn/*.py`, export a
`*_FUNCTIONS` list, and splice it into `_FUNCTIONS` in `vgi_sklearn/worker.py`.

**Entry points / packaging.** Console scripts (`vgi-sklearn`,
`vgi-sklearn-http`) point at `vgi_sklearn.worker:main` / `:main_http` — *inside
the package*, so they ship in the wheel. The repo-root `sklearn_worker.py` /
`serve.py` are thin shims for `uv run` / the Fly container / `import` in tests;
they are deliberately NOT in the wheel (only the `vgi_sklearn` package is). Don't
point entry points back at the root modules — that was a packaging bug (broken
console scripts on `pip install`). PyPI publish is `publish.yml` (GitHub Release
→ CI → `uv build && uv publish`); bump `version` in `pyproject.toml` before
tagging.

## Which VGI primitive for which job

| Need | Primitive | Example here |
| --- | --- | --- |
| Emit rows, no input | `TableFunctionGenerator` (`@bind_fixed_schema` / `@init_single_worker`, or custom `on_bind` for schema-from-args) | `datasets.py` |
| Scalar-per-group over columns | `AggregateFunction[State]` | `metrics.py` |
| `fit_transform` / `fit` (needs whole input) | `TableBufferingFunction` via `buffering.SinkBuffer` | `transforms.py`, `models.FitModel` |
| Score a stream with an already-fit model | `TableInOutGenerator` | `models.PredictModel` |

Conventions for transform/fit/predict: input relation is X via a `(SELECT ...)`
subquery (Arg(0)); name `target` (features = the rest, must be numeric) and an
optional `id` passthrough; generic `fit` takes hyperparameters as a JSON-string
arg, while `fit_<estimator>` exposes them as typed named args.

## Models: registry + BLOB + typed functions

- **fit always returns a `model` BLOB** (estimator + metadata packed by
  `registry.pack_model`) and persists to the registry only if `model_name` is
  given (so `model_name` is optional). `predict` takes **either** `model_name :=`
  or `model :=` (a BLOB); `registry.unpack_meta` reads metadata at bind,
  `unpack_model` loads the estimator at process.
- **Typed `fit_<estimator>` functions** are generated in `typed_models.py` from
  the `_HPARAMS` spec via `types.new_class(name, (SinkBuffer[args, DrainState],),
  ...)` — plain `type()` can't resolve the subscripted-generic base. Each shares
  `models._fit_and_emit`. **To add an estimator, add it to BOTH `_ESTIMATORS`
  (models.py) and `_HPARAMS` (typed_models.py)** — `test_every_estimator_has_a_spec`
  enforces identical keys, and one spec gives you the generic `fit`, the typed
  `fit_<estimator>`, grouped `fit_model`, and a `grid_search` union member at once.
  `test_typed_params_are_valid_for_estimator` guards that every exposed param is
  real for its estimator (hparam types limited to int/float/str/bool — the
  grid_search union only encodes those). `max_depth := 0` maps to `None`
  (unlimited); mlp `hidden_units` maps to `hidden_layer_sizes=(n,)`.
- **predict aligns features by name** (reorder-safe, extra columns ignored);
  missing feature columns raise clear errors at bind.
- **Categorical (string) features auto-encode** (`features.py`). At fit, string
  columns are detected (`categorical_mask`), the estimator is wrapped in a
  `Pipeline(ColumnTransformer(OneHotEncoder(handle_unknown='ignore'),
  passthrough), est)` (`wrap_estimator`), and the per-feature `categorical` mask
  is stored in `ModelMetadata`. Because the whole pipeline is the saved model,
  `predict` replays the encoding — it just rebuilds `X` with the same column
  dtypes (`build_x`: cat cells str, numeric/bool cells float). This is uniform
  across `fit`, `fit_<estimator>`, `grouped.fit_model`, and `grid_search` (which
  prefixes grid keys `est__<param>` when wrapped, see `prefix_grid`). `n_features`
  is the *original* feature count, not the one-hot width. The standalone
  `ordinal_encoder` / `one_hot_encoder` transforms (transforms.py) expose the
  encoding as data — one_hot uses long format `(id, feature, category, value)` to
  dodge the data-dependent-width limit (edge #6).
- **`grid_search` (search.py) is a discriminated union.** The `estimator` arg is
  a sparse Arrow union (`_GRID_UNION`, one member per estimator built from
  `_HPARAMS`, each field a `list<scalar>`); SQL calls it as
  `union_value(<estimator> := {param: [values]})`. The worker reads it as a
  `vgi.TaggedUnion` (`.tag` = estimator, `.value` = grid dict); omitted (NULL)
  hyperparameters stay at defaults. Returns the CV leaderboard (one row per
  combo) with the refit best model BLOB on the single `best_index_` row — grab
  it with `WHERE model IS NOT NULL` (rank 1 can tie). **Dependency/gating:** this
  needs a vgi-python whose argument decoder preserves union tags (`TaggedUnion`,
  > 0.8.2). `worker.py` imports `search` under try/except so older vgi-python
  just omits `grid_search`; the SQL test is gated `require-env
  VGI_SKLEARN_GRID_SEARCH` (set only in `make test-stdio`, which runs the local
  checkout) and `tests/test_search.py` skips when `TaggedUnion` is absent — so CI
  on the released PyPI vgi-python stays green. When a vgi-python with the fix is
  released, bump the pin and drop the gating. (Dense unions are still unsupported
  by the C++ extension; `union_value` produces sparse, which works.)
- **Per-group modeling (grouped.py) is an aggregate + scalars** — table functions
  can't take correlated/lateral args, so column-driven dispatch is impossible;
  `fit_model` is an `AggregateFunction` (`GROUP BY` partitions for free, returns
  one `STRUCT(model BLOB, …)` per group) and `predict_one`/`predict_class_one`
  (string labels)/`predict_proba_one` (`DOUBLE[]`) are `ScalarFunction`s taking a
  per-row model BLOB + feature `STRUCT`. Works on released PyPI vgi-python (no
  gating). Sharp edges hit here:
    - Any-typed input column = `Annotated[pa.Array, Param(...)]` (generic
      `pa.Array` → AnyArrow); fixed struct = `Param(arrow_type=pa.struct(...))`.
      The target is AnyArrow so int/float/**string** labels bind with no cast.
    - **Never name an aggregate `update()` parameter `params`** — the framework
      injects `ProcessParams` into any arg named `params`, clobbering your value.
      Use `hyperparams`.
    - **Aggregate const params can't be optional** — DuckDB requires all declared
      params, so a `ConstParam` default won't bind when omitted (`hyperparams` is
      required; pass `'{}'`).
    - **VARIANT can't be a result type**: DuckDB itself can't move VARIANT over
      Arrow (`Unsupported Arrow type VARIANT`), so the polymorphic prediction is
      split into typed scalars (`predict_one`/`predict_class_one`/`_proba_one`).
    - `_load` is `lru_cache`d on BLOB bytes → a joined per-group model is
      deserialized once, not per row.
- **Serialization is skops, not pickle** (`registry._skops_dumps/_skops_loads`,
  `.skops` files + skops BLOBs). Loading reconstructs only known types and we
  restrict trust to the `sklearn`/`numpy`/`scipy` namespaces (`_TRUSTED_PREFIXES`)
  — anything else raises `UntrustedModelError`. No arbitrary code execution like
  pickle. Still scikit-learn-version-coupled (the version-mismatch warning
  remains); ONNX would be the only fully version-independent option, at coverage
  cost. Documented in the README security section.

## Sharp edges (learned the hard way — read before debugging)

1. **Aggregate state: reassign, don't mutate.** `update()` must do
   `states[gid] = NewState(...)`. The framework persists only groups you
   *assigned* this batch (plus groups carried from a prior batch); an in-place
   mutation of a group first seen in the batch is silently dropped → every
   result NULL. Single-group/whole-table aggregates always hit this. See
   `buffering`-free `metrics._BufferedMetric.update` for the correct pattern.
2. **`pa.Float64Array` does not exist** — the class is `pa.DoubleArray`. A bad
   `Param` type hint does NOT error; the framework warns and registers the
   function with **zero input columns**, so it binds but receives nothing.
   Watch for `UserWarning: ... type hints could not be resolved`.
3. **Table argument syntax is `(SELECT ...)`, not `TABLE(...)`.**
4. **`Arg(0)` = positional, `Arg("name")` = named-only.** Single required args
   that should be callable positionally (e.g. `model_info('m')`) use `Arg(0)`.
   The table input is always `Arg(0)`.
5. **Buffering / in-out state classes must extend `ArrowSerializableDataclass`**
   (e.g. `buffering.DrainState`). The framework raises a clear TypeError if not.
6. **Output schema is fixed at bind.** Fine when width comes from args
   (PCA `n_components`) or mirrors input (scalers). For data-dependent width
   (e.g. OneHotEncoder categories) emit long format instead.
7. **HTTP entry point:** current vgi-python has **no `main_http`**. Serve HTTP
   via `Worker.main()` with `--http`; `serve.py` injects that flag.
8. **Distribution rename:** the package is dist `vgi-python` (import `vgi`).
   PEP 723 headers use `vgi-python = { path = ... }`; the Docker wheel glob is
   `vgi_python-*.whl`. The older `vgi`-based assumptions from vgi-trains break.
9. **Local source skew (local-checkout dev only):** the packaged/CI/PyPI path
   is clean — `vgi-python` 0.8.1 and `vgi-rpc` 0.20.4 are both on PyPI, so
   `pyproject.toml` / `uv.lock` resolve without any override. The skew only bites
   when building against the *local* `~/Development/vgi-rpc` checkout (0.20.3),
   which lags vgi-python's `>=0.20.4` pin: worked around with
   `override-dependencies` in the PEP 723 headers, `--override` in `make venv`,
   and a `sed` in the Dockerfile. Bumping the local `vgi-rpc` checkout to ≥0.20.4
   would let those be removed.
10. **A table function gets at most ONE subquery parameter.** That slot is the
    table input `(SELECT ...)`. You cannot also pass `model := (SELECT model
    FROM ...)` — DuckDB errors "Table function can have at most one subquery
    parameter". To pass a runtime BLOB/scalar, stash it in a session variable
    and read it back as a scalar: `SET VARIABLE m = (SELECT ...)` then
    `predict(..., model := getvariable('m'))`. `:=` and `=>` are equivalent
    named-arg syntaxes; docs use `:=`.
11. **Generating VGI function classes dynamically:** use
    `types.new_class(name, (SinkBuffer[Args, State],), {}, lambda ns:
    ns.update(namespace))`. Plain `type(name, (Base[...],), ns)` raises
    "type() doesn't support MRO entry resolution" for subscripted-generic
    bases. Build the args dataclass with `dataclasses.make_dataclass` using
    `Annotated[t, Arg(...)]` field types; set `FunctionArguments` in the
    namespace explicitly.

## Packaging & CI

The repo is an installable package (`pyproject.toml`, hatchling, `uv.lock`):
`uv sync` resolves PyPI `vgi-python[http]` + scikit-learn + skops and exposes
the `vgi-sklearn` (stdio) and `vgi-sklearn-http` console scripts. Lint/format is
ruff (`uvx ruff check .` / `ruff format`); config in `pyproject.toml`. GitHub
Actions (`.github/workflows/ci.yml` + `ci/`) runs the unit + SQL suites on
Linux/macOS/Windows against the **signed community `vgi` extension** via a
prebuilt `haybarn-unittest` — no C++ build (mirrors `vgi-easter`; see
`ci/README.md`). Keep PyPI deps in `pyproject.toml` in sync with the PEP 723
headers and the Dockerfile `pip install` line when adding a dependency.

## Testing

```sh
uv sync && uv run pytest tests/ -q   # unit tests against PyPI deps (CI's unit job)
make venv && make pytest             # unit tests against local vgi checkouts
make test-stdio                      # SQL tests, worker as subprocess (authoritative)
make test-http                       # SQL tests against a local HTTP server
```

- **SQL tests are authoritative.** Unit tests call classmethods directly and
  can pass while the real RPC path is broken — that's exactly how the aggregate
  state-persistence bug (edge #1) slipped past pytest. Always run the SQL suite.
- SQL tests need a sqllogictest runner with the VGI extension: either a DuckDB
  `unittest` built with vgi at `$(VGI_BUILD_DIR)/test/unittest`, or (what CI
  uses) a standalone `haybarn-unittest` + `INSTALL vgi FROM community`.
- For fast local probing with *real* error messages, `pip install haybarn` and
  drive it from Python (`con.execute("INSTALL vgi FROM community; LOAD vgi")`,
  `ATTACH ... LOCATION 'uv run sklearn_worker.py'`) — far better than reading
  sqllogictest diffs while iterating.
- `make test-stdio` / `test-http` point `SKLEARN_MODELS_DIR` at an isolated
  `.test-models/` so the registry tests don't pollute `./models`.

## Deployment (Fly.io)

```sh
make vendor-sync   # rsync vgi-python/vgi-rpc into vendor/ for the Docker build
make deploy        # build (linux/amd64) -> smoke-test -> push -> fly deploy
fly volumes create sklearn_models --size 1 --region iad   # one-time, registry
```

`fly.toml` bumps VM memory to 1gb (scikit-learn/scipy are heavy) and mounts a
volume at `/data` for the model registry (`SKLEARN_MODELS_DIR=/data/models`).
The Docker smoke test verifies imports + `/health`.

## Model registry

`registry.get_store()` is the single seam selecting the backend.
`LocalDiskStore` (skops `.skops` artifact + JSON metadata sidecar, root from
`SKLEARN_MODELS_DIR`, default `./models`) is the only impl today; an `S3Store`
for S3/R2 is the planned next backend and drops in here without touching
`models.py`. `predict` warns via `duckdb_logs()` if the worker's scikit-learn
version differs from the one a model was fitted with.

**Fitted transformers** (`fit_transformer`/`apply_transform`, stored_transforms.py)
use a *parallel* `registry.get_transformer_store()` seam rooted at
`<SKLEARN_MODELS_DIR>/transformers/` — a separate subdir so `list_models` and
`list_transformers` never see each other's `.skops`/`.json`. Same BLOB layout as
models (`pack_transformer`/`unpack_transformer*` reuse `_split_blob`/`_skops_*`),
but `TransformerMetadata` records the input `feature_names` (apply aligns by name)
plus the fit-time `output_names` (mirror the inputs, or `component_1..k` for
pca/svd) and `output_int` (kbins). The output schema is therefore fixed at bind
from stored metadata, dodging edge #6 even though `transform()` width is
data-derived.
