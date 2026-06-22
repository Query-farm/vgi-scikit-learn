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
sklearn_worker.py     entry point: builds the `sklearn` Catalog, SklearnWorker, main()
serve.py              HTTP entry point (injects --http into Worker.main())
vgi_sklearn/
  datasets.py         dataset table functions (toy, generators, california_housing)
  metrics.py          metric aggregates over (y_true, y_pred)
  table_metrics.py    confusion_matrix / silhouette_score (buffering, table input)
  transforms.py       unsupervised fit_transform (buffering)
  models.py           generic fit / predict / cross_val_predict + registry mgmt
  typed_models.py     generated fit_<estimator> functions with typed hyperparams
  registry.py         ModelStore + LocalDiskStore (S3/R2 seam) + model-BLOB pack/unpack
  buffering.py        shared sink/combine/serialize/matrix helpers (numeric validation)
  schema_utils.py     pa.Field comment helper, name sanitisation, NoArgs
tests/                pytest (in-process harness in tests/harness.py)
test/sql/*.test       DuckDB sqllogictest — the authoritative integration tests
```

To add functions: implement in the relevant `vgi_sklearn/*.py`, export a
`*_FUNCTIONS` list, and splice it into `_FUNCTIONS` in `sklearn_worker.py`.

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
  `models._fit_and_emit`. To add/adjust hyperparameters, edit `_HPARAMS`; the
  `test_typed_params_are_valid_for_estimator` test guards that every exposed
  param is real for its estimator. `max_depth := 0` maps to `None` (unlimited);
  mlp `hidden_units` maps to `hidden_layer_sizes=(n,)`.
- **predict aligns features by name** (reorder-safe, extra columns ignored);
  missing/non-numeric feature columns raise clear errors at bind.

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
9. **Local source skew:** local `vgi-rpc` (0.20.3) lags vgi-python's pin
   (`>=0.20.4`). Worked around with `override-dependencies` in the PEP 723
   headers, `--override` in `make venv`, and a `sed` in the Dockerfile. Bumping
   local `vgi-rpc` to ≥0.20.4 would let these be removed.
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

## Testing

```sh
make venv          # .venv with vgi + scikit-learn from local checkouts
make pytest        # in-process unit tests (fast; uses tests/harness.py)
make test-stdio    # SQL tests, worker as subprocess  (authoritative)
make test-http     # SQL tests against a local HTTP server
```

- **SQL tests are authoritative.** Unit tests call classmethods directly and
  can pass while the real RPC path is broken — that's exactly how the aggregate
  state-persistence bug (edge #1) slipped past pytest. Always run `test-stdio`.
- SQL tests need DuckDB's `unittest` runner built with the VGI extension at
  `$(VGI_BUILD_DIR)/test/unittest`.
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
`LocalDiskStore` (joblib pickle + JSON metadata, root from `SKLEARN_MODELS_DIR`,
default `./models`) is the only impl today; an `S3Store` for S3/R2 is the
planned next backend and drops in here without touching `models.py`. `predict`
warns via `duckdb_logs()` if the worker's scikit-learn version differs from the
one a model was fitted with.
