<p align="center">
  <img src="https://raw.githubusercontent.com/Query-farm/vgi-scikit-learn/main/assets/vgi-logo.png" alt="Vector Gateway Interface" height="104">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/Query-farm/vgi-scikit-learn/main/assets/scikit-learn-logo.png" alt="scikit-learn" height="80">
</p>

# vgi-sklearn

[![CI](https://github.com/Query-farm/vgi-scikit-learn/actions/workflows/ci.yml/badge.svg)](https://github.com/Query-farm/vgi-scikit-learn/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/vgi-sklearn.svg)](https://pypi.org/project/vgi-sklearn/)
[![Python](https://img.shields.io/pypi/pyversions/vgi-sklearn.svg)](https://pypi.org/project/vgi-sklearn/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Train and run machine-learning models in pure SQL.** `vgi-sklearn` exposes
[scikit-learn](https://scikit-learn.org/) to DuckDB as ordinary SQL functions —
so you can scale features, cluster, detect outliers, train a classifier, and
score new rows without leaving your query. No Python notebook, no CSV export, no
glue code: your table goes in, predictions come out, and the model can live in a
DuckDB column.

```sql
-- one-time: load the extension and attach the worker
INSTALL vgi FROM community; LOAD vgi;
ATTACH 'sklearn' (TYPE vgi, LOCATION 'vgi-sklearn');   -- 'uv run sklearn_worker.py' from a checkout

-- train a model on a table and score rows, all in SQL
CREATE TABLE flowers AS SELECT * FROM sklearn.iris();

CREATE TABLE model AS
  SELECT model FROM sklearn.fit_random_forest_classifier(
    (SELECT sample_id, sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm, target FROM flowers),
    target := 'target', id := 'sample_id', n_estimators := 200);

SET VARIABLE m = (SELECT model FROM model);
SELECT sample_id, prediction
FROM sklearn.predict((SELECT * EXCLUDE (target) FROM flowers), model := getvariable('m'), id := 'sample_id')
LIMIT 5;
```

That's the whole loop: `fit_…` returns a trained model, `predict` scores a table
through it. Everything below is variations on that theme.

---

## How it works (read this first — it's quick)

Every modeling function follows the same SQL-friendly contract:

- **Your input table is the feature matrix.** You pass it as a subquery —
  `sklearn.pca((SELECT ...), …)`. (DuckDB allows a table function only one
  subquery argument, so the data goes there; everything else is a named arg.)
- **Named arguments use `:=`** — `n_clusters := 3`, `target := 'label'`.
- **`target`** names your label column (training only). **`id`** names an
  identifier column that is **never used as a feature**. **Every other column is
  treated as a numeric feature** — so `SELECT` only the columns you want as
  features (non-numeric columns raise a clear error).
- **What `id` does depends on the function.** Functions that emit one row per
  input row — `predict`, the transforms, `cross_val_predict` — copy your `id`
  onto each output row, so a plain `JOIN ... USING (id)` reattaches results to
  the source. **`fit` returns a single summary row** (it doesn't echo your data),
  so there `id` does just one thing: keep that identifier column out of the
  feature matrix. Either way, pass `id` so the model never trains on a
  meaningless key like `customer_id`.
- **Features are matched by name, not position.** A model trained on
  `age, income` scores correctly whether you feed it `income, age` or a table
  with extra columns — it pulls its own features by name and errors if one is
  missing.

If you know `GROUP BY` and subqueries, you already know how to use this.

---

## Recipes

### Train a model

Each estimator has its own `fit_<estimator>` function that exposes its real
hyperparameters as **typed, named SQL arguments** (they autocomplete and are
type-checked):

```sql
-- a classifier (your own table: customers you've labeled as churned 0/1)
CREATE TABLE churn AS
  SELECT sample_id AS customer_id, sepal_length_cm AS tenure, sepal_width_cm AS monthly_spend,
         petal_length_cm AS support_tickets, (target = 0)::INT AS churned
  FROM sklearn.iris();

SELECT estimator, task, n_samples, n_features, train_score
FROM sklearn.fit_gradient_boosting_classifier(
  (SELECT customer_id, tenure, monthly_spend, support_tickets, churned FROM churn),
  model_name := 'churn_gb',          -- store it in the registry under this name
  target := 'churned',               -- the label column
  id := 'customer_id',               -- identifier: excluded from features (not trained on)
  n_estimators := 300,
  learning_rate := 0.05,
  max_depth := 3);
```

`fit` returns one summary row describing the trained model (and the model
itself, as a BLOB) — `id` here isn't echoed back, it just tells the worker that
`customer_id` is an identifier, not a feature to learn from.

```sql
-- a regressor
SELECT estimator, task, train_score
FROM sklearn.fit_random_forest_regressor(
  (SELECT * FROM sklearn.diabetes()),
  model_name := 'diabetes_rf', target := 'target', id := 'sample_id',
  n_estimators := 400, max_depth := 0);   -- max_depth := 0 means "no limit"
```

Available estimators (each is `sklearn.fit_<name>`):

| Family | Functions | Common typed args |
| --- | --- | --- |
| Linear | `logistic_regression`, `linear_regression`, `ridge`, `lasso` | `C`, `alpha`, `max_iter`, `fit_intercept`, `penalty`, `solver` |
| Trees / ensembles | `decision_tree_classifier`/`_regressor`, `random_forest_classifier`/`_regressor`, `gradient_boosting_classifier`/`_regressor`, `hist_gradient_boosting_classifier`/`_regressor` | `n_estimators`, `max_depth`, `learning_rate`, `min_samples_split`, `subsample`, `random_state` |
| SVM | `svc`, `svr` | `C`, `kernel`, `gamma`, `degree`, `epsilon` |
| Neighbors | `knn_classifier`, `knn_regressor` | `n_neighbors`, `weights`, `p` |
| Neural net | `mlp_classifier`, `mlp_regressor` | `hidden_units`, `alpha`, `max_iter`, `learning_rate_init` |
| Naive Bayes | `gaussian_nb` | `var_smoothing` |

> Need a hyperparameter that isn't exposed as a typed argument? The generic
> `sklearn.fit((SELECT ...), estimator := 'ridge', target := 'y', params := '{"alpha": 0.3, "solver": "svd"}')`
> accepts any scikit-learn parameter as a JSON object.

Every `fit_…` call **returns the trained model as a `model` BLOB column** *and*,
when you pass `model_name`, saves it to the registry. So you choose where the
model lives (see [Where models live](#where-models-live)).

### Score new data

`predict` streams a table through a stored model. It carries your `id` through
and appends `prediction`:

```sql
-- from a registry model by name
SELECT customer_id, prediction
FROM sklearn.predict(
  (SELECT customer_id, tenure, monthly_spend, support_tickets FROM churn),
  model_name := 'churn_gb', id := 'customer_id');
```

Add `with_proba := true` to also get one probability column per class
(`proba_0`, `proba_1`, …):

```sql
SELECT customer_id, prediction, proba_1 AS churn_probability
FROM sklearn.predict(
  (SELECT customer_id, tenure, monthly_spend, support_tickets FROM churn),
  model_name := 'churn_gb', id := 'customer_id', with_proba := true)
WHERE proba_1 > 0.5;
```

### Evaluate a model honestly (no leakage, nothing stored)

`cross_val_predict` returns out-of-fold predictions — each row scored by a model
that didn't see it — which you then compare to the truth with the metric
functions:

```sql
SELECT sklearn.accuracy_score(c.churned, p.prediction) AS cv_accuracy
FROM sklearn.cross_val_predict(
       (SELECT customer_id, tenure, monthly_spend, support_tickets, churned FROM churn),
       estimator := 'gradient_boosting_classifier', target := 'churned', id := 'customer_id', cv := 5) p
JOIN churn c ON c.customer_id = p.customer_id;
```

### Score predictions you already have

The metric functions are plain aggregates over two columns — point them at any
table of `(actual, predicted)` and group however you like:

```sql
-- one score, or one per segment/model with GROUP BY
SELECT sklearn.r2_score(actual, predicted) AS r2,
       sklearn.mean_absolute_error(actual, predicted) AS mae
FROM my_predictions;

-- a full confusion matrix in long form
SELECT * FROM sklearn.confusion_matrix(
  (SELECT label AS y, predicted AS yhat FROM my_predictions),
  actual := 'y', predicted := 'yhat');
```

### Prepare / transform features

All transforms take your table as a subquery, carry `id` through, and run
`fit_transform` over the whole input:

```sql
-- standardize features (zero mean, unit variance)
SELECT * FROM sklearn.standard_scaler(
  (SELECT customer_id, tenure, monthly_spend, support_tickets FROM churn), id := 'customer_id');

-- reduce to 2 components for plotting
SELECT * FROM sklearn.pca(
  (SELECT customer_id, tenure, monthly_spend, support_tickets FROM churn),
  id := 'customer_id', n_components := 2);

-- fill missing values before modeling
SELECT * FROM sklearn.simple_imputer((SELECT ...), id := 'id', strategy := 'median');
```

Transforms compose — pipe one into the next as nested subqueries (scale, then
cluster).

### Cluster & find outliers

```sql
-- k-means: appends a `cluster` label per row
SELECT customer_id, cluster
FROM sklearn.kmeans(
  (SELECT customer_id, tenure, monthly_spend, support_tickets FROM churn),
  id := 'customer_id', n_clusters := 4);

-- isolation forest: appends `anomaly_score` and `is_outlier`
SELECT customer_id, anomaly_score
FROM sklearn.isolation_forest(
  (SELECT customer_id, tenure, monthly_spend, support_tickets FROM churn),
  id := 'customer_id', contamination := 0.05)
WHERE is_outlier = 1;
```

### Get sample data to play with

scikit-learn's bundled datasets are table functions — handy for trying things or
building demos:

```sql
SELECT * FROM sklearn.iris();
SELECT * FROM sklearn.make_blobs(n_samples := 300, centers := 4);   -- synthetic clusters
```

---

## Function reference

**Datasets** (no input): `iris`, `wine`, `digits`, `breast_cancer`, `diabetes`,
`california_housing`, and generators `make_classification`, `make_regression`,
`make_blobs`, `make_moons`, `make_circles`.

**Models:** `fit_<estimator>` (typed, see the table above), generic `fit`
(escape hatch with JSON `params`), `predict`, `cross_val_predict`,
`list_models`, `model_info`, `drop_model`.

**Transforms** (table in, `id` passthrough): `standard_scaler`, `minmax_scaler`,
`robust_scaler`, `normalizer`, `simple_imputer`, `pca`, `truncated_svd`,
`kmeans`, `dbscan`, `isolation_forest`.

**Metric aggregates** over `(y_true, y_pred)`:
- Regression — `mean_squared_error`, `root_mean_squared_error`,
  `mean_absolute_error`, `r2_score`, `explained_variance_score`,
  `mean_absolute_percentage_error`, `max_error`, `median_absolute_error`
- Classification — `accuracy_score`, `precision_score`, `recall_score`,
  `f1_score`, `balanced_accuracy_score`, `matthews_corrcoef`, `cohen_kappa_score`
- Probability / ranking — `roc_auc_score`, `average_precision_score`, `log_loss`
- Clustering — `adjusted_rand_score`, `normalized_mutual_info_score`,
  `adjusted_mutual_info_score`, `homogeneity_score`, `completeness_score`,
  `v_measure_score`, `fowlkes_mallows_score`

**Metrics over a table:** `confusion_matrix` (long format), `silhouette_score`.

**Manage the registry:** `SELECT * FROM sklearn.list_models();`,
`sklearn.model_info('name')`, `sklearn.drop_model('name')`.

---

## Where models live

A `fit_…` call always returns the model as a **`model` BLOB** *and* saves it to
the registry when you pass `model_name`. Two ways to keep a model:

**In a DuckDB table (BLOB).** Store the `model` column anywhere; pass it to
`predict` via a session variable (the data subquery is the table function's one
allowed subquery, so the model scalar comes through `getvariable`):

```sql
CREATE TABLE models AS
  SELECT 'churn_gb' AS name, model
  FROM sklearn.fit_gradient_boosting_classifier(
    (SELECT customer_id, tenure, monthly_spend, support_tickets, churned FROM churn),
    target := 'churned', id := 'customer_id');

SET VARIABLE m = (SELECT model FROM models WHERE name = 'churn_gb');
SELECT * FROM sklearn.predict((SELECT * FROM churn), model := getvariable('m'), id := 'customer_id');
```

**In the named registry.** Pass `model_name` to `fit_…`, then reference it by
name (`predict(..., model_name := 'churn_gb')`). The registry is local disk by
default (`SKLEARN_MODELS_DIR`, default `./models`); an S3/R2 backend is the
planned drop-in (`registry.get_store()` is the single seam). `predict` takes
**either** `model_name :=` or `model :=`.

DuckDB BLOBs cap near 2 GB, so a very large ensemble may not fit in a column —
use the registry for those.

### Model serialization & safety

Models are stored with [**skops**](https://skops.readthedocs.io/), not pickle:
loading reconstructs only known types instead of executing arbitrary code, and
this worker further restricts the trusted set to the `scikit-learn` / `numpy` /
`scipy` namespaces — a crafted artifact can't smuggle in an arbitrary callable.

> [!NOTE]
> skops removes pickle's code-execution risk, but it isn't a trust oracle — keep
> the registry / `SKLEARN_MODELS_DIR` writable only by trusted users. skops
> stores scikit-learn objects, so it is **not** version-independent: a model may
> fail to load or behave differently under a different scikit-learn version. The
> worker records the fitting version and logs a `duckdb_logs()` warning on
> mismatch. (Fully version-independent inference would mean exporting to ONNX, at
> the cost of estimator coverage.)

---

## Install

```sh
pip install vgi-sklearn        # or: uvx vgi-sklearn
```

This provides the `vgi-sklearn` (stdio, for DuckDB to spawn) and
`vgi-sklearn-http` console scripts. Then `ATTACH 'sklearn' (TYPE vgi, LOCATION
'vgi-sklearn')`. To attach a hosted HTTP deployment instead:
`ATTACH 'sklearn' (TYPE vgi, LOCATION 'https://<host>')`.

## Local development

```sh
uv sync                       # install worker + deps from uv.lock (PyPI vgi-python)
uv run pytest tests/ -q       # unit tests (incl. pydoclint docstring gate)
uvx ruff check . && uvx ruff format --check .
```

To develop against **local** `vgi-python` / `vgi-rpc` checkouts instead of PyPI,
use the Makefile targets (worker = `uv run sklearn_worker.py`):

```sh
make venv
make test-stdio    # SQL integration tests, worker as a subprocess
make test-http     # SQL integration tests against a local HTTP server
```

The `test/sql/*.test` sqllogictest suite is the authoritative integration test.
CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs the unit + SQL
suites on Linux/macOS/Windows against the **signed community `vgi` extension**
via a prebuilt `haybarn-unittest` — no local C++ build (see
[`ci/README.md`](ci/README.md)).

## Publishing

Releases publish to PyPI via [`.github/workflows/publish.yml`](.github/workflows/publish.yml):
publishing a GitHub Release runs the full CI suite, then `uv build && uv publish`
(token in the `PYPI_API_TOKEN` repo secret). Bump `version` in `pyproject.toml`
before tagging.

## Deployment (Fly.io)

```sh
make vendor-sync   # copy vgi-python / vgi-rpc into vendor/ for the Docker build
make deploy        # build, smoke-test, push, deploy
fly volumes create sklearn_models --size 1 --region iad   # one-time, for the registry
```

## Layout

```
vgi_sklearn/
  worker.py            assembles the `sklearn` catalog; main() / main_http() entry points
  datasets.py          dataset table functions
  metrics.py           metric aggregates
  table_metrics.py     confusion_matrix / silhouette_score
  transforms.py        unsupervised fit_transform (buffering)
  models.py            generic fit / predict / cross_val_predict / registry mgmt
  typed_models.py      generated fit_<estimator> functions (typed hyperparameters)
  registry.py          ModelStore (local disk; S3/R2 seam) + model-BLOB pack/unpack
  buffering.py         shared sink/combine/matrix helpers
  schema_utils.py      Arrow schema helpers
sklearn_worker.py      dev/Fly stdio shim over vgi_sklearn.worker (for `uv run`)
serve.py               dev/Fly HTTP shim over vgi_sklearn.worker
```

## License

MIT — see [LICENSE](LICENSE).
