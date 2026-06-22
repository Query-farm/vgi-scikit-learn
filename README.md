# vgi-sklearn

A [VGI](https://github.com/query-farm/vgi-python) worker that brings
[scikit-learn](https://scikit-learn.org/) into DuckDB/SQL: reference datasets,
scoring metrics, unsupervised transforms, and a supervised train/predict model
registry — all callable as SQL functions.

```sql
INSTALL vgi FROM community; LOAD vgi;
ATTACH 'sklearn' (TYPE vgi, LOCATION 'uv run sklearn_worker.py');

SELECT * FROM sklearn.iris();
SELECT sklearn.r2_score(actual, predicted) FROM my_predictions;
SELECT * FROM sklearn.kmeans((SELECT id, x, y FROM points), id := 'id', n_clusters := 3);
```

## How it maps scikit-learn onto SQL

scikit-learn is built around stateful *fit / transform / predict* estimators;
SQL is set-oriented. Each area is mapped to the VGI primitive that fits its
data flow:

| Area | SQL surface | VGI primitive |
| --- | --- | --- |
| **Datasets** | `SELECT * FROM sklearn.iris()` | table function (source) |
| **Metrics** | `sklearn.r2_score(y, yhat)` over `GROUP BY` | aggregate function |
| **Transforms** | `sklearn.pca((SELECT ...), n_components := 2)` | table-buffering (`fit_transform`) |
| **Fit** | `sklearn.fit((SELECT ...), model_name := 'm', ...)` | table-buffering → registry |
| **Predict** | `sklearn.predict((SELECT ...), model_name := 'm')` | streaming table-in-out |

**Conventions** for the transform / fit / predict functions:

- The input relation **is** the feature matrix `X`, passed as a `(SELECT ...)`
  subquery. Named arguments use DuckDB's `name := value` syntax.
- **`id`** names a passthrough column: it is *excluded from the features* and
  copied unchanged onto each output row, so you can join the result back to the
  source (e.g. attach a `cluster`/`prediction` to the row it came from). It is
  optional; omit it if you don't need to line results up with input rows.
- **`target`** (required for supervised `fit` / `cross_val_predict`) names the
  label column, also excluded from features.
- **Every remaining column is treated as a numeric feature.** Non-numeric
  columns raise a clear error — `SELECT` only the columns you want as features
  (don't pass `SELECT *` when it includes text/label columns), or scale/encode
  them first.
- Hyperparameters are passed as a JSON string: `params := '{"n_estimators": 300}'`.
  Unknown hyperparameters are rejected with the list of valid ones for that estimator.
- **`fit`/`predict` align features by name**, not position: `predict` selects
  the model's fitted feature columns by name (input order is irrelevant, extra
  columns are ignored) and errors if a required feature column is missing.

## Function catalog

### Datasets (`sklearn.<name>()`)
`iris`, `wine`, `digits`, `breast_cancer` (classification), `diabetes`,
`california_housing` (regression), and generators `make_classification`,
`make_regression`, `make_blobs`, `make_moons`, `make_circles`.

```sql
SELECT target_name, avg(petal_length_cm) FROM sklearn.iris() GROUP BY target_name;
SELECT * FROM sklearn.make_blobs(n_samples := 300, centers := 4);
```

### Metrics (aggregates over two columns)
Regression: `mean_squared_error`, `root_mean_squared_error`,
`mean_absolute_error`, `r2_score`, `explained_variance_score`,
`mean_absolute_percentage_error`, `max_error`, `median_absolute_error`.
Classification: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`
(macro), `balanced_accuracy_score`, `matthews_corrcoef`, `cohen_kappa_score`.
Probability/ranking: `roc_auc_score`, `average_precision_score`, `log_loss`.
Clustering comparison: `adjusted_rand_score`, `normalized_mutual_info_score`,
`adjusted_mutual_info_score`, `homogeneity_score`, `completeness_score`,
`v_measure_score`, `fowlkes_mallows_score`.

Table-input metrics: `confusion_matrix` (long format), `silhouette_score`.

```sql
SELECT model, sklearn.f1_score(y, yhat) FROM preds GROUP BY model;
SELECT * FROM sklearn.confusion_matrix((SELECT y, yhat FROM preds), actual := 'y', predicted := 'yhat');
```

### Transforms (`fit_transform` over the whole input)
Scalers: `standard_scaler`, `minmax_scaler`, `robust_scaler`, `normalizer`.
Imputation: `simple_imputer`. Decomposition: `pca`, `truncated_svd`.
Clustering: `kmeans`, `dbscan`. Outliers: `isolation_forest`.

```sql
-- pca: select sample_id (carried through) + the 4 numeric features only;
-- target / target_name are NOT selected, since they are not features.
SELECT * FROM sklearn.pca(
  (SELECT sample_id, sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm FROM sklearn.iris()),
  id := 'sample_id', n_components := 2);

-- points(id, x, y): id is passed through; x and y are the features.
SELECT * FROM sklearn.isolation_forest((SELECT id, x, y FROM points), id := 'id', contamination := 0.05);
```

### Models (registry-backed)
`fit`, `fit_<estimator>` (typed), `predict`, `cross_val_predict`,
`list_models`, `model_info`, `drop_model`.

Estimators (for generic `fit`'s `estimator :=` arg): `logistic_regression`,
`random_forest_classifier`/`_regressor`, `gradient_boosting_classifier`/`_regressor`,
`hist_gradient_boosting_classifier`/`_regressor`, `linear_regression`, `ridge`,
`lasso`, `svc`, `svr`, `knn_classifier`/`_regressor`,
`decision_tree_classifier`/`_regressor`, `mlp_classifier`/`_regressor`,
`gaussian_nb`.

```sql
-- train (generic fit: estimator + JSON hyperparameters)
SELECT * FROM sklearn.fit(
  (SELECT sample_id, sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm, target FROM sklearn.iris()),
  model_name := 'iris_rf', estimator := 'random_forest_classifier', target := 'target', id := 'sample_id',
  params := '{"n_estimators": 300}');

-- predict later (optionally with per-class probabilities)
SELECT * FROM sklearn.predict((SELECT * FROM new_flowers), model_name := 'iris_rf', id := 'id', with_proba := true);

-- evaluate without persisting
SELECT sklearn.accuracy_score(i.target, p.prediction)
FROM sklearn.cross_val_predict(
       (SELECT * FROM iris_xy), estimator := 'logistic_regression', target := 'target', id := 'sample_id') p
JOIN iris_xy i USING (sample_id);

SELECT * FROM sklearn.list_models();
SELECT * FROM sklearn.drop_model('iris_rf');
```

### Typed estimator functions (discoverable hyperparameters)

Each estimator also has a `fit_<estimator>` function that exposes its common
hyperparameters as **native, typed SQL named arguments** — visible in the
catalog and DuckDB autocomplete, and type-checked:

```sql
SELECT * FROM sklearn.fit_random_forest_classifier(
  (SELECT sample_id, sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm, target FROM sklearn.iris()),
  model_name := 'iris_rf', target := 'target', id := 'sample_id',
  n_estimators := 300, max_depth := 8);   -- max_depth := 0 means unlimited
```

There is one per estimator (`fit_ridge`, `fit_svc`, `fit_mlp_classifier`, ...).
They behave exactly like `fit`; use the generic `fit` with JSON `params` for any
hyperparameter not surfaced as a typed argument.

## Model storage: registry or in-database BLOB

Every `fit` / `fit_<estimator>` returns the fitted model as a **`model` BLOB**
column (estimator + metadata packed together) *and*, when `model_name` is given,
persists it to the registry. So `model_name` is optional — you choose where the
model lives:

```sql
-- keep models inside the database, in a regular table
CREATE TABLE models AS
SELECT 'iris_rf' AS name, model
FROM sklearn.fit_random_forest_classifier(
  (SELECT sample_id, sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm, target FROM sklearn.iris()),
  target := 'target', id := 'sample_id');

-- predict from a BLOB. A table function allows only one subquery parameter
-- (the data), so pass the model scalar via a session variable:
SET VARIABLE m = (SELECT model FROM models WHERE name = 'iris_rf');
SELECT * FROM sklearn.predict(
  (SELECT sample_id, sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm FROM sklearn.iris()),
  model := getvariable('m'), id := 'sample_id');
```

`predict` takes **either** `model_name :=` (registry) **or** `model :=` (a BLOB).
DuckDB BLOBs are capped near 2 GB, so very large ensembles may not fit in-DB.

The named registry sits behind the `ModelStore` interface in
`vgi_sklearn/registry.py`:

- **Local disk** (default): `SKLEARN_MODELS_DIR` (default `./models`).
- **S3 / Cloudflare R2**: not yet implemented — `get_store()` is the single seam
  where an `S3Store` drops in.

On Fly.io the local store is backed by a mounted volume (see `fly.toml`) so
named models survive machine restarts. `predict` records the scikit-learn version
used to fit and logs a warning (visible in `duckdb_logs()`) if the worker's
version differs.

## Local development

```sh
make venv          # create .venv with vgi + scikit-learn (from local checkouts)
make pytest        # unit tests
make test-stdio    # SQL integration tests with the worker as a subprocess
make test-http     # SQL integration tests against a local HTTP server
```

SQL tests require DuckDB's `unittest` runner built with the VGI extension
(`VGI_BUILD_DIR`).

## Deployment (Fly.io)

```sh
make vendor-sync   # copy vgi-python / vgi-rpc into vendor/ for the Docker build
make deploy        # build, smoke-test, push, and deploy
fly volumes create sklearn_models --size 1 --region iad   # one-time, for the registry
```

`serve.py` runs the worker over HTTP; attach the deployed endpoint with
`ATTACH 'sklearn' (TYPE vgi, LOCATION 'https://<app>.fly.dev');`.

## Layout

```
sklearn_worker.py      entry point; assembles the `sklearn` catalog
serve.py               HTTP entry point (Fly.io)
vgi_sklearn/
  datasets.py          dataset table functions
  metrics.py           metric aggregates
  table_metrics.py     confusion_matrix / silhouette_score
  transforms.py        unsupervised fit_transform (buffering)
  models.py            fit / predict / cross_val_predict / registry mgmt
  registry.py          ModelStore (local disk; S3/R2 seam)
  buffering.py         shared sink/combine/matrix helpers
  schema_utils.py      Arrow schema helpers
```
