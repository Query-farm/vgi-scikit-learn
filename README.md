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
    (SELECT sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm, target FROM flowers),
    target := 'target', n_estimators := 200);

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
- **`target`** names your label column (training only). **Every other column you
  select is a feature** — so for `fit`, just `SELECT` your features and the
  target; don't include an identifier column. Numeric and boolean columns are
  used as-is; **string columns are treated as categorical and one-hot-encoded
  automatically** (the encoding is stored with the model, so `predict` re-applies
  it). Need the encoding as data instead? `ordinal_encoder` / `one_hot_encoder`
  expose it directly.
- **`id` is for getting results back, and it's per-row functions that need it.**
  `predict`, the transforms, and `cross_val_predict` emit one row per input row
  and copy your `id` onto each, so a plain `JOIN ... USING (id)` reattaches
  results to the source. `fit` returns a single summary row — there's nothing to
  join back — so it needs no `id`; you just leave the identifier out of the
  `SELECT`. (`fit` *does* accept an optional `id :=` for the convenience of
  passing a wide projection like `SELECT *`: it then drops that column from the
  features so the model doesn't train on a key.)
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
  -- just the features + the target; leave customer_id out (it isn't a feature)
  (SELECT tenure, monthly_spend, support_tickets, churned FROM churn),
  model_name := 'churn_gb',          -- store it in the registry under this name
  target := 'churned',               -- the label column; everything else is a feature
  n_estimators := 300,
  learning_rate := 0.05,
  max_depth := 3);
```

`fit` returns one summary row (and the model itself as a BLOB); it doesn't echo
your rows, so it needs no `id` — you simply don't select the identifier. If it's
easier to pass a wide projection that already includes an id, add `id :=` and
`fit` will keep that column out of the features:

```sql
-- a regressor; SELECT * includes diabetes()'s sample_id, so name it as the id
SELECT estimator, task, train_score
FROM sklearn.fit_random_forest_regressor(
  (SELECT * FROM sklearn.diabetes()),
  model_name := 'diabetes_rf', target := 'target',
  id := 'sample_id',                       -- keeps sample_id out of the features
  n_estimators := 400, max_depth := 0);    -- max_depth := 0 means "no limit"
```

Available estimators (each is `sklearn.fit_<name>`):

| Family | Functions | Common typed args |
| --- | --- | --- |
| Linear | `logistic_regression`, `linear_regression`, `ridge`, `lasso`, `elastic_net`, `ridge_classifier`, `sgd_classifier`/`_regressor`, `bayesian_ridge`, `huber_regressor`, `quantile_regressor` | `C`, `alpha`, `l1_ratio`, `max_iter`, `fit_intercept`, `penalty`, `solver`, `loss` |
| GLMs | `poisson_regressor`, `gamma_regressor`, `tweedie_regressor` | `alpha`, `power`, `max_iter`, `fit_intercept` |
| Trees / ensembles | `decision_tree_classifier`/`_regressor`, `random_forest_classifier`/`_regressor`, `extra_trees_classifier`/`_regressor`, `gradient_boosting_classifier`/`_regressor`, `hist_gradient_boosting_classifier`/`_regressor`, `ada_boost_classifier`/`_regressor`, `bagging_classifier`/`_regressor` | `n_estimators`, `max_depth`, `learning_rate`, `min_samples_split`, `subsample`, `max_samples`, `random_state` |
| SVM | `svc`, `svr`, `linear_svc`, `linear_svr` | `C`, `kernel`, `gamma`, `degree`, `epsilon`, `loss` |
| Neighbors | `knn_classifier`, `knn_regressor` | `n_neighbors`, `weights`, `p` |
| Neural net | `mlp_classifier`, `mlp_regressor` | `hidden_units`, `alpha`, `max_iter`, `learning_rate_init` |
| Naive Bayes | `gaussian_nb`, `multinomial_nb`, `bernoulli_nb`, `complement_nb` | `var_smoothing`, `alpha`, `fit_prior`, `binarize` |
| Discriminant | `lda`, `qda` | `solver`, `tol`, `reg_param` |

> Need a hyperparameter that isn't exposed as a typed argument? The generic
> `sklearn.fit((SELECT ...), estimator := 'ridge', target := 'y', params := '{"alpha": 0.3, "solver": "svd"}')`
> accepts any scikit-learn parameter as a JSON object.

### Build a pipeline (preprocess → model in one artifact)

`fit_pipeline` chains preprocessing steps and a final estimator, fits them
together, and stores the result as a single model — so it trains and serves
without leakage, and you score it with the **same `predict`** (no separate apply):

```sql
SELECT model_name, estimator, n_features
FROM sklearn.fit_pipeline(
  (SELECT tenure, monthly_spend, support_tickets, churned FROM churn),
  steps := '[{"kind": "simple_imputer", "params": {"strategy": "median"}},
             {"kind": "standard_scaler"},
             {"kind": "pca", "params": {"n_components": 3}}]',
  estimator := 'logistic_regression', target := 'churned', model_name := 'churn_pipe');

-- predict (and cross_val_predict, permutation_importance, ...) work as usual
SELECT * FROM sklearn.predict((SELECT * FROM new_customers), model_name := 'churn_pipe', id := 'customer_id');
```

`steps` is a JSON array of `{kind, params}`; `kind` is any stored-transformer kind
(`standard_scaler`, `simple_imputer`, `pca`, `truncated_svd`, …). String features
are one-hot-encoded ahead of the steps automatically.

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

Prefer the held-out score per fold (mean ± spread)? `cross_val_score` returns one
row per fold:

```sql
SELECT avg(score) AS mean_cv, stddev(score) AS sd
FROM sklearn.cross_val_score(
       (SELECT tenure, monthly_spend, support_tickets, churned FROM churn),
       estimator := 'gradient_boosting_classifier', target := 'churned', cv := 5);
```

### Which features matter? (permutation importance)

`permutation_importance` shuffles each feature in turn and measures the drop in a
stored model's score — model-agnostic, so it works for any estimator:

```sql
SELECT feature, round(importance_mean, 4) AS importance
FROM sklearn.permutation_importance(
       (SELECT * FROM churn), model_name := 'churn_gb', target := 'churned')
ORDER BY importance DESC;
```

For a quick *model-free* filter, `select_k_best` scores each feature against the
target (ANOVA F, mutual information, or chi²) and flags the top `k`;
`variance_threshold` drops near-constant features. Both return one row per
feature, so you pick the winners in SQL:

```sql
SELECT feature FROM sklearn.select_k_best(
         (SELECT tenure, monthly_spend, support_tickets, churned FROM churn),
         target := 'churned', k := 2)
WHERE selected;
```

### Vectorize text

`count_vectorizer` and `tfidf_vectorizer` tokenize a text column into a
document-term matrix in long format — `(id, term, value)` — which you pivot,
join, or rank in SQL:

```sql
-- the 5 highest-weighted terms per document
SELECT id, term, value
FROM sklearn.tfidf_vectorizer((SELECT id, body FROM docs), id := 'id', text := 'body')
QUALIFY row_number() OVER (PARTITION BY id ORDER BY value DESC) <= 5;
```

### Tune hyperparameters (grid search)

`grid_search` cross-validates every combination of the hyperparameters you list
and returns the leaderboard. The estimator and its grid are one **tagged-union**
argument — `union_value(<estimator> := {param: [values], …})` — so you only ever
see the hyperparameters that estimator actually has:

```sql
SELECT params, round(mean_test_score, 3) AS score, rank
FROM sklearn.grid_search(
  (SELECT tenure, monthly_spend, support_tickets, churned FROM churn),
  target := 'churned',
  estimator := union_value(gradient_boosting_classifier := {
    'n_estimators': [100, 300],
    'max_depth':    [2, 3],
    'learning_rate':[0.05, 0.1]}))
ORDER BY rank;
```

Only the hyperparameters you list are searched; the rest stay at their defaults.
The refit best model is attached as a `model` BLOB on the single best row — grab
it with `WHERE model IS NOT NULL`, or pass `model_name :=` to also store it:

```sql
CREATE TABLE best AS
SELECT model FROM sklearn.grid_search(
  (SELECT tenure, monthly_spend, support_tickets, churned FROM churn),
  target := 'churned',
  estimator := union_value(svc := {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}))
WHERE model IS NOT NULL;

SET VARIABLE m = (SELECT model FROM best);
SELECT * FROM sklearn.predict((SELECT * FROM new_customers), model := getvariable('m'), id := 'customer_id');
```

> `grid_search` uses union-typed arguments and needs a vgi-python with
> union-tag-preserving decoding (newer than 0.8.2). Against an older vgi-python
> the function is simply not registered.

### Train a model per group (segment)

Sometimes you want *one model per segment* — per region, per cohort, per product
line. `fit_model` is an **aggregate**, so `GROUP BY` does the partitioning for
free and you get one model per group in a single query. Features go in as a
named `STRUCT`; the target can be numeric **or** string class labels.

```sql
-- one churn model per segment
CREATE TABLE segment_models AS
SELECT (customer_id % 3) AS segment,       -- use your real segment column
       sklearn.fit_model({'tenure': tenure, 'monthly_spend': monthly_spend, 'support_tickets': support_tickets},
                         churned, estimator := 'gradient_boosting_classifier', hyperparams := '{}') AS m
FROM churn
GROUP BY segment;

SELECT segment, m.task, m.n_samples, round(m.train_score, 3) FROM segment_models;
```

`m` is a `STRUCT` holding the `model` BLOB plus diagnostics (`task`, `n_samples`,
`n_features`, `n_classes`, `train_score`). To score, the prediction functions are
**scalars** that take a per-row model BLOB and a feature struct — so each row is
scored by *its* group's model via a plain join:

```sql
SELECT c.customer_id,
       sklearn.predict_class_one(m.m.model,
         {'tenure': c.tenure, 'monthly_spend': c.monthly_spend, 'support_tickets': c.support_tickets}) AS prediction
FROM churn c
JOIN segment_models m ON (c.customer_id % 3) = m.segment;
```

- `predict_one(model, features) → DOUBLE` — regression / numeric class.
- `predict_class_one(model, features) → VARCHAR` — the class label as text (works
  for string *and* numeric labels).
- `predict_proba_one(model, features) → DOUBLE[]` — per-class probabilities.

Features align **by name** (reorder-safe; a missing feature errors), and a model
trained on string labels predicts string labels. The model BLOB is the same
format `fit`/`grid_search` produce, so these scalars also score any model you've
stored in a table or registry.

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

These all refit on whatever you pass them. To **fit a transformer once and reuse
it** — scale your training data and apply the *same* shift/scale to new data,
without leakage — use `fit_transformer` / `apply_transform` (the transform
analogue of `fit` / `predict`):

```sql
-- fit a scaler on training data and store it
SELECT * FROM sklearn.fit_transformer(
  (SELECT tenure, monthly_spend, support_tickets FROM churn_train),
  transformer_name := 'churn_scaler', kind := 'standard_scaler');

-- apply the stored scaler to new data (uses the training mean/variance)
SELECT * FROM sklearn.apply_transform(
  (SELECT customer_id, tenure, monthly_spend, support_tickets FROM churn_new),
  transformer_name := 'churn_scaler', id := 'customer_id');
```

`kind` is any of `standard_scaler`, `minmax_scaler`, `robust_scaler`,
`maxabs_scaler`, `normalizer`, `power_transformer`, `quantile_transformer`,
`simple_imputer`, `binarizer`, `kbins_discretizer`, `pca`, `truncated_svd`
(parameters via a JSON `params :=`). Like `fit`/`predict`, `fit_transformer` also
returns a portable BLOB, and `apply_transform` accepts `transformer :=` instead
of a registry name; `list_transformers` / `drop_transformer` manage the registry.

### Encode categorical (string) columns

`fit`/`predict` already one-hot string features for you, but you can also
materialize the encoding. `ordinal_encoder` keeps a fixed width (one integer
code column per feature); `one_hot_encoder` emits **long format** — one row per
active cell `(id, feature, category, value)` — which sidesteps the unknown width
of a wide one-hot:

```sql
-- integer codes, one column per categorical feature
SELECT * FROM sklearn.ordinal_encoder(
  (SELECT customer_id, plan, region FROM customers), id := 'customer_id');

-- one row per active category; pivot it back to a wide matrix in SQL
PIVOT sklearn.one_hot_encoder(
        (SELECT customer_id, plan FROM customers), id := 'customer_id')
  ON category USING sum(value) GROUP BY customer_id;
```

A `NULL`/unseen value encodes to `-1` (ordinal) or contributes no active cell
(one-hot).

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
(escape hatch with JSON `params`), `fit_pipeline` (preprocessing steps +
estimator as one model), `predict`, `cross_val_predict`,
`cross_val_score` (per-fold held-out scores), `permutation_importance`
(model-agnostic feature importance), `grid_search` / `randomized_search`
(union-typed hyperparameter search), `list_models`, `model_info`, `drop_model`.

**Per-group models:** `fit_model` (aggregate — one model per `GROUP BY` group),
`predict_one` / `predict_class_one` / `predict_proba_one` (scalars — per-row,
by-name features).

**Transforms** (table in, `id` passthrough):
- Scaling / preprocessing — `standard_scaler`, `minmax_scaler`, `robust_scaler`,
  `maxabs_scaler`, `normalizer`, `power_transformer`, `quantile_transformer`,
  `binarizer`, `kbins_discretizer`, `simple_imputer`
- Encoding — `ordinal_encoder`, `one_hot_encoder`
- Text — `count_vectorizer`, `tfidf_vectorizer` (long format `(id, term, value)`)
- Feature selection — `select_k_best`, `variance_threshold` (per-feature scores
  + a `selected` flag)
- Decomposition / manifold — `pca`, `truncated_svd`, `tsne`, `isomap`,
  `spectral_embedding`, `mds`
- Clustering — `kmeans`, `minibatch_kmeans`, `dbscan`, `optics`,
  `agglomerative_clustering`, `spectral_clustering`, `mean_shift`, `birch`,
  `gaussian_mixture`
- Outlier detection — `isolation_forest`, `local_outlier_factor`,
  `one_class_svm`, `elliptic_envelope`

**Stored transformers** (fit once, apply to new data — like `fit`/`predict`):
`fit_transformer`, `apply_transform`, `list_transformers`, `drop_transformer`.

**Metric aggregates** over `(y_true, y_pred)`:
- Regression — `mean_squared_error`, `root_mean_squared_error`,
  `mean_absolute_error`, `r2_score`, `explained_variance_score`,
  `mean_absolute_percentage_error`, `max_error`, `median_absolute_error`,
  `mean_squared_log_error`, `mean_pinball_loss`
- Classification — `accuracy_score`, `precision_score`, `recall_score`,
  `f1_score`, `balanced_accuracy_score`, `matthews_corrcoef`,
  `cohen_kappa_score`, `jaccard_score`, `hamming_loss`, `zero_one_loss`
- Probability / ranking — `roc_auc_score`, `average_precision_score`,
  `log_loss`, `brier_score_loss`
- Clustering — `adjusted_rand_score`, `normalized_mutual_info_score`,
  `adjusted_mutual_info_score`, `homogeneity_score`, `completeness_score`,
  `v_measure_score`, `fowlkes_mallows_score`

**Metrics over a table:** `confusion_matrix` (long format), `silhouette_score`,
and the binary curves `roc_curve`, `precision_recall_curve`, `calibration_curve`.

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
    (SELECT tenure, monthly_spend, support_tickets, churned FROM churn),
    target := 'churned');

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
  search.py            grid_search (discriminated-union hyperparameter search)
  grouped.py           fit_model (aggregate) + predict_* scalars (per-group modeling)
  registry.py          ModelStore (local disk; S3/R2 seam) + model-BLOB pack/unpack
  buffering.py         shared sink/combine/matrix helpers
  schema_utils.py      Arrow schema helpers
sklearn_worker.py      dev/Fly stdio shim over vgi_sklearn.worker (for `uv run`)
serve.py               dev/Fly HTTP shim over vgi_sklearn.worker
```

## License

MIT — see [LICENSE](LICENSE).
