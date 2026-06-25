"""VGI worker exposing scikit-learn to DuckDB/SQL.

Assembles the per-area implementation modules in ``vgi_sklearn`` into a single
``sklearn`` catalog and provides the process entry points. The repo-root
``sklearn_worker.py`` / ``serve.py`` are thin shims over this module for
``uv run`` and the Fly.io container; installed users get the ``vgi-sklearn`` and
``vgi-sklearn-http`` console scripts, which call ``main`` / ``main_http`` here.

    ATTACH 'sklearn' (TYPE vgi, LOCATION 'vgi-sklearn');
    SELECT * FROM sklearn.datasets.iris();
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
from typing import Any

from vgi import Worker
from vgi.catalog import Catalog, ReadOnlyCatalogInterface, Schema, Table
from vgi.catalog.catalog_interface import CatalogAttachResult, CatalogInfo

from vgi_sklearn import __version__
from vgi_sklearn.datasets import DATASET_FUNCTIONS
from vgi_sklearn.feature_selection import FEATURE_SELECTION_FUNCTIONS
from vgi_sklearn.grouped import GROUPED_FUNCTIONS
from vgi_sklearn.metrics import METRIC_FUNCTIONS
from vgi_sklearn.models import MODEL_FUNCTIONS
from vgi_sklearn.pipeline import PIPELINE_FUNCTIONS
from vgi_sklearn.search import SEARCH_FUNCTIONS
from vgi_sklearn.splitters import SPLITTER_FUNCTIONS
from vgi_sklearn.stored_transforms import STORED_TRANSFORM_FUNCTIONS
from vgi_sklearn.table_metrics import TABLE_METRIC_FUNCTIONS
from vgi_sklearn.text import TEXT_FUNCTIONS
from vgi_sklearn.transforms import TRANSFORM_FUNCTIONS
from vgi_sklearn.typed_models import TYPED_FIT_FUNCTIONS

log = logging.getLogger(__name__)

# The version the worker advertises over VGI. `implementation_version` is the
# worker *software* version (a semver per the VGI protocol), so it must be the
# released package version — not a build/commit id. Both it and the data version
# track __version__, which is the single source bumped per release.
IMPLEMENTATION_VERSION = __version__
DATA_VERSION = __version__
# data_version_spec is advertised as a SemVer *range* (a packaging SpecifierSet),
# not a bare version. The worker regenerates its data each release, so it serves
# exactly the current data version — an exact-match range.
DATA_VERSION_SPEC = f"=={DATA_VERSION}"
# Build provenance only (Sentry release / diagnostics) — NOT the advertised
# implementation version, which must stay a semver.
GIT_COMMIT = os.environ.get("VGI_SKLEARN_GIT_COMMIT") or "unknown"

# Functions are split across schemas by scikit-learn area. A single schema is
# capped at 50 objects (VGI117), and there are ~150 functions, so the flat
# namespace is divided onto the existing per-area module lists; each schema stays
# well under the cap. The core fit/predict workflow lives in the default `models`
# schema, so `sklearn.models.fit(...)` etc. still resolve unqualified.
_DEFAULT_SCHEMA = "models"
_SCHEMA_FUNCTIONS: dict[str, list[type]] = {
    "datasets": [*DATASET_FUNCTIONS],
    "metrics": [*METRIC_FUNCTIONS, *TABLE_METRIC_FUNCTIONS],
    "preprocessing": [
        *TRANSFORM_FUNCTIONS,
        *STORED_TRANSFORM_FUNCTIONS,
        *TEXT_FUNCTIONS,
        *FEATURE_SELECTION_FUNCTIONS,
    ],
    "models": [
        *MODEL_FUNCTIONS,
        *PIPELINE_FUNCTIONS,
        *SPLITTER_FUNCTIONS,
        *GROUPED_FUNCTIONS,
        *SEARCH_FUNCTIONS,
    ],
    "estimators": [*TYPED_FIT_FUNCTIONS],
}
_FUNCTIONS: list[type] = [fn for fns in _SCHEMA_FUNCTIONS.values() for fn in fns]

# Provenance / about link advertised on the catalog (VGI source_url).
SOURCE_URL = "https://github.com/query-farm/vgi-scikit-learn"

# Catalog-level metadata surfaced through duckdb_databases() (comment + tags).
# The description_llm/_md tags feed agent/doc consumers; author/copyright/license
# advertise provenance.
_CATALOG_COMMENT = "scikit-learn datasets, metrics, transforms, and a train/predict model registry for DuckDB/SQL"
# Catalog-level description: the high-level "what this worker is".
_CATALOG_DESCRIPTION_LLM = (
    "scikit-learn for SQL. Load toy and generated datasets; compute regression, "
    "classification, and clustering metrics as aggregates; fit and persist "
    "transformers and models (fit returns a model BLOB, predict aligns features "
    "by name and auto-encodes string labels); run cross-validation, grid and "
    "randomized hyperparameter search, pipelines, and per-group modeling — all as "
    "DuckDB table, aggregate, and scalar functions."
)
_CATALOG_DESCRIPTION_MD = (
    "# scikit-learn for SQL\n\n"
    "Exposes [scikit-learn](https://scikit-learn.org) to DuckDB/SQL as VGI functions:\n\n"
    "- **Datasets** — toy datasets and generators (`iris`, `make_classification`, ...)\n"
    "- **Metrics** — regression/classification/clustering scores as aggregates\n"
    "- **Transforms** — scalers, encoders, decomposition (fit-transform + stored)\n"
    "- **Models** — `fit`/`predict`, typed `fit_<estimator>`, cross-validation, "
    "grid/randomized search, pipelines, and per-group modeling\n\n"
    "Models and transformers are stored as reusable BLOBs in a registry."
)
# Guaranteed-runnable, self-contained examples advertised on the catalog
# (VGI509): each is fully schema-qualified and executes as written against a
# freshly attached worker. Multi-statement examples run in order in one session,
# so a model can be fit, stashed in a session variable (the model BLOB can't ride
# a table function's single subquery slot), and then used by predict.
_CATALOG_EXECUTABLE_EXAMPLES = json.dumps(
    [
        {
            "description": "Load the built-in iris dataset.",
            "sql": "SELECT * FROM sklearn.datasets.iris() LIMIT 5",
        },
        {
            "description": "Coefficient of determination over inline (actual, predicted) pairs.",
            "sql": (
                "SELECT sklearn.metrics.r2_score(actual, predicted) "
                "FROM (VALUES (3.0, 2.5), (5.0, 5.1), (7.0, 6.8)) AS t(actual, predicted)"
            ),
        },
        {
            "description": "Fit a random-forest classifier on iris, then predict with the fitted model.",
            "sql": [
                (
                    "SET VARIABLE iris_model = ("
                    "SELECT model FROM sklearn.models.fit((SELECT * FROM sklearn.datasets.iris()), "
                    "estimator := 'random_forest_classifier', target := 'target'))"
                ),
                (
                    "SELECT sample_id, prediction "
                    "FROM sklearn.models.predict((SELECT * FROM sklearn.datasets.iris()), "
                    "model := getvariable('iris_model'), id := 'sample_id') LIMIT 5"
                ),
            ],
        },
        {
            "description": "Reduce iris to two principal components (fit-transform).",
            "sql": (
                "SELECT * FROM sklearn.preprocessing.pca("
                "(SELECT sample_id, sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm "
                "FROM sklearn.datasets.iris()), n_components := 2, id := 'sample_id') LIMIT 5"
            ),
        },
        {
            "description": "Three-fold cross-validated accuracy of a logistic-regression classifier on iris.",
            "sql": (
                "SELECT fold, score FROM sklearn.models.cross_val_score("
                "(SELECT * FROM sklearn.datasets.iris()), "
                "estimator := 'logistic_regression', target := 'target', cv := 3)"
            ),
        },
    ]
)
_CATALOG_TAGS = {
    "vgi.doc_llm": _CATALOG_DESCRIPTION_LLM,
    "vgi.doc_md": _CATALOG_DESCRIPTION_MD,
    "vgi.author": "Query Farm <hello@query.farm>",
    "vgi.copyright": "Copyright 2026 Query Farm LLC - https://query.farm",
    "vgi.license": "MIT",
    "vgi.support_contact": f"{SOURCE_URL}/issues",
    "vgi.support_policy_url": f"{SOURCE_URL}/blob/main/SUPPORT.md",
    "vgi.title": "scikit-learn for SQL",
    "vgi.keywords": json.dumps(
        [
            "scikit-learn",
            "machine learning",
            "datasets",
            "metrics",
            "transforms",
            "models",
            "regression",
            "classification",
            "clustering",
            "cross-validation",
            "hyperparameter search",
        ]
    ),
    "vgi.executable_examples": _CATALOG_EXECUTABLE_EXAMPLES,
}

# Per-schema metadata. Each schema carries its own description/title/keywords and
# a runnable, schema-qualified example query (VGI112/113/124/126/506).
_SCHEMA_META: dict[str, dict[str, str]] = {
    "datasets": {
        "comment": "scikit-learn toy datasets and synthetic generators as table functions.",
        "title": "Datasets",
        "keywords": json.dumps(["datasets", "toy data", "generators", "sample data", "iris"]),
        "doc_llm": (
            "Built-in datasets as table functions: toy datasets (`iris`, `wine`, `digits`, "
            "`breast_cancer`, `diabetes`, `california_housing`) and synthetic generators "
            "(`make_classification`, `make_regression`, `make_blobs`, `make_moons`, "
            "`make_circles`). Each returns a tidy feature matrix plus a `target`/`cluster` "
            "column, ready to feed `sklearn.models.fit`."
        ),
        "doc_md": (
            "### Datasets\n\n"
            "Load reference data without leaving SQL:\n\n"
            "- **Toy** — `iris`, `wine`, `digits`, `breast_cancer`, `diabetes`, `california_housing`\n"
            "- **Generators** — `make_classification`, `make_regression`, `make_blobs`, "
            "`make_moons`, `make_circles` (column count depends on the arguments)\n\n"
            "Every row carries a `sample_id` plus the named features and target."
        ),
        "example_queries": json.dumps(
            [{"description": "Load the iris dataset", "sql": "SELECT * FROM sklearn.datasets.iris() LIMIT 5"}]
        ),
    },
    "metrics": {
        "comment": "Regression, classification, and clustering metrics as SQL aggregates.",
        "title": "Metrics",
        "keywords": json.dumps(["metrics", "evaluation", "scoring", "regression", "classification"]),
        "doc_llm": (
            "Model-evaluation metrics as aggregates over `(actual, predicted)` columns "
            "(`r2_score`, `accuracy_score`, `f1_score`, `roc_auc_score`, clustering scores, "
            "...) plus table functions for curves and matrices (`confusion_matrix`, "
            "`roc_curve`, `precision_recall_curve`, `calibration_curve`, `silhouette_score`). "
            "Score predictions from `sklearn.models.predict` or any other source."
        ),
        "doc_md": (
            "### Metrics\n\n"
            "Evaluate predictions directly in SQL:\n\n"
            "- **Aggregates** over `(actual, predicted)` — regression, classification, and "
            "clustering scores\n"
            "- **Curves & matrices** as table functions — `confusion_matrix`, `roc_curve`, "
            "`precision_recall_curve`, `calibration_curve`, `silhouette_score`\n\n"
            "Pairs naturally with `sklearn.models.predict` output."
        ),
        "example_queries": json.dumps(
            [
                {
                    "description": "R^2 over inline (actual, predicted) pairs",
                    "sql": (
                        "SELECT sklearn.metrics.r2_score(actual, predicted) "
                        "FROM (VALUES (3.0, 2.5), (5.0, 5.1)) AS t(actual, predicted)"
                    ),
                }
            ]
        ),
    },
    "preprocessing": {
        "comment": "Transforms, encoders, decomposition, clustering, and stored transformers.",
        "title": "Preprocessing",
        "keywords": json.dumps(["preprocessing", "transforms", "scaling", "encoding", "decomposition"]),
        "doc_llm": (
            "Feature preparation and unsupervised transforms: scalers/imputers, ordinal and "
            "one-hot encoders, decomposition (`pca`, `truncated_svd`), clustering, outlier "
            "detection, manifold learning, text vectorizers, and feature selection. "
            "`fit_transformer`/`apply_transform` persist a fitted transformer as a reusable "
            "BLOB, mirroring the model registry."
        ),
        "doc_md": (
            "### Preprocessing\n\n"
            "Prepare and reshape features:\n\n"
            "- **Scale / impute / encode** — scalers, `simple_imputer`, ordinal & one-hot encoders\n"
            "- **Decompose / cluster / detect** — `pca`, `truncated_svd`, clustering, outlier detection\n"
            "- **Text & selection** — `count_vectorizer`, `tfidf_vectorizer`, `select_k_best`, "
            "`variance_threshold`\n"
            "- **Stored** — `fit_transformer` → BLOB, replayed by `apply_transform`"
        ),
        "example_queries": json.dumps(
            [
                {
                    "description": "Standard-scale the iris features",
                    "sql": (
                        "SELECT * FROM sklearn.preprocessing.standard_scaler("
                        "(SELECT sample_id, sepal_length_cm, sepal_width_cm, petal_length_cm, "
                        "petal_width_cm FROM sklearn.datasets.iris()), id := 'sample_id') LIMIT 5"
                    ),
                }
            ]
        ),
    },
    "models": {
        "comment": "Train/predict model registry, cross-validation, inspection, and search.",
        "title": "Models",
        "keywords": json.dumps(["models", "fit", "predict", "cross-validation", "model registry"]),
        "doc_llm": (
            "The core supervised workflow: `fit` returns a model BLOB (and optionally persists "
            "it by `model_name`), `predict` aligns features by name and decodes labels, with "
            "`cross_val_predict`/`cross_val_score`, `partial_dependence`, "
            "`permutation_importance`, `fit_pipeline`, CV splitters, `grid_search`/"
            "`randomized_search`, per-group `fit_model` + `predict_*`, and registry management "
            "(`list_models`/`model_info`/`drop_model`). This is the default schema."
        ),
        "doc_md": (
            "### Models\n\n"
            "Train, evaluate, and serve models — the default schema:\n\n"
            "- **fit → BLOB → predict**, features aligned by name, string labels auto-encoded\n"
            "- **Validate** — `cross_val_predict`, `cross_val_score`, CV splitters\n"
            "- **Inspect** — `partial_dependence`, `permutation_importance`\n"
            "- **Search** — `grid_search`, `randomized_search`; **pipelines**; **per-group** modeling\n"
            "- **Registry** — `list_models`, `model_info`, `drop_model`"
        ),
        "example_queries": json.dumps(
            [
                {
                    "description": "3-fold cross-validated accuracy on iris",
                    "sql": (
                        "SELECT fold, score FROM sklearn.models.cross_val_score("
                        "(SELECT * FROM sklearn.datasets.iris()), "
                        "estimator := 'logistic_regression', target := 'target', cv := 3)"
                    ),
                }
            ]
        ),
    },
    "estimators": {
        "comment": "Typed fit_<estimator> functions with named, documented hyperparameters.",
        "title": "Estimators",
        "keywords": json.dumps(["estimators", "fit", "hyperparameters", "training", "typed"]),
        "doc_llm": (
            "One typed `fit_<estimator>` table function per supported scikit-learn estimator "
            "(random forests, gradient boosting, linear/logistic models, SVMs, kNN, MLPs, "
            "naive Bayes, ...), each exposing that estimator's hyperparameters as named, "
            "typed arguments instead of the generic `fit(estimator := ...)` JSON surface. "
            "Returns the same model BLOB used by `sklearn.models.predict`."
        ),
        "doc_md": (
            "### Estimators\n\n"
            "Typed `fit_<estimator>` functions — one per estimator, with named hyperparameters:\n\n"
            "- e.g. `fit_random_forest_classifier`, `fit_gradient_boosting_regressor`, "
            "`fit_logistic_regression`, `fit_svc`, `fit_mlp_classifier`\n"
            "- Hyperparameters are typed named args (no JSON blob)\n"
            "- Output is a model BLOB — predict with `sklearn.models.predict`"
        ),
        "example_queries": json.dumps(
            [
                {
                    "description": "Fit a logistic-regression classifier on iris",
                    "sql": (
                        "SELECT n_samples, n_features FROM sklearn.estimators.fit_logistic_regression("
                        "(SELECT * FROM sklearn.datasets.iris()), target := 'target')"
                    ),
                }
            ]
        ),
    },
}


def _humanize(name: str) -> str:
    """Title-case a snake_case function name for a display title."""
    return name.replace("_", " ").title()


def _apply_discovery_tags(functions: list[type]) -> None:
    """Inject the per-function discovery tags the catalog-quality linter expects.

    ``vgi.title`` and ``vgi.keywords`` (a JSON array of strings) are derived
    mechanically from each function's existing Meta (display name, categories).
    ``vgi.source_url`` is deliberately NOT set here — it is a catalog-only tag
    (VGI139). The richer ``vgi.doc_llm`` / ``vgi.doc_md`` tags are authored per
    function in the implementation modules and are left untouched here.
    """
    for fn in functions:
        meta = getattr(fn, "Meta", None)
        if meta is None:
            continue
        name = getattr(meta, "name", fn.__name__)
        cats = list(getattr(meta, "categories", []) or [])
        tags = dict(getattr(meta, "tags", {}) or {})
        tags.setdefault("vgi.title", _humanize(name))
        keywords = list(dict.fromkeys(cats or name.split("_")))
        tags.setdefault("vgi.keywords", json.dumps(keywords))
        meta.tags = tags


_apply_discovery_tags(_FUNCTIONS)


def _is_parameterless_table_fn(fn: type) -> bool:
    """True for a table function whose argument dataclass has no fields.

    Such a function always returns the same rows, so it is also exposed as a
    plain table (VGI311) — ``SELECT * FROM schema.name`` without parentheses.
    """
    args = getattr(fn, "FunctionArguments", None)
    return args is not None and dataclasses.is_dataclass(args) and not dataclasses.fields(args)


# Table-specific metadata for the parameterless functions also exposed as tables
# (VGI311). Each carries a table-oriented description, a descriptive title (not a
# restatement of the name), a primary key, and a runnable example — distinct from
# the backing function's documentation.
_TABLE_META: dict[str, dict[str, Any]] = {
    "iris": {
        "title": "Fisher's iris flowers",
        "comment": "150 iris flowers with four sepal/petal measurements (cm) and their species.",
        "keywords": json.dumps(["iris", "flowers", "classification", "toy dataset", "fisher"]),
        "primary_key": (("sample_id",),),
        "doc_llm": (
            "Fisher's classic iris table: 150 rows, one per flower. Columns are `sample_id`, "
            "four numeric measurements in centimetres (`sepal_length_cm`, `sepal_width_cm`, "
            "`petal_length_cm`, `petal_width_cm`), an integer `target` (0/1/2), and the "
            "`target_name` species (setosa/versicolor/virginica). Query it directly — "
            "`SELECT * FROM sklearn.datasets.iris` — for a balanced 3-class toy dataset."
        ),
        "doc_md": (
            "### `iris` table\n\n"
            "150 iris flowers, evenly split across three species:\n\n"
            "- `sample_id` — row id\n"
            "- four measurements in **cm** — `sepal_length_cm`, `sepal_width_cm`, "
            "`petal_length_cm`, `petal_width_cm`\n"
            "- `target` (0–2) and `target_name` (species)"
        ),
        "example_queries": json.dumps(
            [
                {
                    "description": "Row count per species",
                    "sql": "SELECT target_name, count(*) FROM sklearn.datasets.iris GROUP BY target_name",
                }
            ]
        ),
    },
    "wine": {
        "title": "Wine cultivar chemistry",
        "comment": "178 wines with 13 chemical measurements and their cultivar of origin.",
        "keywords": json.dumps(["wine", "chemistry", "classification", "toy dataset", "cultivars"]),
        "primary_key": (("sample_id",),),
        "doc_llm": (
            "The wine-recognition table: 178 rows, one per wine sample, with 13 numeric "
            "chemical-analysis features (alcohol, malic acid, ash, magnesium, phenols, "
            "colour intensity, proline, ...), an integer `target`, and the cultivar "
            "`target_name`. A 3-class dataset for classification over continuous features."
        ),
        "doc_md": (
            "### `wine` table\n\n"
            "178 wines from three cultivars:\n\n"
            "- `sample_id` — row id\n"
            "- 13 chemical features (alcohol, phenols, colour intensity, proline, ...)\n"
            "- `target` (0–2) and `target_name` (cultivar)"
        ),
        "example_queries": json.dumps(
            [
                {
                    "description": "Average alcohol per cultivar",
                    "sql": "SELECT target_name, round(avg(alcohol), 2) FROM sklearn.datasets.wine GROUP BY target_name",
                }
            ]
        ),
    },
    "digits": {
        "title": "Handwritten digit images",
        "comment": "1797 handwritten digits as 64 pixel-intensity features (8x8) plus the digit label.",
        "keywords": json.dumps(["digits", "images", "pixels", "classification", "toy dataset"]),
        "primary_key": (("sample_id",),),
        "doc_llm": (
            "Handwritten-digit table: 1797 rows, one per 8x8 grayscale image, flattened into "
            "64 pixel-intensity columns (`pixel_0_0` ... `pixel_7_7`, each 0–16), with an "
            "integer `target` digit (0–9) and `target_name`. A 10-class image-classification "
            "dataset usable entirely in SQL."
        ),
        "doc_md": (
            "### `digits` table\n\n"
            "1797 handwritten digits (8x8 grayscale):\n\n"
            "- `sample_id` — row id\n"
            "- 64 `pixel_*` columns — intensities 0–16\n"
            "- `target` (0–9) and `target_name`"
        ),
        "example_queries": json.dumps(
            [
                {
                    "description": "Images per digit",
                    "sql": "SELECT target, count(*) FROM sklearn.datasets.digits GROUP BY target ORDER BY target",
                }
            ]
        ),
    },
    "breast_cancer": {
        "title": "Breast-cancer cell diagnostics",
        "comment": "569 tumour samples with 30 cell-nucleus measurements and a benign/malignant label.",
        "keywords": json.dumps(["breast cancer", "diagnostics", "classification", "toy dataset", "binary"]),
        "primary_key": (("sample_id",),),
        "doc_llm": (
            "The Wisconsin breast-cancer table: 569 rows, one per tumour, with 30 numeric "
            "features summarising cell-nucleus geometry (mean/standard-error/worst of radius, "
            "texture, area, concavity, ...), a binary `target` (0 = malignant, 1 = benign), "
            "and `target_name`. A standard binary-classification benchmark."
        ),
        "doc_md": (
            "### `breast_cancer` table\n\n"
            "569 tumour samples, binary outcome:\n\n"
            "- `sample_id` — row id\n"
            "- 30 cell-nucleus features (mean / SE / worst of radius, texture, area, ...)\n"
            "- `target` (0 malignant, 1 benign) and `target_name`"
        ),
        "example_queries": json.dumps(
            [
                {
                    "description": "Class balance",
                    "sql": "SELECT target_name, count(*) FROM sklearn.datasets.breast_cancer GROUP BY target_name",
                }
            ]
        ),
    },
    "california_housing": {
        "title": "California housing prices",
        "comment": "20640 census block groups with 8 features and the median house value (regression).",
        "keywords": json.dumps(["california housing", "regression", "prices", "toy dataset", "census"]),
        "primary_key": (("sample_id",),),
        "doc_llm": (
            "California-housing table: 20,640 rows, one per 1990 census block group, with 8 "
            "numeric features (median income, house age, average rooms/bedrooms, population, "
            "occupancy, latitude, longitude) and a continuous `target` — the median house "
            "value in $100,000s. A large regression dataset."
        ),
        "doc_md": (
            "### `california_housing` table\n\n"
            "20,640 census block groups (regression):\n\n"
            "- `sample_id` — row id\n"
            "- 8 features (median income, house age, rooms, location, ...)\n"
            "- `target` — median house value (in $100k)"
        ),
        "example_queries": json.dumps(
            [
                {
                    "description": "Average value by latitude band",
                    "sql": (
                        "SELECT round(latitude) AS lat, round(avg(target), 2) "
                        "FROM sklearn.datasets.california_housing GROUP BY lat ORDER BY lat"
                    ),
                }
            ]
        ),
    },
    "diabetes": {
        "title": "Diabetes progression",
        "comment": "442 patients with 10 baseline measurements and a one-year disease-progression score.",
        "keywords": json.dumps(["diabetes", "regression", "health", "toy dataset", "progression"]),
        "primary_key": (("sample_id",),),
        "doc_llm": (
            "Diabetes table: 442 rows, one per patient, with 10 mean-centred, scaled baseline "
            "features (age, sex, BMI, blood pressure, and six serum measurements) and a "
            "continuous `target` quantifying disease progression one year after baseline. A "
            "small regression dataset."
        ),
        "doc_md": (
            "### `diabetes` table\n\n"
            "442 patients (regression):\n\n"
            "- `sample_id` — row id\n"
            "- 10 scaled baseline features (age, sex, BMI, BP, 6 serum measures)\n"
            "- `target` — one-year disease-progression score"
        ),
        "example_queries": json.dumps(
            [
                {
                    "description": "Target range",
                    "sql": "SELECT min(target), max(target), round(avg(target), 1) FROM sklearn.datasets.diabetes",
                }
            ]
        ),
    },
    "list_models": {
        "title": "Saved model registry",
        "comment": "One row per model persisted in the registry, with its estimator, task, shape, and score.",
        "keywords": json.dumps(["model registry", "models", "metadata", "catalog", "persistence"]),
        "primary_key": (("model_name",),),
        "doc_llm": (
            "A table of every model saved to the registry (by `fit`/`fit_pipeline`/"
            "`fit_<estimator>` with a `model_name`). One row per model keyed by `model_name`, "
            "with its `estimator`, `task`, `target`, training shape (`n_features`/`n_samples`/"
            "`n_classes`), `train_score`, `sklearn_version`, `created_at`, and the `features` "
            "list. Query it to discover what is available to `predict`."
        ),
        "doc_md": (
            "### `list_models` table\n\n"
            "Every model in the registry, one row each:\n\n"
            "- `model_name` (key), `estimator`, `task`, `target`\n"
            "- `n_features` / `n_samples` / `n_classes`, `train_score`\n"
            "- `sklearn_version`, `created_at`, `features`"
        ),
        "example_queries": json.dumps(
            [
                {
                    "description": "How many models are saved",
                    "sql": "SELECT count(*) AS model_count FROM sklearn.models.list_models",
                }
            ]
        ),
    },
    "list_transformers": {
        "title": "Saved transformer registry",
        "comment": "One row per fitted transformer persisted in the registry, with its kind and shape.",
        "keywords": json.dumps(["transformer registry", "transformers", "metadata", "catalog", "persistence"]),
        "primary_key": (("transformer_name",),),
        "doc_llm": (
            "A table of every transformer saved to the registry by `fit_transformer`. One row "
            "per transformer keyed by `transformer_name`, with its `kind`, input/output widths "
            "(`n_features`/`n_output`), the `features` list, `sklearn_version`, and "
            "`created_at`. Query it to find transformers available to `apply_transform`."
        ),
        "doc_md": (
            "### `list_transformers` table\n\n"
            "Every fitted transformer in the registry, one row each:\n\n"
            "- `transformer_name` (key), `kind`\n"
            "- `n_features` / `n_output`, `features`\n"
            "- `sklearn_version`, `created_at`"
        ),
        "example_queries": json.dumps(
            [
                {
                    "description": "How many transformers are saved",
                    "sql": "SELECT count(*) AS transformer_count FROM sklearn.preprocessing.list_transformers",
                }
            ]
        ),
    },
}


def _function_table(fn: type) -> Table:
    """Expose a parameterless table function as a same-named table (VGI311).

    The table carries its own table-oriented metadata from ``_TABLE_META`` — a
    description, descriptive title, primary key, and example — distinct from the
    backing function's documentation.
    """
    meta = getattr(fn, "Meta")  # noqa: B009 - Meta is a dynamic per-function class
    tm = _TABLE_META[meta.name]
    primary_key = tm["primary_key"]
    # The key columns are inherently non-null (VGI804).
    not_null = tuple(dict.fromkeys(col for cols in primary_key for col in cols))
    return Table(
        name=meta.name,
        function=fn,
        comment=tm["comment"],
        primary_key=primary_key,
        not_null=not_null,
        tags={
            "provider": "scikit-learn",
            "domain": "machine-learning",
            "vgi.title": tm["title"],
            "vgi.keywords": tm["keywords"],
            "vgi.doc_llm": tm["doc_llm"],
            "vgi.doc_md": tm["doc_md"],
            "vgi.example_queries": tm["example_queries"],
        },
    )


def _build_schema(name: str, functions: list[type]) -> Schema:
    """Build a ``Schema`` from its function list and the ``_SCHEMA_META`` entry.

    Parameterless table functions are additionally surfaced as tables of the same
    name that scan the function, so they are usable without parentheses.
    """
    meta = _SCHEMA_META[name]
    tables = [_function_table(fn) for fn in functions if _is_parameterless_table_fn(fn)]
    return Schema(
        name=name,
        comment=meta["comment"],
        tags={
            "provider": "scikit-learn",
            "domain": "machine-learning",
            "vgi.title": meta["title"],
            "vgi.keywords": meta["keywords"],
            "vgi.doc_llm": meta["doc_llm"],
            "vgi.doc_md": meta["doc_md"],
            "vgi.example_queries": meta["example_queries"],
        },
        tables=tables,
        functions=functions,
    )


_SKLEARN_CATALOG = Catalog(
    name="sklearn",
    default_schema=_DEFAULT_SCHEMA,
    comment=_CATALOG_COMMENT,
    tags=_CATALOG_TAGS,
    schemas=[_build_schema(name, functions) for name, functions in _SCHEMA_FUNCTIONS.items()],
)


class SklearnCatalog(ReadOnlyCatalogInterface):
    """Advertises the worker's data + implementation version on ATTACH."""

    catalog = _SKLEARN_CATALOG
    catalog_name = _SKLEARN_CATALOG.name

    def catalogs(self) -> list[CatalogInfo]:
        """Advertise the catalog with its implementation and data versions."""
        return [
            CatalogInfo(
                name=self._effective_catalog_name,
                implementation_version=IMPLEMENTATION_VERSION,
                data_version_spec=DATA_VERSION_SPEC,
                source_url=SOURCE_URL,
                attach_option_specs=[spec.serialize() for spec in self.attach_option_specs],
            )
        ]

    def catalog_attach(self, **kwargs: Any) -> CatalogAttachResult:
        """Resolve the data and implementation versions reported on ATTACH."""
        result = super().catalog_attach(**kwargs)
        return dataclasses.replace(
            result,
            resolved_data_version=DATA_VERSION,
            resolved_implementation_version=IMPLEMENTATION_VERSION,
        )


class SklearnWorker(Worker):
    """Worker process hosting the scikit-learn catalog."""

    catalog = _SKLEARN_CATALOG
    catalog_interface = SklearnCatalog


def _warn_if_ephemeral_state() -> None:
    """Warn when the worker's state dirs look container-local (no volume mounted).

    The published image declares a ``/data`` volume (advertised via the
    ``farm.query.vgi.volumes`` image label) that holds the model registry and the
    shared ``BoundStorage`` SQLite. If the worker runs with those defaults but
    ``/data`` is not actually a mounted volume, models and shared state live on
    the container's writable layer and vanish on ``docker run --rm`` — and are not
    shared across instances. Surface that loudly instead of silently losing data.

    A no-op outside that container shape: it only fires when the state dirs are
    rooted under ``/data`` and ``/proc/mounts`` is readable (a Linux container).
    Never raises — an unmounted run is still valid for ephemeral use.
    """
    sqlite_dir = os.path.dirname(os.environ.get("VGI_WORKER_SQLITE_PATH", ""))
    roots = [p for p in (os.environ.get("SKLEARN_MODELS_DIR", ""), sqlite_dir) if p.startswith("/data")]
    if not roots:
        return
    try:
        with open("/proc/mounts", encoding="utf-8") as fh:  # Linux container only
            mountpoints = {parts[1] for line in fh if len(parts := line.split()) > 1}
    except OSError:
        return
    if "/data" not in mountpoints and not any(r in mountpoints for r in roots):
        log.warning(
            "state directory /data is not a mounted volume: the model registry and "
            "shared BoundStorage are container-local and will NOT persist across "
            "restarts or be shared across worker instances. Mount a volume at /data "
            "(the image advertises this via the 'farm.query.vgi.volumes' label)."
        )


def main() -> None:
    """Run the worker (stdio by default; pass ``--http`` for the HTTP server)."""
    _warn_if_ephemeral_state()
    SklearnWorker.main()


def main_http() -> None:
    """Run the worker over HTTP (injects ``--http`` into the worker CLI)."""
    _warn_if_ephemeral_state()
    argv = sys.argv[1:]
    if "--http" not in argv:
        argv = ["--http", *argv]
    sys.argv = [sys.argv[0], *argv]
    SklearnWorker.main()
