"""Example queries that combine several of the worker's functions in one query.

Most per-function examples show one function in isolation. These show the
combinations a real user writes: a clustering scored against the true labels
with a metric aggregate, out-of-fold predictions tabulated into a confusion
matrix, a scaler piped into PCA, a clustering fed to ``silhouette_score``. Each
SQL is defined once here and referenced from both the driving function's
``Meta.examples`` (the agent simulator only shows per-function examples) and the
schema-level ``vgi.example_queries`` list in ``worker.py``.

All of them run offline on the built-in iris dataset and are deterministic
(``kmeans`` defaults to ``random_state := 0``; cross-validation folds are not
shuffled).
"""

from __future__ import annotations

from vgi.metadata import FunctionExample

_IRIS_FEATURES = (
    "SELECT sample_id, sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm FROM sklearn.datasets.iris()"
)

# kmeans + adjusted_rand_score/homogeneity_score/completeness_score + iris.
# Result: adjusted_rand 0.73, homogeneity 0.751, completeness 0.765.
KMEANS_VS_SPECIES = FunctionExample(
    sql=(
        "SELECT round(sklearn.metrics.adjusted_rand_score(i.target, k.cluster), 3) AS adjusted_rand, "
        "round(sklearn.metrics.homogeneity_score(i.target, k.cluster), 3) AS homogeneity, "
        "round(sklearn.metrics.completeness_score(i.target, k.cluster), 3) AS completeness "
        f"FROM sklearn.preprocessing.kmeans(({_IRIS_FEATURES}), id := 'sample_id', n_clusters := 3) AS k "
        "JOIN sklearn.datasets.iris AS i USING (sample_id)"
    ),
    description=(
        "Score 3-cluster k-means on the iris measurements against the true species: adjusted Rand index, "
        "homogeneity, and completeness of the clusters (one row)"
    ),
)

# cross_val_predict + confusion_matrix + iris.
# Result: setosa all right; 3 versicolor -> virginica, 1 virginica -> versicolor.
OOF_CONFUSION_MATRIX = FunctionExample(
    sql=(
        "SELECT actual, predicted, count FROM sklearn.metrics.confusion_matrix("
        "(SELECT i.target AS species, p.prediction AS predicted_species "
        "FROM sklearn.models.cross_val_predict("
        "(SELECT sample_id, sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm, target "
        "FROM sklearn.datasets.iris()), estimator := 'logistic_regression', target := 'target', id := 'sample_id') "
        "AS p JOIN sklearn.datasets.iris AS i USING (sample_id)), "
        "actual := 'species', predicted := 'predicted_species') ORDER BY actual, predicted"
    ),
    description=(
        "Confusion matrix of 5-fold out-of-fold logistic-regression predictions on iris: each prediction "
        "joined back to its true species, one row per (actual, predicted) species code with its count"
    ),
)

# standard_scaler -> pca + iris.
# Result: setosa -2.22, versicolor 0.5, virginica 1.73 on component 1.
SCALED_PCA_BY_SPECIES = FunctionExample(
    sql=(
        "SELECT i.target_name, round(avg(p.component_1), 2) AS mean_pc1, round(avg(p.component_2), 2) AS mean_pc2 "
        "FROM sklearn.preprocessing.pca((SELECT * FROM sklearn.preprocessing.standard_scaler("
        f"({_IRIS_FEATURES}), id := 'sample_id')), id := 'sample_id', n_components := 2) AS p "
        "JOIN sklearn.datasets.iris AS i USING (sample_id) GROUP BY i.target_name ORDER BY mean_pc1"
    ),
    description=(
        "Standardize the iris measurements, project them onto 2 principal components, and average each "
        "component per species (the first component separates the species)"
    ),
)

# kmeans + silhouette_score + iris.
# Result: 0.553.
KMEANS_SILHOUETTE = FunctionExample(
    sql=(
        "SELECT round(silhouette_score, 3) AS silhouette FROM sklearn.metrics.silhouette_score("
        "(SELECT f.*, k.cluster "
        f"FROM sklearn.preprocessing.kmeans(({_IRIS_FEATURES}), id := 'sample_id', n_clusters := 3) AS k "
        f"JOIN ({_IRIS_FEATURES}) AS f USING (sample_id)), label := 'cluster', id := 'sample_id')"
    ),
    description=(
        "Silhouette score of a 3-cluster k-means on the iris measurements: the cluster labels joined back "
        "to the feature columns they were computed from (one row)"
    ),
)


def as_query(example: FunctionExample) -> dict[str, str]:
    """Render an example as a ``vgi.example_queries`` entry (``{description, sql}``)."""
    return {"description": example.description, "sql": example.sql}
