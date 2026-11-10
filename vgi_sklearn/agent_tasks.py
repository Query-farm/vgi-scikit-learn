"""Public agent-evaluation tasks for ``vgi-lint simulate`` (VGI152/VGI520).

Only each task's name and prompt are published in the catalog's ``vgi.agent_test_tasks`` tag.
The reference SQL and grading options are private and live in ``vgi-agent-tests.yaml`` at the
repo root (VGI416), merged by task name: add or rename a task in both places together.
"""

from __future__ import annotations

AGENT_TEST_TASKS: list[dict[str, str]] = [
    {
        "name": "iris_species_counts",
        "prompt": (
            "Using the scikit-learn worker's built-in iris dataset, how many flowers of each species are "
            "there? Return one row per species name with its count."
        ),
    },
    {
        "name": "toy_dataset_sizes",
        "prompt": (
            "Report how many rows each of the worker's built-in toy datasets has: wine, digits, breast "
            "cancer, diabetes, and California housing. Return one row per dataset with its row count."
        ),
    },
    {
        "name": "synthetic_dataset_sizes",
        "prompt": (
            "Generate synthetic data with each of the worker's generators \u2014 a classification problem, a "
            "regression problem, Gaussian blobs, two interleaving moons, and two concentric circles \u2014 120"
            " samples each with random seed 0. Return one row per generator with the number of rows "
            "produced."
        ),
    },
    {
        "name": "inline_r2_score",
        "prompt": (
            "Given these (actual, predicted) pairs: (3.0, 2.5), (5.0, 5.1), (7.0, 6.8), (9.0, 9.4), "
            "compute the coefficient of determination (R squared) of the predictions, rounded to 4 "
            "decimal places."
        ),
    },
    {
        "name": "regression_error_metrics",
        "prompt": (
            "For the (actual, predicted) pairs (3.0, 2.5), (5.0, 5.1), (7.0, 6.8), (9.0, 9.4), report in "
            "one row, each rounded to 4 decimal places: the mean absolute error, mean squared error, root"
            " mean squared error, median absolute error, maximum error, explained variance score, mean "
            "absolute percentage error, mean squared log error, and mean pinball loss (median quantile)."
        ),
    },
    {
        "name": "inline_accuracy",
        "prompt": (
            "A classifier produced these (true label, predicted label) pairs: (0, 0), (1, 1), (1, 0), (0,"
            " 0), (1, 1). What fraction of predictions are correct? Round to 2 decimal places."
        ),
    },
    {
        "name": "binary_label_metrics",
        "prompt": (
            "A binary classifier produced these (true label, predicted label) pairs: (0, 0), (1, 1), (1, "
            "0), (0, 0), (1, 1), (0, 1). Report in one row, each rounded to 4 decimal places: balanced "
            "accuracy, precision, recall, F1, Jaccard index, Matthews correlation coefficient, Cohen's "
            "kappa, Hamming loss, and zero-one loss."
        ),
    },
    {
        "name": "probability_metrics",
        "prompt": (
            "A model gave these (true label, predicted positive-class probability) pairs: (0, 0.1), (1, "
            "0.9), (0, 0.4), (1, 0.6), (1, 0.35), (0, 0.2). Report in one row, each rounded to 4 decimal "
            "places: ROC AUC, average precision, Brier score, and log loss."
        ),
    },
    {
        "name": "cluster_agreement_metrics",
        "prompt": (
            "Compare a clustering against ground truth for these (true class, assigned cluster) pairs: "
            "(0, 1), (0, 1), (1, 0), (1, 0), (2, 2), (2, 1). Report in one row, each rounded to 4 decimal"
            " places: adjusted Rand index, adjusted mutual information, normalized mutual information, "
            "homogeneity, completeness, V-measure, and the Fowlkes-Mallows index."
        ),
    },
    {
        "name": "confusion_matrix_cells",
        "prompt": (
            "For the (true label, predicted label) pairs (0, 0), (1, 1), (1, 0), (0, 0), (1, 1), (0, 1), "
            "build the confusion matrix and return one row per (actual, predicted) cell with its count."
        ),
    },
    {
        "name": "curve_point_counts",
        "prompt": (
            "For the (true label, predicted probability) pairs (0, 0.1), (1, 0.9), (0, 0.4), (1, 0.6), "
            "(1, 0.35), (0, 0.2), how many points are on the ROC curve, on the precision-recall curve, "
            "and on a 5-bin calibration curve? Return one row per curve with its point count."
        ),
    },
    {
        "name": "silhouette_of_two_clusters",
        "prompt": (
            "Points (id, x1, x2, cluster): (1, 0.1, 0.2, 0), (2, 0.2, 0.1, 0), (3, 5.0, 5.1, 1), (4, 5.1,"
            " 5.0, 1). What is the mean silhouette coefficient of this clustering, rounded to 3 decimal "
            "places?"
        ),
    },
    {
        "name": "sepal_length_after_scaling",
        "prompt": (
            "Transform the iris sepal length and sepal width columns with each of these preprocessing "
            "steps: standard scaling, min-max scaling, robust scaling, max-abs scaling, row "
            "normalization, a power transform, a quantile transform (100 quantiles), binarizing at 5.0, "
            "discretizing into 4 bins, and mean imputation. Return one row per step with the largest "
            "transformed sepal length, rounded to 3 decimal places."
        ),
    },
    {
        "name": "polynomial_interaction_term",
        "prompt": (
            "Expand the iris sepal length and sepal width into degree-2 polynomial features. What is the "
            "largest value of the sepal length x sepal width interaction term, rounded to 2 decimal "
            "places?"
        ),
    },
    {
        "name": "species_encodings",
        "prompt": (
            "Encode the iris species name three ways \u2014 as an ordinal integer code, as one-hot indicators,"
            " and by its rate of a binary is-setosa target \u2014 and report how many distinct encoded values "
            "or indicator categories each produces. Return one row per encoder."
        ),
    },
    {
        "name": "embedding_row_counts",
        "prompt": (
            "Reduce the four iris flower measurements to 2 dimensions with PCA, truncated SVD, t-SNE, "
            "Isomap, spectral embedding, and MDS. Return one row per method with the number of embedded "
            "rows it produced."
        ),
    },
    {
        "name": "iris_kmeans_cluster_sizes",
        "prompt": (
            "Cluster the iris flowers into 3 groups with k-means using only the four flower measurements."
            " Return the size of each cluster, largest first."
        ),
    },
    {
        "name": "clustering_label_counts",
        "prompt": (
            "Cluster the four iris flower measurements with agglomerative clustering, spectral "
            "clustering, BIRCH, and mini-batch k-means (3 clusters each), a 3-component Gaussian mixture,"
            " mean shift, DBSCAN (eps 0.5), and OPTICS (min_samples 5). Return one row per method with "
            "the number of distinct labels it assigns (counting noise as a label)."
        ),
    },
    {
        "name": "outlier_counts",
        "prompt": (
            "Flag outliers among the four iris flower measurements with an isolation forest, the local "
            "outlier factor, and an elliptic envelope (each expecting 10% contamination), and a one-class"
            " SVM with nu 0.1. Return one row per detector with how many flowers it flags."
        ),
    },
    {
        "name": "text_vocabulary",
        "prompt": (
            "For three documents \u2014 'the cat sat on the mat', 'the dog sat on the log', and 'a quick brown"
            " fox' \u2014 how many times does each word occur across all documents, and which word has the "
            "highest TF-IDF weight in the third document? Return the total count of the word 'the' and "
            "that top third-document word."
        ),
    },
    {
        "name": "selected_iris_features",
        "prompt": (
            "Which iris measurements matter? Return the two features chosen by univariate ANOVA F-test "
            "selection (k = 2) against the species target, and separately the features whose variance "
            "exceeds 0.5. Return one row per (method, feature)."
        ),
    },
    {
        "name": "classifier_training_accuracy",
        "prompt": (
            "Train every classifier the worker exposes as a typed fit function on the iris measurements "
            "to predict the species code, with default hyperparameters, and report each one's training "
            "accuracy rounded to 2 decimal places. Return one row per estimator."
        ),
    },
    {
        "name": "regressor_training_r2",
        "prompt": (
            "Train every regressor the worker exposes as a typed fit function on the diabetes dataset to "
            "predict disease progression, with default hyperparameters, and report each one's in-sample R"
            " squared rounded to 2 decimal places. Return one row per estimator."
        ),
    },
    {
        "name": "iris_cv_accuracy",
        "prompt": (
            "Estimate how well a logistic-regression classifier predicts the iris species code (the "
            "integer target) from the four flower measurements, using 3-fold cross-validation. Report the"
            " mean fold accuracy rounded to 2 decimal places."
        ),
    },
    {
        "name": "out_of_fold_accuracy",
        "prompt": (
            "Using 5-fold out-of-fold predictions from a logistic-regression classifier on the four iris "
            "measurements, what fraction of flowers get their species code right? Round to 2 decimal "
            "places."
        ),
    },
    {
        "name": "cv_split_sizes",
        "prompt": (
            "Split the 150 iris rows for cross-validation four ways: plain 5-fold, 5-fold stratified by "
            "species, 3-fold grouped by species, and a 5-split forward-chaining time-series split (in "
            "sample_id order). Return one row per method with the number of test rows in its first fold "
            "or split."
        ),
    },
    {
        "name": "stored_model_accuracy",
        "prompt": (
            "Train a random-forest classifier on the four iris measurements to predict the species code, "
            "save it in the model registry as 'agent_iris_rf', then use the stored model to predict every"
            " iris flower and report the fraction predicted correctly, rounded to 2 decimal places."
        ),
    },
    {
        "name": "most_important_feature",
        "prompt": (
            "Train a random-forest classifier on the four iris measurements (save it as "
            "'agent_iris_rf_pi'), then use permutation importance on the same data to find which "
            "measurement the model relies on most. Return that feature's name."
        ),
    },
    {
        "name": "bmi_partial_dependence",
        "prompt": (
            "Train a random-forest regressor on the diabetes dataset (save it as 'agent_diabetes_rf'). "
            "Using partial dependence over a 10-point grid of BMI, does the model's average predicted "
            "disease progression rise with BMI? Return true or false."
        ),
    },
    {
        "name": "model_registry_lifecycle",
        "prompt": (
            "Train a logistic regression on the four iris measurements and store it as 'agent_tmp_model'."
            " Confirm it is listed in the model registry and check how many features it was trained on, "
            "then delete it. Report whether the delete succeeded."
        ),
    },
    {
        "name": "scaled_pca_pipeline",
        "prompt": (
            "Fit a pipeline that standard-scales the four iris measurements, reduces them to 2 principal "
            "components, and then fits a logistic regression on the species code. What is its training "
            "accuracy, rounded to 2 decimal places?"
        ),
    },
    {
        "name": "forest_hyperparameter_search",
        "prompt": (
            "Tune a random-forest classifier on the four iris measurements: run an exhaustive grid search"
            " over n_estimators in (50, 100) and max_depth in (2, 4), and a randomized search drawing 3 "
            "combinations from the same grid with seed 0. Return one row per search with its best mean "
            "cross-validated accuracy, rounded to 2 decimal places."
        ),
    },
    {
        "name": "per_region_churn_models",
        "prompt": (
            "Customers (region, tenure, spend, churned): ('east', 12, 30.0, 0), ('east', 3, 80.0, 1), "
            "('east', 8, 45.0, 0), ('east', 2, 95.0, 1), ('west', 24, 20.0, 0), ('west', 1, 90.0, 1), "
            "('west', 18, 35.0, 0), ('west', 4, 85.0, 1). Fit one logistic-regression churn model per "
            "region from tenure and spend, score every customer with their own region's model, and report"
            " per region how many customers are predicted to churn, plus the average predicted churn "
            "probability rounded to 1 decimal place."
        ),
    },
    {
        "name": "stored_scaler_roundtrip",
        "prompt": (
            "Fit a standard scaler on the iris sepal length and width and save it as the transformer "
            "'agent_sc'. Apply the stored scaler to the iris data and check the mean scaled sepal length,"
            " confirm 'agent_sc' is listed among the stored transformers, then delete it. Report whether "
            "the delete succeeded."
        ),
    },
]
