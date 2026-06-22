"""Unit tests for confusion_matrix / silhouette compute logic.

Full buffering lifecycle is covered by test/sql/sklearn_polish.test.
"""

from __future__ import annotations

from collections import Counter

from sklearn import metrics as skm
from sklearn.datasets import make_blobs


def test_confusion_counts_match_sklearn() -> None:
    yt = [0, 0, 1, 1, 2, 2, 0, 1]
    yp = [0, 1, 1, 1, 2, 0, 0, 2]
    counts = Counter(zip(yt, yp, strict=False))
    cm = skm.confusion_matrix(yt, yp)
    for (a, p), c in counts.items():
        assert cm[a, p] == c


def test_silhouette_on_separated_blobs_is_high() -> None:
    x, labels = make_blobs(n_samples=150, centers=3, cluster_std=0.5, random_state=0)
    score = skm.silhouette_score(x, labels)
    assert score > 0.5
