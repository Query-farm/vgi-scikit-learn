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


def test_roc_curve_matches_sklearn() -> None:
    import numpy as np

    from vgi_sklearn.table_metrics import RocCurve

    yt = np.array([0, 0, 1, 1, 0, 1])
    ys = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])
    out = RocCurve.curve(yt, ys)
    fpr, tpr, thr = skm.roc_curve(yt, ys)
    assert out["fpr"] == [float(v) for v in fpr]
    assert out["tpr"] == [float(v) for v in tpr]
    # the leading inf threshold is NULL'd, the rest match
    assert out["threshold"][0] is None
    assert out["threshold"][1:] == [float(t) for t in thr[1:]]


def test_precision_recall_curve_pads_last_threshold() -> None:
    import numpy as np

    from vgi_sklearn.table_metrics import PrecisionRecallCurve

    yt = np.array([0, 0, 1, 1, 0, 1])
    ys = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])
    out = PrecisionRecallCurve.curve(yt, ys)
    precision, recall, thresholds = skm.precision_recall_curve(yt, ys)
    assert len(out["precision"]) == len(precision)
    # precision/recall have one more point than thresholds; the last threshold is NULL
    assert out["threshold"][-1] is None
    assert len(out["threshold"]) == len(out["precision"])
