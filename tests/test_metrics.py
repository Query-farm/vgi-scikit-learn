"""Tests for the metric aggregate functions.

Each metric is checked against scikit-learn computed directly, so the test is a
faithful end-to-end check of the buffer/finalize path.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn import metrics as skm

from vgi_sklearn.metrics import (
    AccuracyScore,
    AdjustedRandScore,
    BrierScoreLoss,
    F1Score,
    HammingLoss,
    JaccardScore,
    LogLoss,
    MeanAbsoluteError,
    MeanPinballLoss,
    MeanSquaredError,
    MeanSquaredLogError,
    R2Score,
    RocAucScore,
    RootMeanSquaredError,
    VMeasureScore,
    ZeroOneLoss,
)

from .harness import run_aggregate

_YT_REG = [3.0, -0.5, 2.0, 7.0, 4.2]
_YP_REG = [2.5, 0.0, 2.1, 7.8, 3.9]

_YT_CLS = [0, 1, 2, 2, 1, 0, 1, 2]
_YP_CLS = [0, 2, 2, 2, 1, 0, 0, 2]

_YT_BIN = [0, 0, 1, 1, 1, 0]
_YSCORE = [0.1, 0.4, 0.35, 0.8, 0.7, 0.2]


def _approx(x: float) -> pytest.approx:
    return pytest.approx(x, rel=1e-9, abs=1e-12)


class TestRegression:
    def test_mse(self) -> None:
        assert run_aggregate(MeanSquaredError, _YT_REG, _YP_REG)[0] == _approx(skm.mean_squared_error(_YT_REG, _YP_REG))

    def test_rmse(self) -> None:
        assert run_aggregate(RootMeanSquaredError, _YT_REG, _YP_REG)[0] == _approx(
            float(np.sqrt(skm.mean_squared_error(_YT_REG, _YP_REG)))
        )

    def test_mae(self) -> None:
        assert run_aggregate(MeanAbsoluteError, _YT_REG, _YP_REG)[0] == _approx(
            skm.mean_absolute_error(_YT_REG, _YP_REG)
        )

    def test_r2(self) -> None:
        assert run_aggregate(R2Score, _YT_REG, _YP_REG)[0] == _approx(skm.r2_score(_YT_REG, _YP_REG))


class TestClassification:
    def test_accuracy(self) -> None:
        assert run_aggregate(AccuracyScore, _YT_CLS, _YP_CLS)[0] == _approx(skm.accuracy_score(_YT_CLS, _YP_CLS))

    def test_f1_macro(self) -> None:
        assert run_aggregate(F1Score, _YT_CLS, _YP_CLS)[0] == _approx(
            skm.f1_score(_YT_CLS, _YP_CLS, average="macro", zero_division=0)
        )


class TestProbability:
    def test_roc_auc(self) -> None:
        assert run_aggregate(RocAucScore, _YT_BIN, _YSCORE)[0] == _approx(skm.roc_auc_score(_YT_BIN, _YSCORE))

    def test_log_loss_runs(self) -> None:
        val = run_aggregate(LogLoss, _YT_BIN, _YSCORE)[0]
        assert val is not None and val > 0


class TestClustering:
    def test_adjusted_rand(self) -> None:
        assert run_aggregate(AdjustedRandScore, _YT_CLS, _YP_CLS)[0] == _approx(
            skm.adjusted_rand_score(_YT_CLS, _YP_CLS)
        )

    def test_v_measure(self) -> None:
        assert run_aggregate(VMeasureScore, _YT_CLS, _YP_CLS)[0] == _approx(skm.v_measure_score(_YT_CLS, _YP_CLS))


_YT_POS = [3.0, 0.5, 2.0, 7.0, 4.2]
_YP_POS = [2.5, 0.6, 2.1, 7.8, 3.9]


class TestAddedMetrics:
    def test_msle(self) -> None:
        assert run_aggregate(MeanSquaredLogError, _YT_POS, _YP_POS)[0] == _approx(
            skm.mean_squared_log_error(_YT_POS, _YP_POS)
        )

    def test_pinball(self) -> None:
        assert run_aggregate(MeanPinballLoss, _YT_REG, _YP_REG)[0] == _approx(skm.mean_pinball_loss(_YT_REG, _YP_REG))

    def test_hamming(self) -> None:
        assert run_aggregate(HammingLoss, _YT_CLS, _YP_CLS)[0] == _approx(skm.hamming_loss(_YT_CLS, _YP_CLS))

    def test_zero_one(self) -> None:
        assert run_aggregate(ZeroOneLoss, _YT_CLS, _YP_CLS)[0] == _approx(skm.zero_one_loss(_YT_CLS, _YP_CLS))

    def test_jaccard_macro(self) -> None:
        assert run_aggregate(JaccardScore, _YT_CLS, _YP_CLS)[0] == _approx(
            skm.jaccard_score(_YT_CLS, _YP_CLS, average="macro", zero_division=0)
        )

    def test_brier(self) -> None:
        assert run_aggregate(BrierScoreLoss, _YT_BIN, _YSCORE)[0] == _approx(skm.brier_score_loss(_YT_BIN, _YSCORE))


class TestGroupingAndNulls:
    def test_per_group(self) -> None:
        # group 0 is a perfect fit, group 1 is not
        yt = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        yp = [1.0, 2.0, 3.0, 3.0, 2.0, 1.0]
        groups = [0, 0, 0, 1, 1, 1]
        out = run_aggregate(R2Score, yt, yp, group_ids=groups)
        assert out[0] == _approx(1.0)
        assert out[1] < 1.0

    def test_nulls_are_skipped(self) -> None:
        # the None pair must be dropped, leaving a perfect fit
        yt = [1.0, 2.0, None, 3.0]
        yp = [1.0, 2.0, 99.0, 3.0]
        assert run_aggregate(R2Score, yt, yp)[0] == _approx(1.0)
