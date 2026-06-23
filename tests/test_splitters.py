"""Unit tests for splitter helpers + fold-assignment logic.

The full buffering lifecycle is covered by test/sql/sklearn_splitters.test.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from vgi_sklearn.splitters import GroupKFold, KFold, StratifiedKFold, _require_id


def test_require_id_errors_when_empty() -> None:
    schema = pa.schema([pa.field("a", pa.int64())])
    with pytest.raises(ValueError, match="requires 'id'"):
        _require_id(schema, "", "kfold")


def test_require_id_errors_when_absent() -> None:
    schema = pa.schema([pa.field("a", pa.int64())])
    with pytest.raises(ValueError, match="not found"):
        _require_id(schema, "missing", "kfold")


def _assign(splitter, n, y=None, groups=None):
    folds = [None] * n
    x = np.zeros((n, 1))
    for f, (_, test_idx) in enumerate(splitter.split(x, y, groups)):
        for i in test_idx:
            folds[i] = f
    return folds


class TestFoldAssignment:
    def test_kfold_partitions_all_rows(self) -> None:
        folds = _assign(KFold._make_splitter(type("A", (), {"n_splits": 5, "shuffle": False, "random_state": 0})()), 50)
        assert None not in folds
        assert set(folds) == {0, 1, 2, 3, 4}

    def test_stratified_balances_classes(self) -> None:
        y = np.array([0, 1] * 25)
        args = type("A", (), {"n_splits": 5, "shuffle": False, "random_state": 0})()
        folds = _assign(StratifiedKFold._make_splitter(args), 50, y=y)
        # each fold should hold an equal class split (5 of each)
        for f in set(folds):
            classes = [y[i] for i in range(50) if folds[i] == f]
            assert classes.count(0) == classes.count(1)

    def test_group_keeps_groups_together(self) -> None:
        groups = [i % 4 for i in range(40)]
        args = type("A", (), {"n_splits": 4})()
        folds = _assign(GroupKFold._make_splitter(args), 40, groups=groups)
        # every row of a group lands in the same fold
        for g in set(groups):
            gf = {folds[i] for i in range(40) if groups[i] == g}
            assert len(gf) == 1
