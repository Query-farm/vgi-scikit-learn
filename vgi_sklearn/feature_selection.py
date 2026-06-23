"""Feature selection exposed as scoring tables.

Rather than emit a reduced feature matrix (whose width/columns are data-dependent
and so can't be fixed at bind time), these return one row **per feature** with its
score and a ``selected`` flag. That's the SQL-friendly shape: rank features, or
materialize the chosen subset yourself with ``SELECT`` of the selected columns.

* ``select_k_best`` -- univariate scores (ANOVA F, mutual information, or chi2)
  for each feature against the target, flagging the top ``k``.
* ``variance_threshold`` -- per-feature variance, flagging those above a cutoff
  (an unsupervised filter; no target needed).

Both buffer the whole input (scores need every row), then score once in finalize.
Features are the numeric columns other than ``target``/``id``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import numpy as np
import pyarrow as pa
from vgi.arguments import Arg, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_buffering_function import OutputCollector, TableBufferingParams
from vgi.table_function import BindParams

from .buffering import DrainState, SinkBuffer, input_schema_of, matrix
from .schema_utils import field as sfield


def _features_excluding(input_schema: pa.Schema, *exclude: str) -> list[str]:
    drop = {e for e in exclude if e}
    return [n for n in input_schema.names if n not in drop]


# ===========================================================================
# select_k_best (univariate scores vs the target)
# ===========================================================================

# score function name -> (callable factory, target is integer labels)
_SCORE_FUNCS: dict[str, bool] = {
    "f_classif": True,
    "chi2": True,
    "mutual_info_classif": True,
    "f_regression": False,
    "mutual_info_regression": False,
}


def _score_func(name: str) -> Any:
    from sklearn.feature_selection import (
        chi2,
        f_classif,
        f_regression,
        mutual_info_classif,
        mutual_info_regression,
    )

    return {
        "f_classif": f_classif,
        "chi2": chi2,
        "mutual_info_classif": mutual_info_classif,
        "f_regression": f_regression,
        "mutual_info_regression": mutual_info_regression,
    }[name]


@dataclass(slots=True, frozen=True)
class SelectKBestArgs:
    data: Annotated[TableInput, Arg(0, doc="Table of numeric features + the target column.")]
    target: Annotated[str, Arg("target", default="", doc="Name of the target/label column (required).")]
    id: Annotated[str, Arg("id", default="", doc="Optional id column to exclude from features.")]
    k: Annotated[int, Arg("k", default=10, doc="Number of top features to flag as selected (capped at n_features).")]
    score_func: Annotated[
        str,
        Arg(
            "score_func",
            default="f_classif",
            doc="f_classif, f_regression, mutual_info_classif, mutual_info_regression, or chi2.",
        ),
    ]


_SELECT_SCHEMA = pa.schema(
    [
        sfield("feature", pa.string(), "Feature column name.", nullable=False),
        sfield("score", pa.float64(), "Univariate score (higher = more informative).", nullable=False),
        sfield("p_value", pa.float64(), "p-value of the score (NULL for mutual-information scorers)."),
        sfield("selected", pa.bool_(), "True if among the top-k features.", nullable=False),
    ]
)


class SelectKBest(SinkBuffer[SelectKBestArgs, DrainState]):
    FunctionArguments: ClassVar[type] = SelectKBestArgs

    class Meta:
        name = "select_k_best"
        description = "Univariate feature scores vs. the target, flagging the top k"
        categories = ["preprocessing", "feature-selection"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT feature, score, selected FROM sklearn.select_k_best("
                    "(SELECT sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm, target "
                    "FROM sklearn.iris()), target := 'target', k := 2) ORDER BY score DESC"
                ),
                description="Rank iris features by ANOVA F against the species label",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[SelectKBestArgs]) -> BindResponse:
        a = params.args
        if not a.target:
            raise ValueError("select_k_best requires 'target' (the label column name, e.g. target := 'label')")
        if a.score_func not in _SCORE_FUNCS:
            raise ValueError(f"unknown score_func {a.score_func!r}; choose one of: {', '.join(sorted(_SCORE_FUNCS))}")
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        if a.target not in input_schema.names:
            raise ValueError(f"target column {a.target!r} not found in input; columns: {', '.join(input_schema.names)}")
        return BindResponse(output_schema=_SELECT_SCHEMA)

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[SelectKBestArgs]
    ) -> DrainState:
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[SelectKBestArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        if state.done:
            out.finish()
            return
        state.done = True

        from sklearn.feature_selection import SelectKBest as SkSelectKBest

        a = params.args
        input_schema = input_schema_of(params)
        feats = _features_excluding(input_schema, a.target, a.id)

        table = cls.buffered_table(params, input_schema)
        if table is None or table.num_rows == 0:
            raise ValueError("select_k_best received no rows")

        x = matrix(table, feats)
        y = np.asarray(table.column(a.target).to_numpy(zero_copy_only=False))
        y = np.rint(y.astype(float)).astype(int) if _SCORE_FUNCS[a.score_func] else y.astype(float)

        k = max(1, min(a.k, len(feats)))
        selector = SkSelectKBest(_score_func(a.score_func), k=k).fit(x, y)
        support = selector.get_support()
        pvalues = selector.pvalues_

        out.emit(
            pa.RecordBatch.from_pydict(
                {
                    "feature": list(feats),
                    "score": [float(s) for s in selector.scores_],
                    "p_value": [None if pvalues is None else float(pvalues[j]) for j in range(len(feats))],
                    "selected": [bool(v) for v in support],
                },
                schema=params.output_schema,
            )
        )


# ===========================================================================
# variance_threshold (unsupervised low-variance filter)
# ===========================================================================


@dataclass(slots=True, frozen=True)
class VarianceThresholdArgs:
    data: Annotated[TableInput, Arg(0, doc="Table of numeric features.")]
    id: Annotated[str, Arg("id", default="", doc="Optional id column to exclude from features.")]
    threshold: Annotated[float, Arg("threshold", default=0.0, doc="Keep features with variance strictly above this.")]


_VARIANCE_SCHEMA = pa.schema(
    [
        sfield("feature", pa.string(), "Feature column name.", nullable=False),
        sfield("variance", pa.float64(), "Variance of the feature.", nullable=False),
        sfield("selected", pa.bool_(), "True if variance exceeds the threshold.", nullable=False),
    ]
)


class VarianceThreshold(SinkBuffer[VarianceThresholdArgs, DrainState]):
    FunctionArguments: ClassVar[type] = VarianceThresholdArgs

    class Meta:
        name = "variance_threshold"
        description = "Per-feature variance, flagging features above a threshold (unsupervised filter)"
        categories = ["preprocessing", "feature-selection"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT feature, variance, selected FROM sklearn.variance_threshold("
                    "(SELECT sample_id, sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm "
                    "FROM sklearn.iris()), id := 'sample_id', threshold := 0.5) ORDER BY variance DESC"
                ),
                description="Flag iris features whose variance exceeds 0.5",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[VarianceThresholdArgs]) -> BindResponse:
        assert params.bind_call.input_schema is not None
        return BindResponse(output_schema=_VARIANCE_SCHEMA)

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[VarianceThresholdArgs]
    ) -> DrainState:
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[VarianceThresholdArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        if state.done:
            out.finish()
            return
        state.done = True

        a = params.args
        input_schema = input_schema_of(params)
        feats = _features_excluding(input_schema, a.id)

        table = cls.buffered_table(params, input_schema)
        if table is None or table.num_rows == 0:
            raise ValueError("variance_threshold received no rows")

        x = matrix(table, feats)
        variances = x.var(axis=0)
        out.emit(
            pa.RecordBatch.from_pydict(
                {
                    "feature": list(feats),
                    "variance": [float(v) for v in variances],
                    "selected": [bool(v > a.threshold) for v in variances],
                },
                schema=params.output_schema,
            )
        )


FEATURE_SELECTION_FUNCTIONS: list[type] = [SelectKBest, VarianceThreshold]
