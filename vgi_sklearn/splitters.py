"""Cross-validation splitters as table functions.

These assign each row a fold so you can run **custom evaluation in pure SQL**
(``GROUP BY fold``, train on ``fold <> f`` / test on ``fold = f``) instead of
being limited to the built-in ``cv :=`` of fit/grid_search:

* ``kfold`` / ``stratified_kfold`` / ``group_kfold`` -- one row per input row,
  ``(id, fold)``, where ``fold`` is the test fold that row belongs to.
* ``timeseries_split`` -- expanding-window splits where a row can be in several
  splits' training sets, so it emits ``(split, id, role)`` long.

All buffer the input (folds are assigned over the whole set) and carry an ``id``
column so the result joins back to your data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Protocol

import numpy as np
import pyarrow as pa
from vgi.arguments import Arg, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_buffering_function import TableBufferingParams
from vgi.table_function import BindParams
from vgi_rpc.rpc import OutputCollector

from .buffering import DrainState, SinkBuffer, input_schema_of
from .schema_utils import columns_md_rows
from .schema_utils import field as sfield


class _HasId(Protocol):
    """Splitter arguments that carry an ``id`` column name."""

    @property
    def id(self) -> str:
        """The id column name carried onto each output row."""
        ...


_FOLD_MD = columns_md_rows(
    [
        ("<id>", "(input id type)", "The id column, carried through from the input."),
        ("fold", "BIGINT", "Test fold this row belongs to."),
    ]
)


def _require_id(input_schema: pa.Schema, id_col: str, fn: str) -> pa.Field:
    if not id_col:
        raise ValueError(f"{fn} requires 'id' (a column carried onto each row, e.g. id := 'row_id')")
    if id_col not in input_schema.names:
        raise ValueError(f"id column {id_col!r} not found in input; columns: {', '.join(input_schema.names)}")
    return input_schema.field(id_col)


class _FoldFunction[TArgs: _HasId](SinkBuffer[TArgs, DrainState]):
    """Base for splitters that map each row to exactly one test fold ``(id, fold)``."""

    @classmethod
    def _make_splitter(cls, args: Any) -> Any:
        raise NotImplementedError

    @classmethod
    def _y_groups(cls, table: pa.Table, args: Any) -> tuple[np.ndarray | None, list[Any] | None]:
        return None, None

    @classmethod
    def on_bind(cls, params: BindParams[TArgs]) -> BindResponse:
        """Require the id column and declare the (id, fold) output schema."""
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        id_field = _require_id(input_schema, params.args.id, cls.Meta.name)
        return BindResponse(
            output_schema=pa.schema(
                [id_field, sfield("fold", pa.int64(), "Test fold this row belongs to.", nullable=False)]
            )
        )

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[TArgs]) -> DrainState:
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[TArgs],
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
        table = cls.buffered_table(params, input_schema)
        if table is None or table.num_rows == 0:
            out.emit(pa.RecordBatch.from_pydict({a.id: [], "fold": []}, schema=params.output_schema))
            return

        n = table.num_rows
        ids = table.column(a.id).to_pylist()
        y, groups = cls._y_groups(table, a)
        folds: list[int | None] = [None] * n
        x = np.zeros((n, 1))
        for f, (_, test_idx) in enumerate(cls._make_splitter(a).split(x, y, groups)):
            for i in test_idx:
                folds[i] = f
        out.emit(pa.RecordBatch.from_pydict({a.id: ids, "fold": folds}, schema=params.output_schema))


@dataclass(slots=True, frozen=True)
class KFoldArgs:
    """Arguments for the kfold function."""

    data: Annotated[TableInput, Arg(0, doc="Table to split (only the id column is used).")]
    id: Annotated[str, Arg("id", default="", doc="Column carried onto each row (required).")]
    n_splits: Annotated[int, Arg("n_splits", default=5, doc="Number of folds.")]
    shuffle: Annotated[bool, Arg("shuffle", default=False, doc="Shuffle rows before splitting.")]
    random_state: Annotated[int, Arg("random_state", default=0, doc="Random seed (used when shuffle is true).")]


class KFold(_FoldFunction[KFoldArgs]):
    """Assign each row a K-fold test fold (id, fold)."""

    FunctionArguments: ClassVar[type] = KFoldArgs

    class Meta:
        """VGI metadata for the kfold function."""

        name = "kfold"
        description = "Assign each row a K-fold test fold (id, fold)"
        categories = ["model-selection", "evaluation"]
        tags = {"vgi.columns_md": _FOLD_MD}
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM sklearn.kfold((SELECT sample_id FROM sklearn.iris()), "
                    "id := 'sample_id', n_splits := 5)"
                ),
                description="5-fold assignment for iris rows",
            )
        ]

    @classmethod
    def _make_splitter(cls, args: KFoldArgs) -> Any:
        """Build the scikit-learn KFold splitter."""
        from sklearn.model_selection import KFold as SkKFold

        return SkKFold(
            n_splits=args.n_splits, shuffle=args.shuffle, random_state=args.random_state if args.shuffle else None
        )


@dataclass(slots=True, frozen=True)
class StratifiedKFoldArgs:
    """Arguments for the stratified_kfold function."""

    data: Annotated[TableInput, Arg(0, doc="Table to split (id + the stratify label column).")]
    id: Annotated[str, Arg("id", default="", doc="Column carried onto each row (required).")]
    label: Annotated[str, Arg("label", default="", doc="Label column to stratify on (required).")]
    n_splits: Annotated[int, Arg("n_splits", default=5, doc="Number of folds.")]
    shuffle: Annotated[bool, Arg("shuffle", default=False, doc="Shuffle rows before splitting.")]
    random_state: Annotated[int, Arg("random_state", default=0, doc="Random seed (used when shuffle is true).")]


class StratifiedKFold(_FoldFunction[StratifiedKFoldArgs]):
    """K-fold that preserves each class's proportion per fold (id, fold)."""

    FunctionArguments: ClassVar[type] = StratifiedKFoldArgs

    class Meta:
        """VGI metadata for the stratified_kfold function."""

        name = "stratified_kfold"
        description = "K-fold that preserves each class's proportion per fold (id, fold)"
        categories = ["model-selection", "evaluation"]
        tags = {"vgi.columns_md": _FOLD_MD}
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM sklearn.stratified_kfold((SELECT sample_id, target FROM sklearn.iris()), "
                    "id := 'sample_id', label := 'target', n_splits := 5)"
                ),
                description="Class-balanced 5-fold for iris",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[StratifiedKFoldArgs]) -> BindResponse:
        """Require the stratify label column before the base (id, fold) bind."""
        a = params.args
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        if not a.label or a.label not in input_schema.names:
            raise ValueError(f"stratified_kfold requires a 'label' column present in the input (got {a.label!r})")
        return super().on_bind(params)

    @classmethod
    def _y_groups(cls, table: pa.Table, args: StratifiedKFoldArgs) -> tuple[np.ndarray | None, list[Any] | None]:
        y = np.rint(np.asarray(table.column(args.label).to_numpy(zero_copy_only=False), dtype=float)).astype(int)
        return y, None

    @classmethod
    def _make_splitter(cls, args: StratifiedKFoldArgs) -> Any:
        from sklearn.model_selection import StratifiedKFold as SkStratifiedKFold

        return SkStratifiedKFold(
            n_splits=args.n_splits, shuffle=args.shuffle, random_state=args.random_state if args.shuffle else None
        )


@dataclass(slots=True, frozen=True)
class GroupKFoldArgs:
    """Arguments for the group_kfold function."""

    data: Annotated[TableInput, Arg(0, doc="Table to split (id + the group column).")]
    id: Annotated[str, Arg("id", default="", doc="Column carried onto each row (required).")]
    # Named 'group_col' rather than 'group' because `group :=` collides with the SQL GROUP keyword.
    group_col: Annotated[
        str, Arg("group_col", default="", doc="Group column; rows of a group stay in the same fold (required).")
    ]
    n_splits: Annotated[int, Arg("n_splits", default=5, doc="Number of folds.")]


class GroupKFold(_FoldFunction[GroupKFoldArgs]):
    """K-fold that keeps all rows of a group in the same fold (id, fold)."""

    FunctionArguments: ClassVar[type] = GroupKFoldArgs

    class Meta:
        """VGI metadata for the group_kfold function."""

        name = "group_kfold"
        description = "K-fold that keeps all rows of a group in the same fold (id, fold)"
        categories = ["model-selection", "evaluation"]
        tags = {"vgi.columns_md": _FOLD_MD}
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM sklearn.group_kfold((SELECT sample_id, target AS grp FROM sklearn.iris()), "
                    "id := 'sample_id', group_col := 'grp', n_splits := 3)"
                ),
                description="Group-aware 3-fold (no group spans folds)",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[GroupKFoldArgs]) -> BindResponse:
        """Require the group column before the base (id, fold) bind."""
        a = params.args
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        if not a.group_col or a.group_col not in input_schema.names:
            raise ValueError(f"group_kfold requires a 'group_col' column present in the input (got {a.group_col!r})")
        return super().on_bind(params)

    @classmethod
    def _y_groups(cls, table: pa.Table, args: GroupKFoldArgs) -> tuple[np.ndarray | None, list[Any] | None]:
        return None, table.column(args.group_col).to_pylist()

    @classmethod
    def _make_splitter(cls, args: GroupKFoldArgs) -> Any:
        from sklearn.model_selection import GroupKFold as SkGroupKFold

        return SkGroupKFold(n_splits=args.n_splits)


@dataclass(slots=True, frozen=True)
class TimeSeriesSplitArgs:
    """Arguments for the timeseries_split function."""

    data: Annotated[TableInput, Arg(0, doc="Table to split, in time order (only the id column is used).")]
    id: Annotated[str, Arg("id", default="", doc="Column carried onto each row (required).")]
    n_splits: Annotated[int, Arg("n_splits", default=5, doc="Number of expanding-window splits.")]


class TimeSeriesSplit(SinkBuffer[TimeSeriesSplitArgs, DrainState]):
    """Expanding-window splits for ordered data: (split, id, role) long format."""

    FunctionArguments: ClassVar[type] = TimeSeriesSplitArgs

    class Meta:
        """VGI metadata for the timeseries_split function."""

        name = "timeseries_split"
        description = "Expanding-window splits for ordered data: (split, id, role) in {train, test}"
        categories = ["model-selection", "evaluation"]
        tags = {
            "vgi.columns_md": columns_md_rows(
                [
                    ("split", "BIGINT", "Split index (0-based)."),
                    ("<id>", "(input id type)", "The id column, carried through from the input."),
                    ("role", "VARCHAR", "'train' or 'test' for this row in this split."),
                ]
            )
        }
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM sklearn.timeseries_split((SELECT sample_id FROM sklearn.iris() ORDER BY sample_id), "
                    "id := 'sample_id', n_splits := 5)"
                ),
                description="Forward-chaining train/test splits over ordered rows",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[TimeSeriesSplitArgs]) -> BindResponse:
        """Require the id column and declare the (split, id, role) output schema."""
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        id_field = _require_id(input_schema, params.args.id, cls.Meta.name)
        return BindResponse(
            output_schema=pa.schema(
                [
                    sfield("split", pa.int64(), "Split index (0-based).", nullable=False),
                    id_field,
                    sfield("role", pa.string(), "'train' or 'test' for this row in this split.", nullable=False),
                ]
            )
        )

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[TimeSeriesSplitArgs]
    ) -> DrainState:
        """Start with an unfinished drain state."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[TimeSeriesSplitArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Emit each split's train/test row assignments over the buffered table."""
        if state.done:
            out.finish()
            return
        state.done = True

        from sklearn.model_selection import TimeSeriesSplit as SkTimeSeriesSplit

        a = params.args
        input_schema = input_schema_of(params)
        table = cls.buffered_table(params, input_schema)
        if table is None or table.num_rows == 0:
            out.emit(pa.RecordBatch.from_pydict({"split": [], a.id: [], "role": []}, schema=params.output_schema))
            return

        ids = table.column(a.id).to_pylist()
        x = np.zeros((table.num_rows, 1))
        splits: list[int] = []
        id_col: list[Any] = []
        roles: list[str] = []
        for s, (train_idx, test_idx) in enumerate(SkTimeSeriesSplit(n_splits=a.n_splits).split(x)):
            for i in train_idx:
                splits.append(s)
                id_col.append(ids[i])
                roles.append("train")
            for i in test_idx:
                splits.append(s)
                id_col.append(ids[i])
                roles.append("test")
        out.emit(
            pa.RecordBatch.from_pydict({"split": splits, a.id: id_col, "role": roles}, schema=params.output_schema)
        )


SPLITTER_FUNCTIONS: list[type] = [KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit]
