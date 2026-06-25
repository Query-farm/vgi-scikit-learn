"""Persisted, reusable transformers: ``fit_transformer`` + ``apply_transform``.

The standalone transforms (transforms.py) all ``fit_transform`` in one shot, so
you can't fit a scaler/PCA/imputer on training data and re-apply it to new data.
These two close that gap, mirroring ``fit`` / ``predict``:

* ``fit_transformer`` fits a transformer of a given ``kind`` on the input, returns
  it as a self-contained BLOB, and persists it to the registry when
  ``transformer_name`` is given.
* ``apply_transform`` streams a table through a stored transformer (by
  ``transformer_name :=`` or a ``transformer :=`` BLOB), aligning features by name
  and emitting the transformed columns.

    SELECT * FROM sklearn.preprocessing.fit_transformer((SELECT * FROM train), transformer_name := 'sc',
                                          kind := 'standard_scaler');
    SELECT * FROM sklearn.preprocessing.apply_transform((SELECT * FROM test), transformer_name := 'sc', id := 'id');

Output shape is fixed at fit time: transforms whose output mirrors the input keep
the feature column names; reducers (pca, truncated_svd) emit ``component_1..k``.
Input features are numeric (see ``buffering.matrix``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, Any, ClassVar

import numpy as np
import pyarrow as pa
import sklearn
from vgi.arguments import Arg, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_buffering_function import TableBufferingParams
from vgi.table_function import (
    BindParams,
    ProcessParams,
    TableCardinality,
    TableFunctionGenerator,
    bind_fixed_schema,
    init_single_worker,
)
from vgi.table_in_out_function import TableInOutGenerator
from vgi_rpc.rpc import OutputCollector
from vgi_rpc.rpc import OutputCollector as InOutCollector

from .buffering import DrainState, SinkBuffer, input_schema_of, matrix
from .registry import (
    ModelNotFoundError,
    TransformerMetadata,
    get_transformer_store,
    now_iso,
    pack_transformer,
    unpack_transformer,
    unpack_transformer_meta,
    validate_name,
)
from .schema_utils import columns_md, columns_md_rows
from .schema_utils import field as sfield


def _build(kind: str, params: dict[str, Any]) -> Any:
    """Construct an (unfitted) transformer of ``kind`` with ``params``."""
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import (
        Binarizer,
        KBinsDiscretizer,
        MaxAbsScaler,
        MinMaxScaler,
        Normalizer,
        PowerTransformer,
        QuantileTransformer,
        RobustScaler,
        StandardScaler,
    )

    factories: dict[str, Any] = {
        "standard_scaler": lambda p: StandardScaler(**p),
        "minmax_scaler": lambda p: MinMaxScaler(**p),
        "robust_scaler": lambda p: RobustScaler(**p),
        "maxabs_scaler": lambda p: MaxAbsScaler(**p),
        "normalizer": lambda p: Normalizer(**p),
        "power_transformer": lambda p: PowerTransformer(**p),
        "quantile_transformer": lambda p: QuantileTransformer(**p),
        "simple_imputer": lambda p: SimpleImputer(**p),
        "binarizer": lambda p: Binarizer(**p),
        "kbins_discretizer": lambda p: KBinsDiscretizer(encode="ordinal", **p),
        "pca": lambda p: PCA(**p),
        "truncated_svd": lambda p: TruncatedSVD(**p),
    }
    if kind not in factories:
        raise ValueError(f"unknown transformer kind {kind!r}; choose one of: {', '.join(sorted(factories))}")
    try:
        return factories[kind](params)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid params for {kind!r}: {exc}") from exc


TRANSFORMER_KINDS = (
    "standard_scaler",
    "minmax_scaler",
    "robust_scaler",
    "maxabs_scaler",
    "normalizer",
    "power_transformer",
    "quantile_transformer",
    "simple_imputer",
    "binarizer",
    "kbins_discretizer",
    "pca",
    "truncated_svd",
)
_INT_OUTPUT = {"kbins_discretizer"}


def _features_excluding(input_schema: pa.Schema, *exclude: str) -> list[str]:
    drop = {e for e in exclude if e}
    return [n for n in input_schema.names if n not in drop]


def _parse_params(params: str) -> dict[str, Any]:
    params = (params or "").strip()
    if not params:
        return {}
    parsed = json.loads(params)
    if not isinstance(parsed, dict):
        raise ValueError("params must be a JSON object, e.g. '{\"n_components\": 2}'")
    return parsed


# ===========================================================================
# fit_transformer
# ===========================================================================


@dataclass(slots=True, frozen=True)
class FitTransformerArgs:
    """Arguments for the fit_transformer function."""

    data: Annotated[TableInput, Arg(0, doc="Training rows: feature columns [+ id].")]
    transformer_name: Annotated[
        str, Arg("transformer_name", default="", doc="Name to store the fitted transformer under (optional).")
    ]
    kind: Annotated[str, Arg("kind", default="standard_scaler", doc="Transformer kind, e.g. 'standard_scaler', 'pca'.")]
    id: Annotated[str, Arg("id", default="", doc="Optional id column to exclude from features.")]
    params: Annotated[str, Arg("params", default="", doc="JSON object of transformer parameters.")]


_FIT_TRANSFORMER_SCHEMA = pa.schema(
    [
        sfield("transformer_name", pa.string(), "Name it was stored under ('' if not persisted).", nullable=False),
        sfield("kind", pa.string(), "Transformer kind.", nullable=False),
        sfield("n_features", pa.int64(), "Number of input features.", nullable=False),
        sfield("n_output", pa.int64(), "Number of output columns.", nullable=False),
        sfield("output_names", pa.list_(pa.string()), "Ordered output column names.", nullable=False),
        sfield("transformer", pa.binary(), "The fitted transformer as a self-contained BLOB.", nullable=False),
    ]
)


def _output_names(feats: list[str], n_output: int) -> list[str]:
    """Mirror the input feature names when the width is preserved, else component_1..k."""
    if n_output == len(feats):
        return list(feats)
    return [f"component_{i + 1}" for i in range(n_output)]


class FitTransformer(SinkBuffer[FitTransformerArgs, DrainState]):
    """Buffer a training table, fit a transformer, return it as a BLOB; persist if named."""

    FunctionArguments: ClassVar[type] = FitTransformerArgs

    class Meta:
        """VGI metadata for the fit_transformer function."""

        name = "fit_transformer"
        description = "Fit a transformer (scaler/PCA/imputer/...) and return it as a BLOB; persist if named"
        categories = ["preprocessing", "registry"]
        tags = {
            "vgi.result_columns_md": columns_md(_FIT_TRANSFORMER_SCHEMA),
            "vgi.doc_llm": (
                "Table function that fits a *reusable* transformer on a training table and returns it as a "
                "self-contained BLOB -- the persistent counterpart of the one-shot transforms, so you can "
                "fit on training data and re-apply to new data with `apply_transform`. It buffers the "
                "numeric feature relation `(SELECT ...)` (Arg(0)), builds a transformer of the given "
                "`kind :=` (`'standard_scaler'`, `'minmax_scaler'`, `'robust_scaler'`, `'maxabs_scaler'`, "
                "`'normalizer'`, `'power_transformer'`, `'quantile_transformer'`, `'simple_imputer'`, "
                "`'binarizer'`, `'kbins_discretizer'`, `'pca'`, `'truncated_svd'`) with optional JSON "
                "`params :=`, fits it, and persists to the registry only if `transformer_name :=` is given. "
                "`id :=` excludes a key column from features. It returns one summary row: the name, kind, "
                "input/output counts, output column names, and the `transformer` BLOB."
            ),
            "vgi.doc_md": (
                "**Fit transformer** — fit a reusable scaler/PCA/imputer and return it as a BLOB.\n\n"
                "The stored-model counterpart to the one-shot transforms: fit once on training data, then "
                "re-apply to new rows via `apply_transform` (saved to the registry only when named).\n\n"
                "- Input: `(SELECT ...)` numeric training table; `id :=` column excluded from features\n"
                "- `kind :=` one of the scalers/`pca`/`truncated_svd`/`simple_imputer`/`kbins_discretizer`/"
                "etc.; `params :=` a JSON object of transformer params; `transformer_name :=` to persist\n"
                "- Output (one row): `transformer_name`, `kind`, `n_features`, `n_output`, `output_names`, "
                "and the `transformer` BLOB\n"
                "- Output width is fixed at fit time (feature names preserved, or `component_1..k` for "
                "reducers), so `apply_transform` has a stable schema"
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    "SELECT transformer_name, kind, n_output FROM sklearn.preprocessing.fit_transformer("
                    "(SELECT sepal_length_cm, sepal_width_cm FROM sklearn.datasets.iris()), "
                    "transformer_name := 'sc', kind := 'standard_scaler')"
                ),
                description="Fit a StandardScaler and store it as 'sc'",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[FitTransformerArgs]) -> BindResponse:
        """Validate the kind and name and declare the fit summary output schema."""
        a = params.args
        if a.kind not in TRANSFORMER_KINDS:
            raise ValueError(f"unknown transformer kind {a.kind!r}; choose one of: {', '.join(TRANSFORMER_KINDS)}")
        if a.transformer_name:
            validate_name(a.transformer_name)
        assert params.bind_call.input_schema is not None
        return BindResponse(output_schema=_FIT_TRANSFORMER_SCHEMA)

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[FitTransformerArgs]
    ) -> DrainState:
        """Start with an unfinished drain state."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[FitTransformerArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Fit the transformer on the buffered table, persist if named, emit a summary."""
        if state.done:
            out.finish()
            return
        state.done = True

        a = params.args
        input_schema = input_schema_of(params)
        feats = _features_excluding(input_schema, a.id)

        table = cls.buffered_table(params, input_schema)
        if table is None or table.num_rows == 0:
            raise ValueError("fit_transformer received no training rows")

        x = matrix(table, feats)
        transformer = _build(a.kind, _parse_params(a.params))
        xt = np.asarray(transformer.fit_transform(x))
        n_output = int(xt.shape[1]) if xt.ndim == 2 else 1
        output_names = _output_names(feats, n_output)

        meta = TransformerMetadata(
            name=a.transformer_name,
            kind=a.kind,
            feature_names=feats,
            output_names=output_names,
            output_int=a.kind in _INT_OUTPUT,
            params=_parse_params(a.params),
            n_features=len(feats),
            n_output=n_output,
            sklearn_version=sklearn.__version__,
            created_at=now_iso(),
        )
        if a.transformer_name:
            get_transformer_store().save(transformer, meta)

        out.emit(
            pa.RecordBatch.from_pydict(
                {
                    "transformer_name": [a.transformer_name],
                    "kind": [a.kind],
                    "n_features": [meta.n_features],
                    "n_output": [meta.n_output],
                    "output_names": [output_names],
                    "transformer": [pack_transformer(transformer, meta)],
                },
                schema=params.output_schema,
            )
        )


# ===========================================================================
# apply_transform
# ===========================================================================


@dataclass(slots=True, frozen=True)
class ApplyTransformArgs:
    """Arguments for the apply_transform function."""

    data: Annotated[TableInput, Arg(0, doc="Rows to transform (must contain the transformer's feature columns).")]
    transformer_name: Annotated[
        str, Arg("transformer_name", default="", doc="Name of a stored transformer. Provide this OR transformer.")
    ]
    transformer: Annotated[
        bytes, Arg("transformer", default=b"", doc="A fitted transformer (from fit_transformer). Provide this OR name.")
    ]
    id: Annotated[str, Arg("id", default="", doc="Optional id column to carry through.")]


@lru_cache(maxsize=64)
def _load_blob(blob: bytes) -> tuple[Any, TransformerMetadata]:
    return unpack_transformer(blob)


class ApplyTransform(TableInOutGenerator[ApplyTransformArgs]):
    """Stream a table through a stored, already-fitted transformer."""

    FunctionArguments: ClassVar[type] = ApplyTransformArgs

    class Meta:
        """VGI metadata for the apply_transform function."""

        name = "apply_transform"
        description = "Stream a table through a stored, already-fitted transformer"
        categories = ["preprocessing", "registry", "inference"]
        tags = {
            "vgi.result_columns_md": columns_md_rows(
                [],
                note=(
                    "One column per transformer output (DOUBLE, or BIGINT for kbins_discretizer): the "
                    "fit-time output names -- the input feature names when width is preserved, else "
                    "`component_1..k`. If an `id` column is named, it is carried through as the first column."
                ),
            ),
            "vgi.doc_llm": (
                "Table function that streams a table through an already-fitted transformer (the inference "
                "counterpart of `fit_transformer`, mirroring how `predict` consumes a stored model). Pass "
                "the data as `(SELECT ...)` (Arg(0)) and identify the transformer with **either** "
                "`transformer_name :=` (a stored name) **or** `transformer :=` (a BLOB from "
                "`fit_transformer`). Features are aligned by name -- the input must contain the "
                "transformer's feature columns (extra columns are ignored, missing ones raise at bind) -- "
                "and `id :=` carries a key through as the first column. It emits the fit-time output columns "
                "(`DOUBLE`, or `BIGINT` for `kbins_discretizer`): the original feature names when width is "
                "preserved, else `component_1..k`. Use it to apply a training-fitted scaler/PCA/imputer to "
                "fresh data without refitting."
            ),
            "vgi.doc_md": (
                "**Apply transform** — run new rows through a stored, already-fitted transformer.\n\n"
                "The inference step for `fit_transformer`: aligns features by name and transforms each batch "
                "without refitting (so train/test stats match).\n\n"
                "- Input: `(SELECT ...)` table containing the transformer's feature columns; `id :=` "
                "passthrough column\n"
                "- Identify the transformer by **`transformer_name :=`** (stored) **or** `transformer :=` "
                "(a BLOB) -- one or the other\n"
                "- Output: the fit-time output columns (`DOUBLE`, or `BIGINT` for `kbins_discretizer`) -- "
                "feature names when width is preserved, else `component_1..k`\n"
                "- Missing required feature columns error at bind; extra input columns are ignored"
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM sklearn.preprocessing.apply_transform((SELECT * FROM sklearn.datasets.iris()), "
                    "transformer_name := 'sc', id := 'sample_id')"
                ),
                description="Apply the stored 'sc' transformer to new data",
            )
        ]

    @classmethod
    def _meta(cls, a: ApplyTransformArgs) -> TransformerMetadata:
        if a.transformer_name:
            try:
                return get_transformer_store().load_meta(a.transformer_name)
            except ModelNotFoundError as exc:
                raise ValueError(f"transformer {a.transformer_name!r} not found in the registry") from exc
        return unpack_transformer_meta(a.transformer)

    @classmethod
    def on_bind(cls, params: BindParams[ApplyTransformArgs]) -> BindResponse:
        """Resolve the stored transformer and declare its output column schema."""
        a = params.args
        if not a.transformer_name and not a.transformer:
            raise ValueError("apply_transform requires either 'transformer_name' or 'transformer' (a BLOB)")
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        meta = cls._meta(a)
        missing = [f for f in meta.feature_names if f not in input_schema.names]
        if missing:
            raise ValueError(
                f"transformer requires feature column(s) {', '.join(missing)} not present in the input; "
                f"transformer features: {', '.join(meta.feature_names)}"
            )
        value_type = pa.int64() if meta.output_int else pa.float64()
        fields: list[pa.Field] = []
        if a.id:
            fields.append(input_schema.field(a.id))
        fields.extend(
            sfield(n, value_type, f"Transformed value ({meta.kind}).", nullable=False) for n in meta.output_names
        )
        return BindResponse(output_schema=pa.schema(fields))

    @classmethod
    def _load(cls, params: ProcessParams[ApplyTransformArgs]) -> tuple[Any, TransformerMetadata]:
        a = params.args
        if a.transformer_name:
            return get_transformer_store().load(a.transformer_name)
        return _load_blob(a.transformer)

    @classmethod
    def process(
        cls,
        params: ProcessParams[ApplyTransformArgs],
        state: None,
        batch: pa.RecordBatch,
        out: InOutCollector,
    ) -> None:
        """Transform each batch with the loaded transformer and emit the result."""
        a = params.args
        transformer, meta = cls._load(params)

        table = pa.Table.from_batches([batch])
        x = matrix(table, meta.feature_names)
        xt = np.asarray(transformer.transform(x))
        if xt.ndim == 1:
            xt = xt.reshape(-1, 1)

        columns: dict[str, list[Any]] = {}
        if a.id:
            columns[a.id] = batch.column(a.id).to_pylist()
        cast = int if meta.output_int else float
        for j, name in enumerate(meta.output_names):
            columns[name] = [cast(v) for v in xt[:, j]]
        out.emit(pa.RecordBatch.from_pydict(columns, schema=params.output_schema))


# ===========================================================================
# Registry management: list_transformers / drop_transformer
# ===========================================================================

_TRANSFORMER_INFO_SCHEMA = pa.schema(
    [
        sfield("transformer_name", pa.string(), "Stored transformer name.", nullable=False),
        sfield("kind", pa.string(), "Transformer kind.", nullable=False),
        sfield("n_features", pa.int32(), "Number of input features.", nullable=False),
        sfield("n_output", pa.int32(), "Number of output columns.", nullable=False),
        sfield("features", pa.list_(pa.string()), "Ordered input feature names.", nullable=False),
        sfield("sklearn_version", pa.string(), "scikit-learn version used to fit."),
        sfield("created_at", pa.string(), "UTC timestamp the transformer was stored."),
    ]
)


def _transformer_rows(metas: list[TransformerMetadata]) -> dict[str, list[Any]]:
    return {
        "transformer_name": [m.name for m in metas],
        "kind": [m.kind for m in metas],
        "n_features": [m.n_features for m in metas],
        "n_output": [m.n_output for m in metas],
        "features": [m.feature_names for m in metas],
        "sklearn_version": [m.sklearn_version for m in metas],
        "created_at": [m.created_at for m in metas],
    }


@dataclass(slots=True, frozen=True)
class _NoArgs:
    pass


@init_single_worker
@bind_fixed_schema
class ListTransformers(TableFunctionGenerator[_NoArgs]):
    """List all stored transformers."""

    FIXED_SCHEMA: ClassVar[pa.Schema] = _TRANSFORMER_INFO_SCHEMA

    class Meta:
        """VGI metadata for the list_transformers function."""

        name = "list_transformers"
        description = "List all stored transformers"
        categories = ["preprocessing", "registry"]
        tags = {
            "vgi.result_columns_md": columns_md(_TRANSFORMER_INFO_SCHEMA),
            "vgi.doc_llm": (
                "Table function that lists every transformer persisted in the registry by `fit_transformer` "
                "(a separate namespace from `list_models`, so the two never mix). Takes no arguments and "
                "emits one row per stored transformer with its `transformer_name`, `kind`, input/output "
                "feature counts, the ordered input `features` list, the `sklearn_version` it was fitted "
                "with, and the `created_at` timestamp. Use it to discover available transformers and read "
                "the metadata you need (name and feature columns) before calling `apply_transform`."
            ),
            "vgi.doc_md": (
                "**List transformers** — inventory of transformers stored by `fit_transformer`.\n\n"
                "Takes no arguments and returns one row per persisted transformer (its own registry "
                "namespace, distinct from models).\n\n"
                "- Columns: `transformer_name`, `kind`, `n_features`, `n_output`, `features` (ordered input "
                "names), `sklearn_version`, `created_at`\n"
                "- The discovery step before `apply_transform`: find the name and the feature columns a "
                "transformer expects"
            ),
        }
        examples = [
            FunctionExample(
                sql="SELECT * FROM sklearn.preprocessing.list_transformers()", description="List stored transformers"
            )
        ]

    @classmethod
    def cardinality(cls, params: BindParams[_NoArgs]) -> TableCardinality:
        """Estimate the number of stored transformers."""
        return TableCardinality(estimate=10, max=10000)

    @classmethod
    def process(cls, params: ProcessParams[_NoArgs], state: None, out: OutputCollector) -> None:
        """Emit one row per stored transformer."""
        out.emit(
            pa.RecordBatch.from_pydict(_transformer_rows(get_transformer_store().list()), schema=params.output_schema)
        )
        out.finish()


@dataclass(slots=True, frozen=True)
class DropTransformerArgs:
    """Arguments for the drop_transformer function."""

    transformer_name: Annotated[str, Arg(0, doc="Name of the transformer to delete.")]


_DROP_TRANSFORMER_SCHEMA = pa.schema(
    [
        sfield("transformer_name", pa.string(), "Name of the transformer.", nullable=False),
        sfield("dropped", pa.bool_(), "True if a transformer was deleted, False if it did not exist.", nullable=False),
    ]
)


@init_single_worker
@bind_fixed_schema
class DropTransformer(TableFunctionGenerator[DropTransformerArgs]):
    """Delete a transformer from the registry."""

    FIXED_SCHEMA: ClassVar[pa.Schema] = _DROP_TRANSFORMER_SCHEMA

    class Meta:
        """VGI metadata for the drop_transformer function."""

        name = "drop_transformer"
        description = "Delete a transformer from the registry"
        categories = ["preprocessing", "registry"]
        tags = {
            "vgi.result_columns_md": columns_md(_DROP_TRANSFORMER_SCHEMA),
            "vgi.doc_llm": (
                "Table function that deletes a stored transformer from the registry by name. Pass the "
                "transformer name positionally (Arg(0), e.g. `drop_transformer('sc')`); it removes the "
                "transformer's artifact and metadata and returns a single row with `transformer_name` and a "
                "`dropped` boolean -- `true` if it existed and was deleted, `false` if no such transformer "
                "was found (idempotent, never errors on a missing name). Use it to clean up transformers "
                "fitted by `fit_transformer`; it only affects the transformer registry, not the model "
                "registry."
            ),
            "vgi.doc_md": (
                "**Drop transformer** — delete a stored transformer by name.\n\n"
                "Removes the named transformer's artifact and metadata from the registry; idempotent, so a "
                "missing name is reported rather than erroring.\n\n"
                "- Input: the transformer name, passed positionally (`drop_transformer('sc')`)\n"
                "- Output (one row): `transformer_name` and `dropped` (`true` if deleted, `false` if it did "
                "not exist)\n"
                "- Touches only the transformer registry (not models); the cleanup counterpart of "
                "`fit_transformer`"
            ),
        }
        examples = [
            FunctionExample(
                sql="SELECT * FROM sklearn.preprocessing.drop_transformer('sc')",
                description="Delete a stored transformer",
            )
        ]

    @classmethod
    def cardinality(cls, params: BindParams[DropTransformerArgs]) -> TableCardinality:
        """The drop result is always a single row."""
        return TableCardinality(estimate=1, max=1)

    @classmethod
    def process(cls, params: ProcessParams[DropTransformerArgs], state: None, out: OutputCollector) -> None:
        """Delete the named transformer and emit whether it existed."""
        name = params.args.transformer_name
        dropped = get_transformer_store().delete(name)
        out.emit(
            pa.RecordBatch.from_pydict({"transformer_name": [name], "dropped": [dropped]}, schema=params.output_schema)
        )
        out.finish()


STORED_TRANSFORM_FUNCTIONS: list[type] = [FitTransformer, ApplyTransform, ListTransformers, DropTransformer]
