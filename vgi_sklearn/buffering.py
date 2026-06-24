"""Shared plumbing for table-buffering functions.

Buffering functions (fit_transform, model fit, cross_val_predict) all need the
whole input before producing output. The sink phase serializes each input batch
to execution-scoped storage; finalize reassembles the full table. This module
holds the serialization, storage, and matrix-assembly helpers plus the
single-bucket sink/combine implementation so each function only writes its
finalize logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
from vgi.table_buffering_function import TableBufferingFunction, TableBufferingParams
from vgi_rpc import ArrowSerializableDataclass

_DATA_KEY = b"input_batches"


@dataclass(kw_only=True)
class DrainState(ArrowSerializableDataclass):
    """Per-finalize-stream cursor: emit the result once, then finish."""

    done: bool = False


def serialize_batch(batch: pa.RecordBatch) -> bytes:
    """Serialize one record batch to the Arrow IPC stream format."""
    sink = pa.BufferOutputStream()
    # pa.ipc.new_stream is untyped in pyarrow's partial stubs.
    with pa.ipc.new_stream(sink, batch.schema) as writer:  # type: ignore[no-untyped-call]
        writer.write_batch(batch)
    return bytes(sink.getvalue().to_pybytes())


def deserialize_batches(value: bytes) -> list[pa.RecordBatch]:
    """Read back the record batches from an Arrow IPC stream blob."""
    # pa.ipc.open_stream is untyped in pyarrow's partial stubs.
    reader = pa.ipc.open_stream(pa.BufferReader(value))  # type: ignore[no-untyped-call]
    batches: list[pa.RecordBatch] = reader.read_all().to_batches()
    return batches


def matrix(table: pa.Table, feature_names: list[str], *, what: str = "feature") -> npt.NDArray[np.float64]:
    """Assemble the named columns (in the given order) into a 2D float64 array.

    Selects ``feature_names`` by name, so input column order does not matter and
    extra columns are ignored. Raises a clear error -- rather than an opaque
    pyarrow KeyError or numpy ValueError -- when a column is missing or not
    numeric. ``what`` labels the columns in error messages (e.g. "feature").
    """
    present = set(table.schema.names)
    missing = [n for n in feature_names if n not in present]
    if missing:
        raise ValueError(
            f"missing required {what} column(s): {', '.join(missing)}; "
            f"input has columns: {', '.join(table.schema.names)}"
        )
    non_numeric = [
        n
        for n in feature_names
        if not pa.types.is_floating(table.schema.field(n).type)
        and not pa.types.is_integer(table.schema.field(n).type)
        and not pa.types.is_boolean(table.schema.field(n).type)
    ]
    if non_numeric:
        raise ValueError(
            f"{what} column(s) must be numeric, but these are not: "
            + ", ".join(f"{n} ({table.schema.field(n).type})" for n in non_numeric)
            + ". Select only numeric columns, or encode/scale them first."
        )
    cols = [np.asarray(table.column(name).to_numpy(zero_copy_only=False), dtype=np.float64) for name in feature_names]
    if not cols:
        return np.empty((table.num_rows, 0), dtype=np.float64)
    return np.column_stack(cols)


class SinkBuffer[TArgs, TState](TableBufferingFunction[TArgs, TState]):
    """Single-bucket sink/combine: buffer every input batch under one key.

    Subclasses implement ``on_bind``, ``initial_finalize_state``, and
    ``finalize`` (calling ``buffered_table(params)`` to get the full input).
    """

    @classmethod
    def process(cls, batch: pa.RecordBatch, params: TableBufferingParams[TArgs]) -> bytes:
        """Append the (non-empty) input batch to the single buffer bucket."""
        if batch.num_rows:
            params.storage.state_append(_DATA_KEY, b"", serialize_batch(batch))
        return params.execution_id

    @classmethod
    def combine(cls, state_ids: list[bytes], params: TableBufferingParams[TArgs]) -> list[bytes]:
        """Collapse partial state ids back to the single execution id."""
        return [params.execution_id]

    @classmethod
    def buffered_table(cls, params: TableBufferingParams[TArgs], input_schema: pa.Schema) -> pa.Table | None:
        """Reassemble every buffered batch into the full input table, or None if empty."""
        batches: list[pa.RecordBatch] = []
        for _sid, value in params.storage.state_log_scan(_DATA_KEY, b""):
            batches.extend(deserialize_batches(value))
        if not batches:
            return None
        return pa.Table.from_batches(batches, schema=input_schema)


def input_schema_of(params: Any) -> pa.Schema:
    """Input schema from a process/finalize params object."""
    schema = params.init_call.bind_call.input_schema
    assert schema is not None
    return schema
