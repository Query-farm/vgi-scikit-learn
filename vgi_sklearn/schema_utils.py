"""Shared Arrow-schema helpers for the scikit-learn worker.

Keeps column-comment plumbing and name sanitisation in one place so every
dataset/transform/model function exposes consistent, documented schemas to
DuckDB.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import pyarrow as pa


def field(
    name: str,
    type: pa.DataType,
    comment: str,
    *,
    nullable: bool = True,
) -> pa.Field:
    """Build a ``pa.Field`` carrying a column comment in its metadata.

    The ``comment`` metadata key is the framework's transport for column
    comments -- DuckDB surfaces it via ``duckdb_columns()`` and ``DESCRIBE``.
    """
    return pa.field(
        name,
        type,
        nullable=nullable,
        metadata={b"comment": comment.encode("utf-8")},
    )


def _sql_type(t: pa.DataType) -> str:
    """Map a common Arrow type to a readable DuckDB-ish type name."""
    if pa.types.is_dictionary(t):
        return _sql_type(t.value_type)
    if pa.types.is_int64(t):
        return "BIGINT"
    if pa.types.is_int32(t):
        return "INTEGER"
    if pa.types.is_int16(t):
        return "SMALLINT"
    if pa.types.is_int8(t):
        return "TINYINT"
    if pa.types.is_float64(t):
        return "DOUBLE"
    if pa.types.is_float32(t):
        return "FLOAT"
    if pa.types.is_string(t):
        return "VARCHAR"
    if pa.types.is_boolean(t):
        return "BOOLEAN"
    if pa.types.is_binary(t):
        return "BLOB"
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        return f"{_sql_type(t.value_type)}[]"
    return str(t)


def result_columns_schema(schema: pa.Schema) -> str:
    """Render a table function's fixed RETURN columns as ``vgi.result_columns_schema``.

    Emits a JSON array of ``{name, type, description}`` objects, one per field,
    reading each field's ``comment`` metadata. Deriving it from the real output
    schema keeps the declared and the returned columns in lockstep.
    """
    columns = []
    for f in schema:
        desc = ""
        if f.metadata and b"comment" in f.metadata:
            desc = f.metadata[b"comment"].decode("utf-8")
        columns.append({"name": f.name, "type": _sql_type(f.type), "description": desc})
    return json.dumps(columns, separators=(",", ":"))


def dynamic_columns_md(rows: list[tuple[str, str, str]], *, note: str | None = None) -> str:
    """Render ``vgi.result_dynamic_columns_md`` from (name, type, description) rows.

    For functions whose output schema is computed at bind from the arguments or
    the input relation: one ``Name | Type | Description`` table, where
    placeholder names such as ``<feature>`` stand for argument-driven columns.
    """
    lines = ["| Name | Type | Description |", "| --- | --- | --- |"]
    lines += [f"| `{n}` | {t} | {d} |" for n, t, d in rows]
    md = "\n".join(lines)
    if note:
        md += f"\n\n{note}"
    return md


def dynamic_variants_md(variants: list[tuple[str, list[tuple[str, str, str]]]], *, note: str | None = None) -> str:
    """Render ``vgi.result_dynamic_columns_md`` with one captioned table per variant.

    Each ``(caption, rows)`` pair becomes a ``### caption`` heading followed by
    its ``Name | Type | Description`` table, for functions whose result columns
    differ by argument (e.g. classifier vs. regressor output).
    """
    md = "\n\n".join(f"### {caption}\n\n{dynamic_columns_md(rows)}" for caption, rows in variants)
    if note:
        md += f"\n\n{note}"
    return md


_NON_IDENT = re.compile(r"[^0-9a-z]+")


def snake_case(name: str) -> str:
    """Normalise a scikit-learn feature label to a SQL-friendly column name.

    ``"sepal length (cm)"`` -> ``"sepal_length_cm"``. Collapses any run of
    non-alphanumeric characters to a single underscore and lowercases.
    """
    cleaned = _NON_IDENT.sub("_", name.strip().lower()).strip("_")
    if not cleaned:
        return "feature"
    if cleaned[0].isdigit():
        cleaned = f"f_{cleaned}"
    return cleaned


def dedupe_names(names: list[str]) -> list[str]:
    """Ensure column names are unique by suffixing collisions (``_2``, ``_3`` ...)."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen[name] = 1
            out.append(name)
        else:
            seen[name] += 1
            out.append(f"{name}_{seen[name]}")
    return out


@dataclass(slots=True, frozen=True, kw_only=True)
class NoArgs:
    """Empty argument set for functions that take no user-facing parameters."""
