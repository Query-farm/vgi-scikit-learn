"""Text vectorizers: turn a text column into numeric features.

``count_vectorizer`` and ``tfidf_vectorizer`` tokenize a string column and emit
the document-term matrix in **long format** -- one row per non-zero cell,
``(id, term, value)``. Long format sidesteps the data-dependent-width limit (the
vocabulary isn't known until fit time) and is the natural shape for SQL: pivot it
back to a wide matrix, join term weights, or aggregate per document.

    -- top terms per document
    SELECT id, term, value
    FROM sklearn.tfidf_vectorizer((SELECT id, body FROM docs), id := 'id', text := 'body')
    QUALIFY row_number() OVER (PARTITION BY id ORDER BY value DESC) <= 5;

Both buffer the whole corpus (the vocabulary -- and idf, for tf-idf -- need every
document), then fit_transform once. The text column is taken from ``text :=`` or,
if omitted, the single non-id column.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

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

_VECTORIZER_ID_NOTE = "If an `id` column is named, it is carried through as the first column on each row."


@dataclass(slots=True, frozen=True)
class _VectorizerArgs:
    data: Annotated[TableInput, Arg(0, doc="Table with an id column and a text column.")]
    id: Annotated[str, Arg("id", default="", doc="Optional id column to carry onto each emitted row.")]
    text: Annotated[
        str, Arg("text", default="", doc="Text column to vectorize (defaults to the single non-id column).")
    ]
    lowercase: Annotated[bool, Arg("lowercase", default=True, doc="Lowercase before tokenizing.")]
    stop_words: Annotated[str, Arg("stop_words", default="", doc="'english' to drop English stop words; '' for none.")]
    ngram_max: Annotated[int, Arg("ngram_max", default=1, doc="Max n-gram size (uses ngram_range=(1, ngram_max)).")]
    min_df: Annotated[int, Arg("min_df", default=1, doc="Ignore terms in fewer than this many documents.")]
    max_df: Annotated[float, Arg("max_df", default=1.0, doc="Ignore terms in more than this fraction of documents.")]
    max_features: Annotated[int, Arg("max_features", default=0, doc="Keep only the top-N terms by frequency; 0 = all.")]


def _text_column(input_schema: pa.Schema, id_col: str, text_arg: str) -> str:
    """Resolve which column holds the text to vectorize."""
    if text_arg:
        if text_arg not in input_schema.names:
            raise ValueError(f"text column {text_arg!r} not found in input; columns: {', '.join(input_schema.names)}")
        return text_arg
    candidates = [n for n in input_schema.names if n != id_col]
    if len(candidates) != 1:
        raise ValueError(
            "could not infer the text column; pass text := 'column' "
            f"(non-id columns: {', '.join(candidates) or '<none>'})"
        )
    return str(candidates[0])


class _Vectorizer(SinkBuffer[_VectorizerArgs, DrainState]):
    """Buffer the corpus, fit_transform once, emit the document-term matrix long."""

    FunctionArguments: ClassVar[type] = _VectorizerArgs
    # Subclasses set these:
    _value_type: ClassVar[pa.DataType] = pa.float64()
    _value_doc: ClassVar[str] = "Term weight."

    @classmethod
    def _make_vectorizer(cls, args: _VectorizerArgs) -> Any:
        raise NotImplementedError

    @classmethod
    def _common_kwargs(cls, args: _VectorizerArgs) -> dict[str, Any]:
        return {
            "lowercase": args.lowercase,
            "stop_words": args.stop_words or None,
            "ngram_range": (1, max(1, args.ngram_max)),
            "min_df": args.min_df,
            "max_df": args.max_df,
            "max_features": args.max_features or None,
        }

    @classmethod
    def _output_schema(cls, input_schema: pa.Schema, args: _VectorizerArgs) -> pa.Schema:
        fields: list[pa.Field] = []
        if args.id:
            fields.append(input_schema.field(args.id))
        fields.append(sfield("term", pa.string(), "Vocabulary term (or n-gram).", nullable=False))
        fields.append(sfield("value", cls._value_type, cls._value_doc, nullable=False))
        return pa.schema(fields)

    @classmethod
    def on_bind(cls, params: BindParams[_VectorizerArgs]) -> BindResponse:
        a = params.args
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        text_col = _text_column(input_schema, a.id, a.text)
        if not pa.types.is_string(input_schema.field(text_col).type) and not pa.types.is_large_string(
            input_schema.field(text_col).type
        ):
            raise ValueError(f"text column {text_col!r} must be a string/VARCHAR column")
        return BindResponse(output_schema=cls._output_schema(input_schema, a))

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[_VectorizerArgs]
    ) -> DrainState:
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[_VectorizerArgs],
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
        out_schema = params.output_schema

        table = cls.buffered_table(params, input_schema)
        if table is None or table.num_rows == 0:
            out.emit(pa.RecordBatch.from_pydict({n: [] for n in out_schema.names}, schema=out_schema))
            return
        out.emit(pa.RecordBatch.from_pydict(cls.encode(table, a), schema=out_schema))

    @classmethod
    def encode(cls, table: pa.Table, args: _VectorizerArgs) -> dict[str, list[Any]]:
        """Vectorize the text column and return the long-format columns."""
        text_col = _text_column(table.schema, args.id, args.text)
        docs = [t if t is not None else "" for t in table.column(text_col).to_pylist()]
        id_vals = table.column(args.id).to_pylist() if args.id else None

        vec = cls._make_vectorizer(args)
        matrix = vec.fit_transform(docs).tocoo()
        terms = vec.get_feature_names_out()
        cast = int if pa.types.is_integer(cls._value_type) else float

        ids: list[Any] = []
        term_col: list[str] = []
        value_col: list[Any] = []
        for r, c, v in zip(matrix.row.tolist(), matrix.col.tolist(), matrix.data.tolist(), strict=True):
            if id_vals is not None:
                ids.append(id_vals[r])
            term_col.append(str(terms[c]))
            value_col.append(cast(v))

        columns: dict[str, list[Any]] = {}
        if args.id:
            columns[args.id] = ids
        columns["term"] = term_col
        columns["value"] = value_col
        return columns


class CountVectorizerFn(_Vectorizer):
    """Tokenize a text column into a document-term count matrix (long format)."""

    _value_type: ClassVar[pa.DataType] = pa.int64()
    _value_doc: ClassVar[str] = "Term count in the document."

    class Meta:
        """VGI metadata for the count_vectorizer function."""

        name = "count_vectorizer"
        description = "Tokenize a text column into a document-term count matrix (long format)"
        categories = ["preprocessing", "text", "encoding"]
        examples = [
            FunctionExample(
                sql=("SELECT * FROM sklearn.count_vectorizer((SELECT id, body FROM docs), id := 'id', text := 'body')"),
                description="Term counts per document",
            )
        ]
        tags = {
            "vgi.columns_md": columns_md_rows(
                [
                    ("term", "VARCHAR", "Vocabulary term (or n-gram)."),
                    ("value", "BIGINT", "Term count in the document."),
                ],
                note=_VECTORIZER_ID_NOTE,
            )
        }

    @classmethod
    def _make_vectorizer(cls, args: _VectorizerArgs) -> Any:
        from sklearn.feature_extraction.text import CountVectorizer

        return CountVectorizer(**cls._common_kwargs(args))


class TfidfVectorizerFn(_Vectorizer):
    """Tokenize a text column into a TF-IDF document-term matrix (long format)."""

    _value_type: ClassVar[pa.DataType] = pa.float64()
    _value_doc: ClassVar[str] = "TF-IDF weight of the term in the document."

    class Meta:
        """VGI metadata for the tfidf_vectorizer function."""

        name = "tfidf_vectorizer"
        description = "Tokenize a text column into a TF-IDF document-term matrix (long format)"
        categories = ["preprocessing", "text", "encoding"]
        examples = [
            FunctionExample(
                sql=("SELECT * FROM sklearn.tfidf_vectorizer((SELECT id, body FROM docs), id := 'id', text := 'body')"),
                description="TF-IDF weights per document",
            )
        ]
        tags = {
            "vgi.columns_md": columns_md_rows(
                [
                    ("term", "VARCHAR", "Vocabulary term (or n-gram)."),
                    ("value", "DOUBLE", "TF-IDF weight of the term in the document."),
                ],
                note=_VECTORIZER_ID_NOTE,
            )
        }

    @classmethod
    def _make_vectorizer(cls, args: _VectorizerArgs) -> Any:
        from sklearn.feature_extraction.text import TfidfVectorizer

        return TfidfVectorizer(**cls._common_kwargs(args))


TEXT_FUNCTIONS: list[type] = [CountVectorizerFn, TfidfVectorizerFn]
