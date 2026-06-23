"""Unit tests for the text vectorizers' encode logic + output schema.

The full buffering lifecycle is covered by test/sql/sklearn_text.test.
"""

from __future__ import annotations

from types import SimpleNamespace

import pyarrow as pa
import pytest

from vgi_sklearn.text import CountVectorizerFn, TfidfVectorizerFn, _text_column


def _docs() -> pa.Table:
    return pa.table(
        {
            "id": [1, 2, 3],
            "body": ["the cat sat on the mat", "the dog sat on the log", "cats and dogs are friends"],
        }
    )


def _args(**kw: object) -> SimpleNamespace:
    base = {
        "id": "id",
        "text": "body",
        "lowercase": True,
        "stop_words": "",
        "ngram_max": 1,
        "min_df": 1,
        "max_df": 1.0,
        "max_features": 0,
    }
    base.update(kw)
    return SimpleNamespace(**base)


class TestTextColumn:
    def test_explicit(self) -> None:
        assert _text_column(_docs().schema, "id", "body") == "body"

    def test_inferred_single_non_id(self) -> None:
        assert _text_column(_docs().schema, "id", "") == "body"

    def test_ambiguous_errors(self) -> None:
        schema = pa.schema([pa.field("id", pa.int64()), pa.field("a", pa.string()), pa.field("b", pa.string())])
        with pytest.raises(ValueError, match="could not infer the text column"):
            _text_column(schema, "id", "")

    def test_missing_errors(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            _text_column(_docs().schema, "id", "nope")


class TestCountVectorizer:
    def test_counts(self) -> None:
        out = CountVectorizerFn.encode(_docs(), _args())
        # 'the' appears twice in document 1
        the_in_1 = [v for i, t, v in zip(out["id"], out["term"], out["value"], strict=True) if i == 1 and t == "the"]
        assert the_in_1 == [2]
        assert all(isinstance(v, int) for v in out["value"])

    def test_stop_words_drop_the(self) -> None:
        out = CountVectorizerFn.encode(_docs(), _args(stop_words="english"))
        assert "the" not in set(out["term"])

    def test_bigrams(self) -> None:
        out = CountVectorizerFn.encode(_docs(), _args(ngram_max=2))
        assert any(" " in t for t in out["term"])

    def test_output_schema_int_value(self) -> None:
        schema = CountVectorizerFn._output_schema(_docs().schema, _args())
        assert schema.names == ["id", "term", "value"]
        assert schema.field("value").type == pa.int64()


class TestTfidfVectorizer:
    def test_weights_float_and_top_term(self) -> None:
        out = TfidfVectorizerFn.encode(_docs(), _args())
        assert all(isinstance(v, float) for v in out["value"])
        # document 3's highest-weighted term
        doc3 = [(t, v) for i, t, v in zip(out["id"], out["term"], out["value"], strict=True) if i == 3]
        top = max(doc3, key=lambda tv: tv[1])[0]
        assert top == "cats"

    def test_output_schema_float_value(self) -> None:
        schema = TfidfVectorizerFn._output_schema(_docs().schema, _args())
        assert schema.field("value").type == pa.float64()
