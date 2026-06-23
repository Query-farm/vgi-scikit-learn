"""Unit tests for stored-transformer helpers + the transformer BLOB/registry.

The full fit/apply lifecycle is covered by test/sql/sklearn_stored_transforms.test.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from vgi_sklearn.registry import (
    TransformerMetadata,
    pack_transformer,
    unpack_transformer,
    unpack_transformer_meta,
)
from vgi_sklearn.stored_transforms import (
    TRANSFORMER_KINDS,
    _build,
    _output_names,
    _parse_params,
)


class TestBuild:
    def test_known_kinds_construct(self) -> None:
        for kind in TRANSFORMER_KINDS:
            assert _build(kind, {}) is not None

    def test_kbins_forces_ordinal_encoding(self) -> None:
        enc = _build("kbins_discretizer", {"n_bins": 4})
        assert enc.encode == "ordinal"
        assert enc.n_bins == 4

    def test_pca_params_passed(self) -> None:
        assert _build("pca", {"n_components": 2}).n_components == 2

    def test_unknown_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown transformer kind"):
            _build("nope", {})

    def test_bad_params_raise(self) -> None:
        with pytest.raises(ValueError, match="invalid params"):
            _build("standard_scaler", {"not_a_param": 1})


class TestOutputNames:
    def test_mirror_when_width_preserved(self) -> None:
        assert _output_names(["a", "b", "c"], 3) == ["a", "b", "c"]

    def test_components_when_reduced(self) -> None:
        assert _output_names(["a", "b", "c"], 2) == ["component_1", "component_2"]


def test_parse_params() -> None:
    assert _parse_params("") == {}
    assert _parse_params("  ") == {}
    assert _parse_params('{"n_components": 2}') == {"n_components": 2}
    with pytest.raises(ValueError, match="must be a JSON object"):
        _parse_params("[1, 2]")


class TestTransformerBlob:
    def test_roundtrip_preserves_state_and_meta(self) -> None:
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler = StandardScaler().fit(x)
        meta = TransformerMetadata(
            name="sc",
            kind="standard_scaler",
            feature_names=["a", "b"],
            output_names=["a", "b"],
            n_features=2,
            n_output=2,
        )
        blob = pack_transformer(scaler, meta)

        # cheap metadata read
        assert unpack_transformer_meta(blob).kind == "standard_scaler"

        # full load reproduces the fitted transform
        loaded, loaded_meta = unpack_transformer(blob)
        assert loaded_meta.output_names == ["a", "b"]
        assert np.allclose(loaded.transform(x), scaler.transform(x))

    def test_pca_meta_records_component_outputs(self) -> None:
        x = np.random.default_rng(0).normal(size=(20, 4))
        pca = PCA(n_components=2).fit(x)
        meta = TransformerMetadata(
            name="p",
            kind="pca",
            feature_names=["a", "b", "c", "d"],
            output_names=["component_1", "component_2"],
            n_features=4,
            n_output=2,
        )
        loaded, loaded_meta = unpack_transformer(pack_transformer(pca, meta))
        assert loaded_meta.output_names == ["component_1", "component_2"]
        assert loaded.transform(x).shape == (20, 2)
