"""Model registry: persist fitted estimators behind a swappable storage backend.

Phase 4 ships a local-disk store (``SKLEARN_MODELS_DIR``, default ``./models``).
The ``ModelStore`` interface is the seam where an S3/R2 backend drops in later
without touching ``models.py``.

Each model is two artifacts:
* ``<name>.skops``  -- the scikit-learn estimator, serialized with skops (a
  safe, non-pickle format that reconstructs only known types on load)
* ``<name>.json``   -- ``ModelMetadata`` (estimator type, ordered feature names,
  target, classes, hyperparameters, train score, library versions, timestamp)
"""

from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import skops.io as sio

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

# Estimators are persisted with skops (not pickle): loading reconstructs only a
# known set of types instead of executing arbitrary code. We additionally
# restrict the trusted set to the scikit-learn / numpy / scipy namespaces, so a
# crafted artifact cannot smuggle in an arbitrary callable (e.g. os.system).
_TRUSTED_PREFIXES = ("sklearn.", "numpy.", "numpy", "scipy.", "scipy")


class UntrustedModelError(ValueError):
    """Raised when a serialized model contains types outside the trusted namespaces."""


def _skops_dumps(estimator: Any) -> bytes:
    return sio.dumps(estimator)


def _skops_loads(data: bytes) -> Any:
    """Safely load a skops-serialized estimator, trusting only sklearn/numpy/scipy types."""
    untrusted = sio.get_untrusted_types(data=data)
    disallowed = [t for t in untrusted if not t.startswith(_TRUSTED_PREFIXES)]
    if disallowed:
        raise UntrustedModelError(f"refusing to load model containing untrusted type(s): {', '.join(disallowed)}")
    return sio.loads(data, trusted=untrusted)


class ModelNameError(ValueError):
    """Raised for model names that are empty or unsafe as a filename."""


class ModelNotFoundError(KeyError):
    """Raised when a requested model is not in the registry."""


def validate_name(name: str) -> str:
    if not name or not _NAME_RE.match(name) or "/" in name or ".." in name:
        raise ModelNameError(
            f"invalid model name {name!r}: use letters, digits, '_', '-', '.' and do not start with a separator"
        )
    return name


@dataclass(kw_only=True)
class ModelMetadata:
    """Everything needed to score new data and describe a stored model."""

    name: str
    estimator: str
    task: str  # "classification" | "regression"
    target: str
    feature_names: list[str]
    # Per-feature flag (aligned with feature_names): True for categorical (string)
    # columns, which fit one-hot-encodes and predict must rebuild as strings.
    categorical: list[bool] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    classes: list[Any] | None = None
    n_samples: int = 0
    n_features: int = 0
    train_score: float | None = None
    sklearn_version: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelMetadata:
        known = {f for f in cls.__dataclass_fields__}  # noqa: C416
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(kw_only=True)
class TransformerMetadata:
    """Everything needed to apply a stored, already-fitted transformer to new data."""

    name: str
    kind: str  # e.g. "standard_scaler", "pca"
    feature_names: list[str]  # input columns the transformer was fit on (aligned by name)
    output_names: list[str]  # output column names (mirror features, or component_1..k)
    output_int: bool = False  # True when the output is integer codes (kbins_discretizer)
    params: dict[str, Any] = field(default_factory=dict)
    n_features: int = 0
    n_output: int = 0
    sklearn_version: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TransformerMetadata:
        known = {f for f in cls.__dataclass_fields__}  # noqa: C416
        return cls(**{k: v for k, v in d.items() if k in known})


class ModelStore:
    """Abstract model store. Implementations persist (estimator, metadata) by name."""

    def save(self, estimator: Any, meta: ModelMetadata) -> None:
        raise NotImplementedError

    def load(self, name: str) -> tuple[Any, ModelMetadata]:
        raise NotImplementedError

    def load_meta(self, name: str) -> ModelMetadata:
        raise NotImplementedError

    def list(self) -> list[ModelMetadata]:
        raise NotImplementedError

    def delete(self, name: str) -> bool:
        raise NotImplementedError

    def exists(self, name: str) -> bool:
        raise NotImplementedError


class LocalDiskStore(ModelStore):
    """Stores models as ``<root>/<name>.skops`` + ``<root>/<name>.json``."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    def _paths(self, name: str) -> tuple[Path, Path]:
        validate_name(name)
        return self.root / f"{name}.skops", self.root / f"{name}.json"

    def save(self, estimator: Any, meta: ModelMetadata) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        model_path, meta_path = self._paths(meta.name)
        model_path.write_bytes(_skops_dumps(estimator))
        meta_path.write_text(json.dumps(meta.to_dict(), indent=2, default=str))

    def load(self, name: str) -> tuple[Any, ModelMetadata]:
        model_path, _ = self._paths(name)
        if not model_path.exists():
            raise ModelNotFoundError(name)
        return _skops_loads(model_path.read_bytes()), self.load_meta(name)

    def load_meta(self, name: str) -> ModelMetadata:
        _, meta_path = self._paths(name)
        if not meta_path.exists():
            raise ModelNotFoundError(name)
        return ModelMetadata.from_dict(json.loads(meta_path.read_text()))

    def list(self) -> list[ModelMetadata]:
        if not self.root.exists():
            return []
        out: list[ModelMetadata] = []
        for meta_path in sorted(self.root.glob("*.json")):
            try:
                out.append(ModelMetadata.from_dict(json.loads(meta_path.read_text())))
            except (json.JSONDecodeError, OSError):
                continue
        return out

    def delete(self, name: str) -> bool:
        model_path, meta_path = self._paths(name)
        existed = model_path.exists() or meta_path.exists()
        model_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        return existed

    def exists(self, name: str) -> bool:
        model_path, _ = self._paths(name)
        return model_path.exists()


_store: ModelStore | None = None


def get_store() -> ModelStore:
    """Return the process-wide model store, configured from the environment.

    ``SKLEARN_MODELS_DIR`` selects the local-disk root (default ``./models``).
    A future S3/R2 backend would be selected here behind the same interface.
    """
    global _store
    if _store is None:
        root = os.environ.get("SKLEARN_MODELS_DIR", "models")
        _store = LocalDiskStore(root)
    return _store


def set_store(store: ModelStore | None) -> None:
    """Override the process-wide store (used by tests)."""
    global _store
    _store = store


class TransformerStore:
    """Abstract store for fitted transformers (parallel to ``ModelStore``)."""

    def save(self, transformer: Any, meta: TransformerMetadata) -> None:
        raise NotImplementedError

    def load(self, name: str) -> tuple[Any, TransformerMetadata]:
        raise NotImplementedError

    def load_meta(self, name: str) -> TransformerMetadata:
        raise NotImplementedError

    def list(self) -> list[TransformerMetadata]:
        raise NotImplementedError

    def delete(self, name: str) -> bool:
        raise NotImplementedError


class LocalTransformerStore(TransformerStore):
    """Stores transformers as ``<root>/transformers/<name>.skops`` + ``.json``.

    The ``transformers/`` subdirectory keeps them out of the model namespace, so
    ``list_models`` and ``list_transformers`` never see each other's artifacts.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root) / "transformers"

    def _paths(self, name: str) -> tuple[Path, Path]:
        validate_name(name)
        return self.root / f"{name}.skops", self.root / f"{name}.json"

    def save(self, transformer: Any, meta: TransformerMetadata) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        obj_path, meta_path = self._paths(meta.name)
        obj_path.write_bytes(_skops_dumps(transformer))
        meta_path.write_text(json.dumps(meta.to_dict(), indent=2, default=str))

    def load(self, name: str) -> tuple[Any, TransformerMetadata]:
        obj_path, _ = self._paths(name)
        if not obj_path.exists():
            raise ModelNotFoundError(name)
        return _skops_loads(obj_path.read_bytes()), self.load_meta(name)

    def load_meta(self, name: str) -> TransformerMetadata:
        _, meta_path = self._paths(name)
        if not meta_path.exists():
            raise ModelNotFoundError(name)
        return TransformerMetadata.from_dict(json.loads(meta_path.read_text()))

    def list(self) -> list[TransformerMetadata]:
        if not self.root.exists():
            return []
        out: list[TransformerMetadata] = []
        for meta_path in sorted(self.root.glob("*.json")):
            try:
                out.append(TransformerMetadata.from_dict(json.loads(meta_path.read_text())))
            except (json.JSONDecodeError, OSError):
                continue
        return out

    def delete(self, name: str) -> bool:
        obj_path, meta_path = self._paths(name)
        existed = obj_path.exists() or meta_path.exists()
        obj_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        return existed


_transformer_store: TransformerStore | None = None


def get_transformer_store() -> TransformerStore:
    """Return the process-wide transformer store (same root as the model store)."""
    global _transformer_store
    if _transformer_store is None:
        root = os.environ.get("SKLEARN_MODELS_DIR", "models")
        _transformer_store = LocalTransformerStore(root)
    return _transformer_store


def set_transformer_store(store: TransformerStore | None) -> None:
    """Override the process-wide transformer store (used by tests)."""
    global _transformer_store
    _transformer_store = store


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Self-contained model BLOB (estimator + metadata in one value)
#
# Layout: 4-byte big-endian metadata-JSON length || metadata JSON || skops bytes.
# Lets a fitted model flow through SQL as a single BLOB column and live inside a
# DuckDB table instead of (or alongside) the on-disk registry. DuckDB BLOB
# values are capped near 2 GB, so very large ensembles may not fit.
# ---------------------------------------------------------------------------


def pack_model(estimator: Any, meta: ModelMetadata) -> bytes:
    """Serialize ``(estimator, metadata)`` into one self-describing BLOB."""
    est_bytes = _skops_dumps(estimator)
    meta_bytes = json.dumps(meta.to_dict(), default=str).encode("utf-8")
    return struct.pack(">I", len(meta_bytes)) + meta_bytes + est_bytes


def _split_blob(blob: bytes) -> tuple[bytes, bytes]:
    if len(blob) < 4:
        raise ValueError("not a valid sklearn model BLOB (too short)")
    (n,) = struct.unpack(">I", blob[:4])
    if len(blob) < 4 + n:
        raise ValueError("not a valid sklearn model BLOB (truncated metadata)")
    return blob[4 : 4 + n], blob[4 + n :]


def unpack_meta(blob: bytes) -> ModelMetadata:
    """Read just the metadata from a model BLOB (cheap; no estimator load)."""
    meta_bytes, _ = _split_blob(blob)
    return ModelMetadata.from_dict(json.loads(meta_bytes))


def unpack_model(blob: bytes) -> tuple[Any, ModelMetadata]:
    """Read both estimator and metadata from a model BLOB (skops safe-load)."""
    meta_bytes, est_bytes = _split_blob(blob)
    meta = ModelMetadata.from_dict(json.loads(meta_bytes))
    return _skops_loads(est_bytes), meta


def pack_transformer(transformer: Any, meta: TransformerMetadata) -> bytes:
    """Serialize ``(transformer, metadata)`` into one self-describing BLOB."""
    obj_bytes = _skops_dumps(transformer)
    meta_bytes = json.dumps(meta.to_dict(), default=str).encode("utf-8")
    return struct.pack(">I", len(meta_bytes)) + meta_bytes + obj_bytes


def unpack_transformer_meta(blob: bytes) -> TransformerMetadata:
    """Read just the metadata from a transformer BLOB (cheap; no object load)."""
    meta_bytes, _ = _split_blob(blob)
    return TransformerMetadata.from_dict(json.loads(meta_bytes))


def unpack_transformer(blob: bytes) -> tuple[Any, TransformerMetadata]:
    """Read both transformer and metadata from a transformer BLOB (skops safe-load)."""
    meta_bytes, obj_bytes = _split_blob(blob)
    meta = TransformerMetadata.from_dict(json.loads(meta_bytes))
    return _skops_loads(obj_bytes), meta
