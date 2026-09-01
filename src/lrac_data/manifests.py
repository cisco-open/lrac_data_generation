# Copyright 2026 Cisco Systems, Inc. and its affiliates
# Apache-2.0

"""Deterministic JSONL serialization for LRAC data contracts."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from lrac_data.models import ManifestItem

ModelT = TypeVar("ModelT", bound=BaseModel)


class ManifestError(ValueError):
    """Raised when a JSONL contract cannot be decoded or is inconsistent."""


def write_jsonl(
    path: str | Path,
    records: Iterable[BaseModel | Mapping[str, Any]],
) -> Path:
    """Atomically write canonical JSONL in content-sorted order.

    Object keys and records are sorted, making output byte-identical for the same
    record set regardless of input iteration order.  The destination directory is
    created only when this function is called.
    """

    destination = Path(path)
    lines = sorted(_canonical_line(record) for record in records)
    return _write_canonical_lines(destination, lines)


def read_jsonl(path: str | Path, model_type: type[ModelT]) -> tuple[ModelT, ...]:
    """Read and validate JSONL records as ``model_type`` instances."""

    source = Path(path)
    records: list[ModelT] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ManifestError(f"{source}:{line_number}: blank JSONL line")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ManifestError(f"{source}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(value, dict):
                raise ManifestError(
                    f"{source}:{line_number}: expected a JSON object, got {type(value).__name__}"
                )
            try:
                records.append(model_type.model_validate(value))
            except ValidationError as error:
                raise ManifestError(
                    f"{source}:{line_number}: invalid {model_type.__name__}: {error}"
                ) from error
    return tuple(records)


def write_ordered_manifest(path: str | Path, records: Iterable[ManifestItem]) -> Path:
    """Atomically stream records that are already strictly ordered by stable ID."""

    destination = Path(path)
    previous_id: str | None = None

    def lines() -> Iterable[str]:
        nonlocal previous_id
        for record in records:
            if record.id == previous_id:
                raise ManifestError(f"{destination}: duplicate manifest ID {record.id!r}")
            if previous_id is not None and record.id < previous_id:
                raise ManifestError(
                    f"{destination}: manifest IDs are not strictly ordered: {record.id!r}"
                )
            previous_id = record.id
            yield _canonical_line(record)

    return _write_canonical_lines(destination, lines())


def read_manifest(path: str | Path) -> tuple[ManifestItem, ...]:
    """Read and validate a final manifest, preserving its on-disk order."""

    records = read_jsonl(path, ManifestItem)
    _ensure_unique_ids(records, path)
    return records


def _ensure_unique_ids(records: Iterable[ManifestItem], path: str | Path) -> None:
    seen: set[str] = set()
    for record in records:
        if record.id in seen:
            raise ManifestError(f"{path}: duplicate manifest ID {record.id!r}")
        seen.add(record.id)


def _write_canonical_lines(path: Path, lines: Iterable[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            for line in lines:
                stream.write(line)
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _canonical_line(record: BaseModel | Mapping[str, Any]) -> str:
    if isinstance(record, BaseModel):
        value = record.model_dump(mode="json", exclude_none=True)
    elif isinstance(record, Mapping):
        value = dict(record)
    else:
        raise TypeError(
            f"JSONL records must be Pydantic models or mappings, got {type(record).__name__}"
        )
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=_json_default,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")
