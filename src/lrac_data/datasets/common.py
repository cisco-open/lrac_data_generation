"""Small, side-effect-free helpers for corpus inventory implementations."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def require_unique(items: Iterable[tuple[str, Path]], dataset: str) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    seen: dict[str, Path] = {}
    for source_id, path in items:
        if source_id in seen:
            raise ValueError(
                f"Duplicate source ID {source_id!r} in {dataset}: {seen[source_id]} and {path}"
            )
        seen[source_id] = path
        result.append((source_id, path))
    return result


def read_text(path: Path, default: str | None = None) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return default


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def read_transcript_gzip(path: Path, prefix: str = "") -> dict[str, str]:
    transcripts: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            try:
                source_id, text = line.split(maxsplit=1)
            except ValueError as exc:
                raise ValueError(f"Malformed transcript line {line_number} in {path}") from exc
            key = f"{prefix}{source_id}"
            if key in transcripts:
                raise ValueError(f"Duplicate transcript ID {key!r} in {path}")
            transcripts[key] = text
    return transcripts
