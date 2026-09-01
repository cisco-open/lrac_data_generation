# Copyright 2026 Cisco Systems, Inc. and its affiliates
# Apache-2.0

"""GLOBE Hugging Face parquet adapter."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from lrac_data.models import InventoryItem, MediaKind
from lrac_data.state import atomic_write_text, canonical_json, sha256_file

from .base import DatasetAdapter
from .io import (
    DownloadRequest,
    download_many,
    require_checksum_map,
    trusted_file_sha256,
)

_SHARD_INDEXES = tuple(range(108))
_BATCH_SIZE = 256


class GLOBEAdapter(DatasetAdapter):
    def _managed_path(self, path: Path) -> Path:
        root = self.extracted_dir.resolve()
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(f"GLOBE managed path escapes {root}: {resolved}") from error
        if resolved == root:
            raise ValueError(f"GLOBE managed path must not be its extraction root: {root}")
        return resolved

    @property
    def _index_path(self) -> Path:
        return self.extracted_dir / "inventory.jsonl"

    @property
    def _shard_dir(self) -> Path:
        return self.extracted_dir / ".shards"

    def _fragment_path(self, parquet_path: Path) -> Path:
        return self._shard_dir / f"{parquet_path.stem}.jsonl"

    def _completion_path(self, parquet_path: Path) -> Path:
        return self._shard_dir / f"{parquet_path.stem}.complete.json"

    def _write_audio(self, path: Path, content: bytes) -> None:
        path = self._managed_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.name}.part")
        partial.unlink(missing_ok=True)
        try:
            descriptor = os.open(
                partial,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
            partial.replace(path)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _safe_component(value: str, name: str) -> str:
        if value in {"", ".", ".."} or "/" in value or "\\" in value:
            raise ValueError(f"Unsafe GLOBE {name}: {value!r}")
        return value

    def _fragment_records(
        self,
        path: Path,
        *,
        require_audio: bool = True,
    ) -> Iterator[dict[str, Any]]:
        seen: set[str] = set()
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Malformed GLOBE shard fragment {path}:{line_number}"
                    ) from error
                if not isinstance(record, dict):
                    raise ValueError(f"Malformed GLOBE shard fragment {path}:{line_number}")
                source_id = record.get("source_id")
                relative_path = record.get("source_path")
                if not isinstance(source_id, str) or not isinstance(relative_path, str):
                    raise ValueError(f"Malformed GLOBE shard fragment {path}:{line_number}")
                if source_id in seen:
                    raise ValueError(f"Duplicate GLOBE source ID in {path}: {source_id}")
                seen.add(source_id)
                if require_audio:
                    audio_path = self._managed_path(self.extracted_dir / relative_path)
                    if not audio_path.is_file():
                        raise FileNotFoundError(
                            f"GLOBE shard references missing audio: {audio_path}"
                        )
                yield record

    def _shard_is_complete(self, parquet_path: Path, input_sha256: str) -> bool:
        fragment = self._fragment_path(parquet_path)
        completion = self._completion_path(parquet_path)
        if not fragment.is_file() or not completion.is_file():
            return False
        try:
            state = json.loads(completion.read_text(encoding="utf-8"))
            if not isinstance(state, dict) or set(state) != {
                "input_sha256",
                "fragment_sha256",
                "row_count",
            }:
                return False
            if (
                state["input_sha256"] != input_sha256
                or not isinstance(state["fragment_sha256"], str)
                or type(state["row_count"]) is not int
                or state["row_count"] < 0
                or sha256_file(fragment) != state["fragment_sha256"]
            ):
                return False
            return sum(1 for _ in self._fragment_records(fragment)) == state["row_count"]
        except (OSError, TypeError, ValueError):
            return False

    def _process_shard(
        self,
        parquet_path: Path,
        input_sha256: str,
        parquet: Any,
    ) -> Path:
        fragment = self._fragment_path(parquet_path)
        fragment.parent.mkdir(parents=True, exist_ok=True)
        partial = fragment.with_name(f"{fragment.name}.part")
        partial.unlink(missing_ok=True)
        fields = ("audio", "speaker_id", "transcript", "language", "gender")
        seen: set[str] = set()
        shard_rows = 0
        try:
            bundle = parquet.ParquetFile(parquet_path)
            available = set(bundle.schema_arrow.names)
            missing = {"audio", "speaker_id"}.difference(available)
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(f"GLOBE parquet shard lacks required columns: {names}")
            columns = [name for name in fields if name in available]
            with partial.open("w", encoding="utf-8") as output:
                for batch in bundle.iter_batches(batch_size=_BATCH_SIZE, columns=columns):
                    for sample in batch.to_pylist():
                        audio = sample["audio"]
                        if not isinstance(audio, Mapping):
                            raise ValueError(f"Malformed GLOBE audio value in {parquet_path}")
                        audio_bytes = audio.get("bytes")
                        if audio_bytes is None:
                            raise ValueError(f"GLOBE row has no embedded audio in {parquet_path}")
                        if isinstance(audio_bytes, memoryview):
                            audio_bytes = audio_bytes.tobytes()
                        elif isinstance(audio_bytes, bytearray):
                            audio_bytes = bytes(audio_bytes)
                        if not isinstance(audio_bytes, bytes):
                            raise ValueError(f"Malformed GLOBE audio bytes in {parquet_path}")
                        speaker = self._optional_string(sample["speaker_id"])
                        if not speaker:
                            raise ValueError(f"GLOBE row has no speaker_id in {parquet_path}")
                        speaker = self._safe_component(speaker, "speaker ID")
                        original_name = Path(str(audio.get("path") or "audio.flac")).name
                        utterance = self._safe_component(Path(original_name).stem, "utterance ID")
                        source_id = f"{speaker}_{utterance}"
                        if source_id in seen:
                            raise ValueError(f"Duplicate GLOBE source ID: {source_id}")
                        seen.add(source_id)
                        shard_rows += 1
                        suffix = Path(original_name).suffix or ".flac"
                        relative_path = Path("train") / "audio" / speaker / f"{utterance}{suffix}"
                        audio_path = self._managed_path(self.extracted_dir / relative_path)
                        self._write_audio(audio_path, audio_bytes)
                        gender_name = self._optional_string(sample.get("gender"))
                        record = {
                            "source_id": source_id,
                            "source_path": relative_path.as_posix(),
                            "speaker_id": f"globe_{speaker}",
                            "text": self._optional_string(sample.get("transcript")),
                            "language": self._optional_string(sample.get("language")) or "en",
                            "gender": gender_name[0].lower() if gender_name else None,
                        }
                        output.write(f"{canonical_json(record)}\n")
                expected_rows = bundle.metadata.num_rows
                if shard_rows != expected_rows:
                    raise ValueError(
                        f"GLOBE parquet row-count mismatch in {parquet_path}: "
                        f"expected {expected_rows}, found {shard_rows}"
                    )
                output.flush()
                os.fsync(output.fileno())
            partial.replace(fragment)
            state = {
                "input_sha256": input_sha256,
                "fragment_sha256": sha256_file(fragment),
                "row_count": shard_rows,
            }
            atomic_write_text(
                self._completion_path(parquet_path),
                f"{canonical_json(state)}\n",
            )
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        return fragment

    def _assemble_index(self, fragments: list[Path]) -> None:
        partial = self._index_path.with_suffix(".jsonl.part")
        partial.unlink(missing_ok=True)
        seen: set[str] = set()
        try:
            with partial.open("w", encoding="utf-8") as output:
                for fragment in fragments:
                    for record in self._fragment_records(fragment, require_audio=False):
                        source_id = record["source_id"]
                        if source_id in seen:
                            raise ValueError(f"Duplicate GLOBE source ID: {source_id}")
                        seen.add(source_id)
                        output.write(f"{canonical_json(record)}\n")
                output.flush()
                os.fsync(output.fileno())
            partial.replace(self._index_path)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise

    def fetch(self) -> Path:
        source = self.source("parquet_shards")
        if source.url is None:
            raise ValueError("GLOBE parquet_shards source requires a URL template")
        filenames = tuple(f"{index:04d}.parquet" for index in _SHARD_INDEXES)
        checksums = require_checksum_map(
            source.artifact_checksums,
            filenames,
            label="GLOBE parquet_shards",
        )
        parquet_paths = download_many(
            (
                DownloadRequest(
                    url=source.url.format(index=index),
                    destination=self.download_dir / filename,
                    checksum=checksums[filename],
                )
                for index, filename in zip(_SHARD_INDEXES, filenames, strict=True)
            ),
            max_workers=self.workers,
        )

        try:
            import pyarrow.parquet as parquet  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("GLOBE preparation requires pyarrow") from exc

        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        fragments: list[Path] = []
        for parquet_path in parquet_paths:
            input_sha256 = trusted_file_sha256(parquet_path)
            fragment = self._fragment_path(parquet_path)
            if not self._shard_is_complete(parquet_path, input_sha256):
                fragment = self._process_shard(
                    parquet_path,
                    input_sha256,
                    parquet,
                )
            fragments.append(fragment)
        self._assemble_index(fragments)
        return self.extracted_dir

    def fetch_selected(self, items: tuple[InventoryItem, ...]) -> None:
        """Repair missing embedded audio without discarding completed shards."""

        missing = [item.source_path for item in items if not item.source_path.is_file()]
        if not missing:
            return
        self.fetch()
        remaining = [path for path in missing if not path.is_file()]
        if remaining:
            raise FileNotFoundError(remaining[0])

    def inventory(self) -> list[InventoryItem]:
        records = []
        for indexed in self._fragment_records(self._index_path):
            data = dict(indexed)
            source_id = data.pop("source_id")
            relative_path = data.pop("source_path")
            records.append(
                self.item(
                    source_id,
                    MediaKind.SPEECH,
                    self.extracted_dir / relative_path,
                    **data,
                )
            )
        return sorted(records, key=lambda item: item.id)
