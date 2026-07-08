"""GLOBE Hugging Face parquet adapter."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lrac_data.models import InventoryItem, MediaKind

from .base import DatasetAdapter
from .io import DownloadRequest, download_file, download_many


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
        return self._managed_path(self.extracted_dir / "inventory.jsonl")

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

    def fetch(self) -> Path:
        source = self.source("parquet_shards")
        if source.url is None:
            raise ValueError("GLOBE parquet_shards source requires a URL template")
        first = int(source.options.get("first", 0))
        last = int(source.options.get("last", 107))
        batch_size = int(source.options.get("batch_size", 256))
        if batch_size < 1:
            raise ValueError("GLOBE parquet batch_size must be at least 1")
        parquet_paths = download_many(
            (
                DownloadRequest(
                    url=source.url.format(index=index),
                    destination=self.download_dir / f"{index:04d}.parquet",
                )
                for index in range(first, last + 1)
            ),
            max_workers=self.workers,
            downloader=download_file,
        )

        try:
            import pyarrow.parquet as parquet  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("GLOBE preparation requires pyarrow") from exc

        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        partial_index = self._index_path.with_suffix(".jsonl.part")
        partial_index.unlink(missing_ok=True)
        seen: set[str] = set()
        fields = ("audio", "speaker_id", "transcript", "language", "gender")
        try:
            with partial_index.open("w", encoding="utf-8") as output:
                for parquet_path in parquet_paths:
                    bundle = parquet.ParquetFile(parquet_path)
                    available = set(bundle.schema_arrow.names)
                    missing = {"audio", "speaker_id"}.difference(available)
                    if missing:
                        names = ", ".join(sorted(missing))
                        raise ValueError(f"GLOBE parquet shard lacks required columns: {names}")
                    columns = [name for name in fields if name in available]
                    shard_rows = 0
                    for batch in bundle.iter_batches(batch_size=batch_size, columns=columns):
                        for sample in batch.to_pylist():
                            audio = sample["audio"]
                            if not isinstance(audio, Mapping):
                                raise ValueError(f"Malformed GLOBE audio value in {parquet_path}")
                            audio_bytes = audio.get("bytes")
                            if audio_bytes is None:
                                raise ValueError(
                                    f"GLOBE row has no embedded audio in {parquet_path}"
                                )
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
                            utterance = self._safe_component(
                                Path(original_name).stem,
                                "utterance ID",
                            )
                            source_id = f"{speaker}_{utterance}"
                            if source_id in seen:
                                raise ValueError(f"Duplicate GLOBE source ID: {source_id}")
                            seen.add(source_id)
                            shard_rows += 1
                            suffix = Path(original_name).suffix or ".flac"
                            relative_path = (
                                Path("train") / "audio" / speaker / f"{utterance}{suffix}"
                            )
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
                            output.write(
                                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                            )
                    expected_rows = bundle.metadata.num_rows
                    if shard_rows != expected_rows:
                        raise ValueError(
                            f"GLOBE parquet row-count mismatch in {parquet_path}: "
                            f"expected {expected_rows}, found {shard_rows}"
                        )
                output.flush()
                os.fsync(output.fileno())
            partial_index.replace(self._index_path)
        except BaseException:
            partial_index.unlink(missing_ok=True)
            raise
        self.ensure_expected_files()
        return self.extracted_dir

    def inventory(self) -> list[InventoryItem]:
        self.ensure_expected_files()
        records = []
        seen: set[str] = set()
        with self._index_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                data = json.loads(line)
                source_id = data.pop("source_id")
                relative_path = data.pop("source_path")
                if source_id in seen:
                    raise ValueError(
                        f"Duplicate GLOBE source ID {source_id!r} at line {line_number}"
                    )
                seen.add(source_id)
                source_path = self._managed_path(self.extracted_dir / relative_path)
                if not source_path.is_file():
                    raise FileNotFoundError(
                        f"GLOBE inventory references missing audio: {source_path}"
                    )
                records.append(
                    self.item(
                        source_id,
                        MediaKind.SPEECH,
                        source_path,
                        **data,
                    )
                )
        return sorted(records, key=lambda item: item.id)
