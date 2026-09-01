# Copyright 2026 Cisco Systems, Inc. and its affiliates
# Apache-2.0

"""Urgent Track 1 subsets of Multilingual LibriSpeech."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from lrac_data.models import InventoryItem, MediaKind

from .base import DatasetAdapter
from .common import read_transcript_gzip, require_unique
from .io import (
    DownloadRequest,
    download_many,
    safe_extract_tar,
)

_LANGUAGES = ("german", "french", "spanish")


class MLSAdapter(DatasetAdapter):
    def _metadata_dir(self) -> Path:
        metadata_source = self.source("source_metadata")
        if metadata_source.path is None:
            raise ValueError("MLS source_metadata source requires a local path")
        return metadata_source.path

    def fetch(self) -> Path:
        source = self.source("track1_shards")
        if source.url is None:
            raise ValueError("MLS track1_shards source requires a URL template")
        artifacts: list[tuple[str, str, str, Path]] = []
        for remote_path, checksum in source.artifact_checksums.items():
            parts = PurePosixPath(remote_path).parts
            if (
                len(parts) != 5
                or parts[0] not in _LANGUAGES
                or parts[1:3] != ("train_track1", "audio")
                or not parts[3].isdecimal()
                or not parts[4].removesuffix(".tar.gz").isdecimal()
            ):
                raise ValueError(f"Invalid MLS shard path: {remote_path!r}")
            language = parts[0]
            archive_name = f"{parts[3]}_{parts[4]}"
            artifacts.append(
                (
                    language,
                    remote_path,
                    checksum,
                    self.download_dir / language / archive_name,
                )
            )
        if {language for language, *_ in artifacts} != set(_LANGUAGES):
            raise ValueError("MLS shard checksums must cover German, French, and Spanish")

        requests = [
            DownloadRequest(
                url=source.url.format(path=remote_path),
                destination=destination,
                checksum=checksum,
            )
            for _, remote_path, checksum, destination in artifacts
        ]

        archives = download_many(
            requests,
            max_workers=self.workers,
        )

        def extract_shard(language_archive: tuple[str, Path]) -> None:
            language, archive = language_archive
            destination = self.extracted_dir / language / "train" / "audio"
            safe_extract_tar(archive, destination)

        with ThreadPoolExecutor(max_workers=min(self.workers, 4)) as executor:
            languages = (language for language, *_ in artifacts)
            list(executor.map(extract_shard, zip(languages, archives, strict=True)))
        return self.extracted_dir

    def inventory(self) -> list[InventoryItem]:
        records: list[InventoryItem] = []
        for language in _LANGUAGES:
            transcript_path = self._metadata_dir() / f"{language}_train_transcripts.gz"
            transcripts = read_transcript_gzip(transcript_path, prefix=f"mls_{language}_")
            paths = sorted((self.extracted_dir / language).rglob("*.flac"))
            pairs = require_unique(
                ((f"mls_{language}_{path.stem}", path) for path in paths),
                f"{self.config.id}/{language}",
            )
            for source_id, path in pairs:
                fields = source_id.split("_")
                if len(fields) < 4:
                    raise ValueError(f"Unexpected MLS filename: {path.name}")
                records.append(
                    self.item(
                        source_id,
                        MediaKind.SPEECH,
                        path,
                        speaker_id=f"mls_{language}_{fields[2]}",
                        text=transcripts.get(source_id),
                        language=language,
                    )
                )
        return sorted(records, key=lambda item: item.id)
