"""Urgent Track 1 subsets of Multilingual LibriSpeech."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lrac_data.models import InventoryItem, MediaKind

from .base import DatasetAdapter
from .common import read_transcript_gzip, require_unique
from .io import (
    DownloadRequest,
    download_file,
    download_many,
    require_checksum_map,
    safe_extract_tar,
)


class MLSAdapter(DatasetAdapter):
    def _metadata_dir(self) -> Path:
        try:
            metadata_source = self.source("source_metadata")
        except KeyError:
            # Retain a simple fallback for standalone fixtures and 2025 checkouts.
            return self.repo_root / "datafiles" / "mls"
        if metadata_source.path is None:
            raise ValueError("MLS source_metadata source requires a local path")
        return metadata_source.path

    def _shard_list(self, language: str) -> Path:
        return self._metadata_dir() / f"mls_{language}_train_track1_data.txt"

    def fetch(self) -> Path:
        source = self.source("track1_shards")
        if source.url is None:
            raise ValueError("MLS track1_shards source requires a URL template")
        artifacts: list[tuple[str, str, str, Path]] = []
        for language in self.option("languages", ["german", "french", "spanish"]):
            shard_list = self._shard_list(language)
            if not shard_list.is_file():
                raise FileNotFoundError(f"MLS shard list is missing: {shard_list}")
            for remote_path in shard_list.read_text(encoding="utf-8").splitlines():
                remote_path = remote_path.strip()
                if not remote_path:
                    continue
                path = Path(remote_path)
                archive_name = f"{path.parent.name}_{path.name}"
                checksum_key = f"{language}/{archive_name}"
                artifacts.append(
                    (
                        language,
                        remote_path,
                        checksum_key,
                        self.download_dir / language / archive_name,
                    )
                )

        checksums = require_checksum_map(
            source.artifact_checksums,
            (checksum_key for _, _, checksum_key, _ in artifacts),
            label="MLS track1_shards",
        )
        requests = [
            DownloadRequest(
                url=source.url.format(path=remote_path),
                destination=destination,
                checksum=checksums[checksum_key],
            )
            for _, remote_path, checksum_key, destination in artifacts
        ]
        languages = [language for language, _, _, _ in artifacts]

        archives = download_many(
            requests,
            max_workers=self.workers,
            downloader=download_file,
        )

        def extract_shard(language_archive: tuple[str, Path]) -> None:
            language, archive = language_archive
            destination = self.extracted_dir / language / "train" / "audio"
            safe_extract_tar(archive, destination)

        with ThreadPoolExecutor(max_workers=min(self.workers, 4)) as executor:
            list(executor.map(extract_shard, zip(languages, archives, strict=True)))
        self.ensure_expected_files()
        return self.extracted_dir

    def inventory(self) -> list[InventoryItem]:
        self.ensure_expected_files()
        records: list[InventoryItem] = []
        for language in self.option("languages", ["german", "french", "spanish"]):
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
                        speaker_id=fields[2],
                        text=transcripts.get(source_id),
                        language=language,
                    )
                )
        return sorted(records, key=lambda item: item.id)
