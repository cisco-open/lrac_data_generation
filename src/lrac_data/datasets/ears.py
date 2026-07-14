"""Meta EARS corpus adapter."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path

from lrac_data.models import InventoryItem, MediaKind

from .base import DatasetAdapter
from .common import read_json, require_unique
from .io import (
    DownloadRequest,
    download_file,
    download_many,
    require_checksum_map,
    safe_extract_zip,
)


class EARSAdapter(DatasetAdapter):
    def fetch(self) -> Path:
        participants, participants_url, _ = self.remote_source("participants")
        first = int(participants.options.get("first", 1))
        last = int(participants.options.get("last", 107))
        participant_ids = tuple(f"p{participant:03d}" for participant in range(first, last + 1))
        filenames = tuple(f"{participant_id}.zip" for participant_id in participant_ids)
        checksums = require_checksum_map(
            participants.artifact_checksums,
            filenames,
            label="EARS participants",
        )
        participant_requests = [
            DownloadRequest(
                url=participants_url.format(speaker=participant_id),
                destination=self.download_dir / filename,
                checksum=checksums[filename],
            )
            for participant_id, filename in zip(participant_ids, filenames, strict=True)
        ]
        participant_archives = download_many(
            participant_requests,
            max_workers=self.workers,
            downloader=download_file,
        )
        with ThreadPoolExecutor(max_workers=min(self.workers, 4)) as executor:
            list(
                executor.map(
                    safe_extract_zip,
                    participant_archives,
                    repeat(self.extracted_dir),
                )
            )

        metadata, metadata_url, metadata_filename = self.remote_source("metadata")
        metadata_checksums = require_checksum_map(
            metadata.artifact_checksums,
            (metadata_filename,),
            label="EARS metadata",
        )
        metadata_archive = download_file(
            metadata_url,
            self.download_dir / metadata_filename,
            checksum=metadata_checksums[metadata_filename],
        )
        metadata_root = self.extracted_dir / "metadata"
        safe_extract_zip(metadata_archive, metadata_root)
        self.ensure_expected_files()
        return self.extracted_dir

    def inventory(self) -> list[InventoryItem]:
        self.ensure_expected_files()
        transcript_files = sorted((self.extracted_dir / "metadata").rglob("transcripts.json"))
        statistic_files = sorted((self.extracted_dir / "metadata").rglob("speaker_statistics.json"))
        transcripts = read_json(transcript_files[0]) if transcript_files else {}
        statistics = read_json(statistic_files[0]) if statistic_files else {}

        paths = sorted(self.extracted_dir.glob("p[0-9][0-9][0-9]/**/*.wav"))
        pairs = require_unique(
            ((f"{path.parent.name}_{path.stem}", path) for path in paths),
            self.config.id,
        )
        records = []
        for source_id, path in pairs:
            speaker = source_id.split("_", 1)[0]
            gender_name = statistics.get(speaker, {}).get("gender")
            gender = {"male": "m", "female": "f"}.get(gender_name, "o")
            records.append(
                self.item(
                    source_id,
                    MediaKind.SPEECH,
                    path,
                    speaker_id=f"ears_{speaker}",
                    text=transcripts.get(path.stem),
                    language="en",
                    gender=gender,
                )
            )
        return records
