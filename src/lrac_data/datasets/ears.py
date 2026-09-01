# Copyright 2026 Cisco Systems, Inc. and its affiliates
# Apache-2.0

"""Meta EARS corpus adapter."""

from __future__ import annotations

from pathlib import Path

from lrac_data.models import InventoryItem, MediaKind

from .base import DatasetAdapter
from .common import read_json, require_unique
from .io import (
    DownloadRequest,
    download_many,
    require_checksum_map,
    safe_extract_zip,
)

_PARTICIPANT_IDS = tuple(f"p{participant:03d}" for participant in range(1, 108))


class EARSAdapter(DatasetAdapter):
    def fetch(self) -> Path:
        participants = self.source("participants")
        if participants.url is None:
            raise ValueError("EARS participants requires a URL template")
        filenames = tuple(f"{participant_id}.zip" for participant_id in _PARTICIPANT_IDS)
        checksums = require_checksum_map(
            participants.artifact_checksums,
            filenames,
            label="EARS participants",
        )
        participant_requests = [
            DownloadRequest(
                url=participants.url.format(speaker=participant_id),
                destination=self.download_dir / filename,
                checksum=checksums[filename],
            )
            for participant_id, filename in zip(_PARTICIPANT_IDS, filenames, strict=True)
        ]
        participant_archives = download_many(
            participant_requests,
            max_workers=self.workers,
        )
        for archive in participant_archives:
            safe_extract_zip(archive, self.extracted_dir)

        (metadata_archive,) = self.download_remote_sources("metadata")
        metadata_root = self.extracted_dir / "metadata"
        safe_extract_zip(metadata_archive, metadata_root)
        return self.extracted_dir

    def inventory(self) -> list[InventoryItem]:
        transcript_files = sorted((self.extracted_dir / "metadata").rglob("transcripts.json"))
        statistic_files = sorted((self.extracted_dir / "metadata").rglob("speaker_statistics.json"))
        if len(transcript_files) != 1 or len(statistic_files) != 1:
            raise FileNotFoundError("EARS requires one transcript and one speaker-statistics file")
        transcripts = read_json(transcript_files[0])
        statistics = read_json(statistic_files[0])

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
