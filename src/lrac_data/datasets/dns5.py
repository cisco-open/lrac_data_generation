"""Microsoft DNS Challenge 5 speech, noise, and impulse-response adapter."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lrac_data.models import InventoryItem, MediaKind

from .base import DatasetAdapter
from .common import require_unique
from .io import (
    DownloadRequest,
    download_file,
    download_many,
    safe_extract_multipart_tar,
    safe_extract_tar,
)


class DNS5Adapter(DatasetAdapter):
    def fetch(self) -> Path:
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_dir.mkdir(parents=True, exist_ok=True)

        speech, speech_url, _ = self.remote_source("read_speech_parts")
        suffixes = self.option(
            "speech_part_suffixes",
            [
                "aa",
                "ab",
                "ac",
                "ad",
                "ae",
                "af",
                "ag",
                "ah",
                "ai",
                "aj",
                "ak",
                "al",
                "am",
                "an",
                "ao",
                "ap",
                "aq",
                "ar",
                "as",
                "at",
                "au",
            ],
        )
        checksums = speech.options.get("checksums", {})
        speech_requests = [
            DownloadRequest(
                url=speech_url.format(suffix=suffix),
                destination=self.download_dir / "speech" / f"read_speech.tgz.part{suffix}",
                checksum=checksums.get(suffix),
            )
            for suffix in suffixes
        ]
        part_paths = download_many(
            speech_requests,
            max_workers=self.workers,
            downloader=download_file,
        )
        speech_root = self.extracted_dir / "Track1_Headset"
        safe_extract_multipart_tar(part_paths, speech_root)

        noise_sources = [
            source for source in self.config.sources if source.name.startswith("noise_")
        ]
        noise_requests = []
        for source in noise_sources:
            if source.url is None or source.filename is None:
                raise ValueError(f"DNS5 source {source.name!r} requires url and filename")
            noise_requests.append(
                DownloadRequest(
                    url=source.url,
                    destination=self.download_dir / "noise" / source.filename,
                    checksum=source.checksum,
                )
            )

        noise_archives = download_many(
            noise_requests,
            max_workers=self.workers,
            downloader=download_file,
        )

        def extract_noise(index: int) -> None:
            source = noise_sources[index]
            archive = noise_archives[index]
            shard = source.options["shard"]
            shard_root = self.extracted_dir / "datasets_fullband" / "noise_fullband" / shard
            safe_extract_tar(
                archive,
                shard_root,
                strip_prefix="datasets_fullband/noise_fullband",
            )

        with ThreadPoolExecutor(max_workers=min(self.workers, 4)) as executor:
            list(executor.map(extract_noise, range(len(noise_archives))))

        rir, rir_url, rir_filename = self.remote_source("impulse_responses")
        rir_archive = download_file(
            rir_url,
            self.download_dir / rir_filename,
            checksum=rir.checksum,
        )
        safe_extract_tar(rir_archive, self.extracted_dir)

        self.ensure_expected_files()
        return self.extracted_dir

    def inventory(self) -> list[InventoryItem]:
        self.ensure_expected_files()
        records: list[InventoryItem] = []

        speech_paths = sorted((self.extracted_dir / "Track1_Headset" / "mnt").rglob("*.wav"))
        stem_counts: defaultdict[str, int] = defaultdict(int)
        for path in speech_paths:
            stem_counts[path.stem] += 1
            occurrence = stem_counts[path.stem]
            source_id = path.stem if occurrence == 1 else f"{path.stem}({occurrence})"
            parts = path.stem.split("_")
            if len(parts) < 6 or parts[4] != "reader":
                raise ValueError(f"Unexpected DNS5 LibriVox filename: {path.name}")
            records.append(
                self.item(
                    source_id,
                    MediaKind.SPEECH,
                    path,
                    speaker_id=f"dns5_{parts[4]}_{parts[5]}",
                    language="en",
                )
            )

        noise_root = self.extracted_dir / "datasets_fullband" / "noise_fullband"
        noise = require_unique(
            ((path.stem, path) for path in sorted(noise_root.rglob("*.wav"))),
            self.config.id,
        )
        records.extend(self.item(source_id, MediaKind.NOISE, path) for source_id, path in noise)

        rir_root = self.extracted_dir / "datasets_fullband" / "impulse_responses"
        rir_paths = sorted(rir_root.rglob("*.wav"))
        rirs = require_unique(
            (
                (
                    path.relative_to(rir_root).with_suffix("").as_posix(),
                    path,
                )
                for path in rir_paths
            ),
            self.config.id,
        )
        records.extend(self.item(source_id, MediaKind.RIR, path) for source_id, path in rirs)
        return sorted(records, key=lambda item: item.id)
