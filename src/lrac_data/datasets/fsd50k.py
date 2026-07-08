"""FSD50K development-audio adapter."""

from __future__ import annotations

from pathlib import Path

from lrac_data.models import InventoryItem, MediaKind

from .base import DatasetAdapter
from .inventory import FileTreeRule, build_file_inventory
from .io import (
    DownloadRequest,
    download_file,
    download_many,
    remove_derived_archive,
    safe_extract_zip,
    unsplit_zip,
)


class FSD50KAdapter(DatasetAdapter):
    def fetch(self) -> Path:
        self.download_dir.mkdir(parents=True, exist_ok=True)
        requests = []
        for source in self.config.sources:
            if source.url is None or source.filename is None:
                raise ValueError(f"FSD50K source {source.name!r} requires url and filename")
            requests.append(
                DownloadRequest(
                    url=source.url,
                    destination=self.download_dir / source.filename,
                    checksum=source.checksum,
                )
            )

        archives = download_many(
            requests,
            max_workers=self.workers,
            downloader=download_file,
        )
        for source, archive in zip(self.config.sources, archives, strict=True):
            if source.options.get("extract"):
                safe_extract_zip(archive, self.extracted_dir)

        joined = unsplit_zip(
            self.download_dir,
            "FSD50K.dev_audio.zip",
            self.download_dir / "FSD50K.dev_audio.unsplit.zip",
        )
        safe_extract_zip(joined, self.extracted_dir)
        self.ensure_expected_files()
        remove_derived_archive(joined)
        return self.extracted_dir

    def inventory(self) -> list[InventoryItem]:
        return build_file_inventory(
            self,
            (
                FileTreeRule(
                    root="FSD50K.dev_audio",
                    pattern="**/*.wav",
                    media_kind=MediaKind.NOISE,
                    source_id_prefix="fsd50k_",
                ),
            ),
        )
