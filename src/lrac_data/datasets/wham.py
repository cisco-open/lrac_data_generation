"""High-resolution WHAM! background-noise adapter."""

from __future__ import annotations

from pathlib import Path

from lrac_data.models import InventoryItem, MediaKind

from .base import DatasetAdapter
from .inventory import FileTreeRule, build_file_inventory
from .io import safe_extract_zip


class WHAMAdapter(DatasetAdapter):
    def fetch(self) -> Path:
        (archive,) = self.download_remote_sources("high_res_wham")
        safe_extract_zip(archive, self.extracted_dir)
        self.ensure_expected_files()
        return self.extracted_dir

    def inventory(self) -> list[InventoryItem]:
        return build_file_inventory(
            self,
            (
                FileTreeRule(
                    root="high_res_wham/audio",
                    pattern="**/*.wav",
                    media_kind=MediaKind.NOISE,
                ),
            ),
        )
