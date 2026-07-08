"""Free Music Archive medium-subset adapter."""

from __future__ import annotations

from pathlib import Path

from lrac_data.models import InventoryItem, MediaKind

from .base import DatasetAdapter
from .inventory import FileTreeRule, build_file_inventory
from .io import safe_extract_zip


class FMAAdapter(DatasetAdapter):
    def fetch(self) -> Path:
        archives = self.download_remote_sources("metadata", "medium")
        for archive in archives:
            safe_extract_zip(archive, self.extracted_dir)
        self.ensure_expected_files()
        return self.extracted_dir

    def inventory(self) -> list[InventoryItem]:
        return build_file_inventory(
            self,
            (
                FileTreeRule(
                    root="fma_medium",
                    pattern="**/*.mp3",
                    media_kind=MediaKind.NOISE,
                    source_id_prefix="fma_",
                ),
            ),
        )
