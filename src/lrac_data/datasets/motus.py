"""MOTUS raw room-impulse-response adapter."""

from __future__ import annotations

from pathlib import Path

from lrac_data.models import InventoryItem, MediaKind

from .base import DatasetAdapter
from .inventory import FileTreeRule, build_file_inventory
from .io import safe_extract_zip


class MOTUSAdapter(DatasetAdapter):
    def fetch(self) -> Path:
        (archive,) = self.download_remote_sources("raw_rirs")
        safe_extract_zip(archive, self.extracted_dir / "raw_rirs")
        self.ensure_expected_files()
        return self.extracted_dir

    def inventory(self) -> list[InventoryItem]:
        return build_file_inventory(
            self,
            (
                FileTreeRule(
                    root="raw_rirs",
                    pattern="**/*.wav",
                    media_kind=MediaKind.RIR,
                    source_id_prefix="motus_",
                ),
            ),
        )
