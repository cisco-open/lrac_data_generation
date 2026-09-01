# Copyright 2026 Cisco Systems, Inc. and its affiliates
# Apache-2.0

"""Free Music Archive medium-subset adapter."""

from __future__ import annotations

from pathlib import Path

from lrac_data.models import InventoryItem, MediaKind

from .base import DatasetAdapter
from .inventory import build_file_inventory
from .io import safe_extract_zip


class FMAAdapter(DatasetAdapter):
    def fetch(self) -> Path:
        (archive,) = self.download_remote_sources("medium")
        safe_extract_zip(archive, self.extracted_dir)
        return self.extracted_dir

    def inventory(self) -> list[InventoryItem]:
        return build_file_inventory(
            self,
            "fma_medium",
            "**/*.mp3",
            MediaKind.NOISE,
            "fma_",
        )
