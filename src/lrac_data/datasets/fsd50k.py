# Copyright 2026 Cisco Systems, Inc. and its affiliates
# Apache-2.0

"""FSD50K development-audio adapter."""

from __future__ import annotations

from pathlib import Path

from lrac_data.models import InventoryItem, MediaKind

from .base import DatasetAdapter
from .inventory import build_file_inventory
from .io import safe_extract_zip, unsplit_zip

_SOURCES = ("audio_zip", "audio_z01", "audio_z02", "audio_z03", "audio_z04", "audio_z05")


class FSD50KAdapter(DatasetAdapter):
    def fetch(self) -> Path:
        self.download_remote_sources(*_SOURCES)

        joined = unsplit_zip(
            self.download_dir,
            "FSD50K.dev_audio.zip",
            self.download_dir / "FSD50K.dev_audio.unsplit.zip",
        )
        safe_extract_zip(joined, self.extracted_dir)
        joined.unlink(missing_ok=True)
        return self.extracted_dir

    def inventory(self) -> list[InventoryItem]:
        return build_file_inventory(
            self,
            "FSD50K.dev_audio",
            "**/*.wav",
            MediaKind.NOISE,
            "fsd50k_",
        )
