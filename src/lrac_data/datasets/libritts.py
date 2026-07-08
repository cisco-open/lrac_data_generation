"""LibriTTS train-clean adapter."""

from __future__ import annotations

import csv
from pathlib import Path

from lrac_data.models import InventoryItem, MediaKind

from .base import DatasetAdapter
from .common import read_text, require_unique
from .io import safe_extract_tar


class LibriTTSAdapter(DatasetAdapter):
    def fetch(self) -> Path:
        archives = self.download_remote_sources(
            "train_clean_100",
            "train_clean_360",
        )
        for archive in archives:
            safe_extract_tar(archive, self.extracted_dir)
        self.ensure_expected_files()
        return self.extracted_dir

    def _genders(self) -> dict[str, str]:
        path = self.extracted_dir / "LibriTTS" / "speakers.tsv"
        if not path.is_file():
            return {}
        genders: dict[str, str] = {}
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = csv.reader(stream, delimiter="\t")
            next(rows, None)
            for row in rows:
                if len(row) >= 2:
                    genders[row[0]] = row[1].strip().lower()[:1]
        return genders

    def inventory(self) -> list[InventoryItem]:
        self.ensure_expected_files()
        paths: list[Path] = []
        for split in self.option("splits", ["train-clean-100", "train-clean-360"]):
            paths.extend((self.extracted_dir / "LibriTTS" / split).rglob("*.wav"))
        pairs = require_unique(((path.stem, path) for path in sorted(paths)), self.config.id)
        genders = self._genders()
        records = []
        for source_id, path in pairs:
            speaker = source_id.split("_", 1)[0]
            records.append(
                self.item(
                    source_id,
                    MediaKind.SPEECH,
                    path,
                    speaker_id=f"libritts_{speaker}",
                    text=read_text(path.with_suffix(".normalized.txt")),
                    language="en",
                    gender=genders.get(speaker),
                )
            )
        return records
