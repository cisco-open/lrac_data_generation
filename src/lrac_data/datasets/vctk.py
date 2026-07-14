"""VCTK 0.92 adapter."""

from __future__ import annotations

from pathlib import Path

from lrac_data.models import InventoryItem, MediaKind

from .base import DatasetAdapter
from .common import read_text, require_unique
from .io import download_file, require_checksum_map, safe_extract_zip


class VCTKAdapter(DatasetAdapter):
    def fetch(self) -> Path:
        source, url, filename = self.remote_source("corpus")
        checksums = require_checksum_map(
            source.artifact_checksums,
            (filename,),
            label="VCTK corpus",
        )
        outer = download_file(
            url,
            self.download_dir / filename,
            checksum=checksums[filename],
        )
        outer_root = self.extracted_dir / "outer"
        inner = outer_root / "VCTK-Corpus-0.92.zip"
        safe_extract_zip(outer, outer_root)
        corpus_root = self.extracted_dir / "VCTK-Corpus"
        safe_extract_zip(inner, corpus_root)
        self.ensure_expected_files()
        inner.unlink(missing_ok=True)
        return self.extracted_dir

    def _genders(self) -> dict[str, str]:
        genders: dict[str, str] = {}
        path = self.extracted_dir / "VCTK-Corpus" / "speaker-info.txt"
        if not path.is_file():
            return genders
        for line in path.read_text(encoding="utf-8").splitlines()[1:]:
            fields = line.split()
            if len(fields) >= 3:
                speaker = fields[0]
                if not speaker.startswith("p"):
                    speaker = f"p{speaker}"
                gender = {"f": "f", "female": "f", "m": "m", "male": "m"}.get(fields[2].casefold())
                if gender is None:
                    raise ValueError(
                        f"{path}: unknown gender {fields[2]!r} for speaker {speaker!r}"
                    )
                genders[speaker] = gender
        return genders

    def inventory(self) -> list[InventoryItem]:
        self.ensure_expected_files()
        corpus = self.extracted_dir / "VCTK-Corpus"
        paths = list(corpus.glob("wav48_silence_trimmed/**/*.flac"))
        pairs = require_unique(((path.stem, path) for path in sorted(paths)), self.config.id)
        genders = self._genders()
        records = []
        for source_id, path in pairs:
            speaker = source_id.split("_", 1)[0]
            transcript_id = source_id[:-5] if source_id.endswith(("_mic1", "_mic2")) else source_id
            text = None
            if speaker != "p315":
                text = read_text(corpus / "txt" / speaker / f"{transcript_id}.txt")
            records.append(
                self.item(
                    source_id,
                    MediaKind.SPEECH,
                    path,
                    speaker_id=f"vctk_{speaker}",
                    text=text,
                    language="en",
                    gender=genders.get(speaker),
                )
            )
        return records
