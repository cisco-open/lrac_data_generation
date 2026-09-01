# Copyright 2026 Cisco Systems, Inc. and its affiliates
# Apache-2.0

"""OpenSLR 93 (AISHELL-3) speech adapter."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from lrac_data.models import InventoryItem, MediaKind

from .base import DatasetAdapter
from .io import safe_extract_tar

_AISHELL_SPEAKER = re.compile(r"SSB[0-9]{4}")


@dataclass(frozen=True, slots=True)
class _ContentRecord:
    text: str
    pinyin: str
    annotated_text: str


@dataclass(frozen=True, slots=True)
class _SpeakerRecord:
    age_group: str
    gender: str
    accent: str


class OpenSLR93Adapter(DatasetAdapter):
    """Inventory the complete AISHELL-3 WAV tree before edition selection."""

    def fetch(self) -> Path:
        (archive,) = self.download_remote_sources("corpus")
        safe_extract_tar(archive, self.extracted_dir)
        return self.extracted_dir

    def inventory(self) -> list[InventoryItem]:
        root = self.extracted_dir.resolve()
        content = _read_all_content(root)
        speakers = _read_speaker_info(root)
        wavs: list[tuple[str, Path, Path]] = []
        for candidate in root.rglob("*.wav"):
            if not candidate.is_file():
                continue
            path = candidate.resolve()
            try:
                relative = path.relative_to(root)
            except ValueError as error:
                raise ValueError(
                    f"AISHELL-3 audio escapes the extracted directory: {path}"
                ) from error
            wavs.append((relative.as_posix(), path, relative))

        records: list[InventoryItem] = []
        matched_content: set[str] = set()
        for _, path, relative in sorted(wavs):
            source_id = relative.with_suffix("").as_posix()
            speaker = _speaker_from_path(relative)
            if speaker is None:
                raise ValueError(
                    f"AISHELL-3 WAV does not match the speaker layout: {relative.as_posix()}"
                )
            try:
                transcript = content[path.name]
            except KeyError as error:
                raise ValueError(
                    f"AISHELL-3 WAV has no content metadata: {relative.as_posix()}"
                ) from error
            try:
                speaker_record = speakers[speaker]
            except KeyError as error:
                raise ValueError(
                    f"AISHELL-3 WAV references unknown speaker {speaker!r}: {relative.as_posix()}"
                ) from error
            matched_content.add(path.name)
            records.append(
                self.item(
                    source_id,
                    MediaKind.SPEECH,
                    path,
                    speaker_id=f"{self.config.id}_{speaker}",
                    text=transcript.text,
                    language="cmn",
                    gender=speaker_record.gender,
                    metadata={
                        "pinyin": transcript.pinyin,
                        "annotated_text": transcript.annotated_text,
                        "age_group": speaker_record.age_group,
                        "accent": speaker_record.accent,
                    },
                )
            )
        unmatched = sorted(set(content) - matched_content)
        if unmatched:
            raise ValueError(f"AISHELL-3 content metadata has no WAV for {unmatched[0]!r}")
        return records


def _speaker_from_path(relative_path: Path) -> str | None:
    """Return the speaker only when directory and utterance IDs corroborate it."""

    speaker = relative_path.parent.name
    if _AISHELL_SPEAKER.fullmatch(speaker) and relative_path.stem.startswith(speaker):
        return speaker
    return None


def _read_all_content(root: Path) -> dict[str, _ContentRecord]:
    paths = sorted(root.rglob("content.txt"), key=lambda path: path.relative_to(root).as_posix())
    if not paths:
        raise FileNotFoundError(f"AISHELL-3 has no content.txt under {root}")
    records: dict[str, _ContentRecord] = {}
    origins: dict[str, Path] = {}
    for path in paths:
        for filename, record in _read_content(path).items():
            previous = origins.get(filename)
            if previous is not None:
                raise ValueError(
                    f"Duplicate AISHELL-3 content filename {filename!r} in {previous} and {path}"
                )
            origins[filename] = path
            records[filename] = record
    return records


def _read_content(path: Path) -> dict[str, _ContentRecord]:
    records: dict[str, _ContentRecord] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) != 2:
                raise ValueError(f"{path}:{line_number}: expected two tab-separated content fields")
            filename = fields[0].strip()
            annotated_text = fields[1].strip()
            candidate = PurePosixPath(filename)
            if (
                not filename
                or candidate.name != filename
                or "\\" in filename
                or candidate.suffix != ".wav"
            ):
                raise ValueError(
                    f"{path}:{line_number}: invalid AISHELL-3 content filename {filename!r}"
                )
            tokens = annotated_text.split()
            if not tokens or len(tokens) % 2:
                raise ValueError(
                    f"{path}:{line_number}: content must alternate written and pinyin tokens"
                )
            if filename in records:
                raise ValueError(
                    f"{path}:{line_number}: duplicate AISHELL-3 content filename {filename!r}"
                )
            records[filename] = _ContentRecord(
                text="".join(tokens[0::2]),
                pinyin=" ".join(tokens[1::2]),
                annotated_text=annotated_text,
            )
    if not records:
        raise ValueError(f"AISHELL-3 content file is empty: {path}")
    return records


def _read_speaker_info(root: Path) -> dict[str, _SpeakerRecord]:
    paths = sorted(root.rglob("spk-info.txt"), key=lambda path: path.relative_to(root).as_posix())
    if len(paths) != 1:
        raise FileNotFoundError(
            f"AISHELL-3 expected one spk-info.txt under {root}, found {len(paths)}"
        )
    path = paths[0]
    records: dict[str, _SpeakerRecord] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = [field.strip() for field in line.split("\t")]
            if len(fields) != 4 or any(not field for field in fields):
                raise ValueError(f"{path}:{line_number}: expected four non-empty speaker fields")
            speaker_id, age_group, raw_gender, accent = fields
            if _AISHELL_SPEAKER.fullmatch(speaker_id) is None:
                raise ValueError(
                    f"{path}:{line_number}: invalid AISHELL-3 speaker ID {speaker_id!r}"
                )
            try:
                gender = {"female": "f", "male": "m"}[raw_gender]
            except KeyError as error:
                raise ValueError(
                    f"{path}:{line_number}: unknown AISHELL-3 gender {raw_gender!r}"
                ) from error
            if speaker_id in records:
                raise ValueError(
                    f"{path}:{line_number}: duplicate AISHELL-3 speaker {speaker_id!r}"
                )
            records[speaker_id] = _SpeakerRecord(
                age_group=age_group,
                gender=gender,
                accent=accent,
            )
    if not records:
        raise ValueError(f"AISHELL-3 speaker metadata is empty: {path}")
    return records
