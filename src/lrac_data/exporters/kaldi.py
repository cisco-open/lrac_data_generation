"""Export the canonical manifest to an ESPnet/Kaldi-style data directory."""

from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..state import atomic_write_text

_GENERATED_FILENAMES = {
    "spk1.scp",
    "spk2gender",
    "spk2utt",
    "text",
    "utt2category",
    "utt2fs",
    "utt2spk",
    "wav.scp",
}


def export_kaldi(
    manifest: Path,
    output: Path,
    *,
    workspace: Path | None = None,
) -> dict[str, int]:
    manifest = manifest.expanduser().resolve()
    records = sorted(_read_records(manifest), key=lambda record: str(record["id"]))
    _validate_unique_ids(records)
    workspace = (
        workspace.expanduser().resolve() if workspace is not None else _infer_workspace(manifest)
    )

    wav_scp: list[str] = []
    spk1_scp: list[str] = []
    utt2spk: list[str] = []
    text: list[str] = []
    utt2fs: list[str] = []
    utt2category: list[str] = []
    speakers: dict[str, list[str]] = defaultdict(list)
    genders: dict[str, str] = {}

    for record in records:
        item_id = str(record["id"])
        legacy_audio = record.get("audio") or {}
        audio_value = record.get("audio_path") or legacy_audio.get("path")
        if not audio_value:
            raise ValueError(f"manifest item {item_id!r} has no audio path")
        audio_path = Path(str(audio_value))
        if not audio_path.is_absolute():
            audio_path = (workspace / audio_path).resolve()
        _reject_newline(str(audio_path), field="audio path", item_id=item_id)
        wav_scp.append(f"{item_id} {audio_path}")

        sample_rate = int(record.get("sample_rate_hz") or legacy_audio.get("sample_rate_hz", 0))
        channels = int(record.get("channels") or legacy_audio.get("channels", 0))
        if sample_rate <= 0 or channels <= 0:
            raise ValueError(f"manifest item {item_id!r} has invalid audio metadata")
        utt2fs.append(f"{item_id} {sample_rate}")
        utt2category.append(f"{item_id} {channels}ch_{sample_rate}Hz")

        kind = _enum_value(record.get("media_kind") or record.get("kind"))
        if kind != "speech":
            continue
        speaker_id = str(record.get("speaker_id") or f"unknown:{item_id}")
        _reject_newline(speaker_id, field="speaker ID", item_id=item_id)
        utterance_text = str(record.get("text") or "<not-available>").replace("\n", " ")
        utt2spk.append(f"{item_id} {speaker_id}")
        text.append(f"{item_id} {utterance_text}")
        spk1_scp.append(f"{item_id} {audio_path}")
        speakers[speaker_id].append(item_id)
        gender = record.get("gender")
        if gender:
            value = str(gender).lower()[0]
            if speaker_id in genders and genders[speaker_id] != value:
                raise ValueError(f"speaker {speaker_id!r} has conflicting gender metadata")
            genders[speaker_id] = value

    files: dict[str, Iterable[str]] = {
        "wav.scp": wav_scp,
        "utt2fs": utt2fs,
        "utt2category": utt2category,
    }
    if utt2spk:
        files.update(
            {
                "spk1.scp": spk1_scp,
                "utt2spk": utt2spk,
                "spk2utt": (
                    f"{speaker} {' '.join(sorted(utterances))}"
                    for speaker, utterances in sorted(speakers.items())
                ),
                "text": text,
                "spk2gender": (
                    f"{speaker} {gender}" for speaker, gender in sorted(genders.items())
                ),
            }
        )
    contents = {
        filename: "".join(f"{line}\n" for line in lines) for filename, lines in files.items()
    }
    _publish_generation(output.expanduser().resolve(), contents)
    return {filename: value.count("\n") for filename, value in contents.items()}


def _infer_workspace(manifest: Path) -> Path:
    for parent in manifest.parents:
        if parent.name == "manifests":
            return parent.parent
    raise ValueError(
        "cannot infer workspace from a manifest outside a 'manifests' directory; "
        "pass workspace explicitly"
    )


def _publish_generation(output: Path, contents: dict[str, str]) -> None:
    """Replace an export directory as one complete generation."""

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and (not output.is_dir() or output.is_symlink()):
        raise ValueError(f"Kaldi export destination is not a directory: {output}")
    if output.exists():
        unexpected = sorted(
            path.name
            for path in output.iterdir()
            if path.name not in _GENERATED_FILENAMES or not path.is_file() or path.is_symlink()
        )
        if unexpected:
            raise ValueError(
                f"Kaldi export destination contains files not owned by lrac-data: {unexpected}"
            )

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    backup: Path | None = None
    try:
        for filename, text in contents.items():
            atomic_write_text(staging / filename, text)

        if output.exists():
            backup = output.with_name(f".{output.name}.previous-{uuid.uuid4().hex}")
            output.replace(backup)
        try:
            staging.replace(output)
        except BaseException:
            if backup is not None and backup.exists() and not output.exists():
                backup.replace(output)
                backup = None
            raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    if backup is not None:
        shutil.rmtree(backup)


def _read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL line")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: manifest record must be an object")
            records.append(value)
    return records


def _validate_unique_ids(records: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for record in records:
        item_id = str(record.get("id", ""))
        if not item_id:
            raise ValueError("manifest item is missing an ID")
        if item_id in seen:
            raise ValueError(f"duplicate manifest ID: {item_id}")
        seen.add(item_id)


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _reject_newline(value: str, *, field: str, item_id: str) -> None:
    if "\n" in value or "\r" in value:
        raise ValueError(f"manifest item {item_id!r} has a newline in {field}")
