# Copyright 2026 Cisco Systems, Inc. and its affiliates
# Apache-2.0

"""Export the canonical manifest to an ESPnet/Kaldi-style data directory."""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import tempfile
import uuid
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from itertools import groupby
from pathlib import Path
from typing import TextIO

from ..manifests import ManifestError
from ..models import ManifestItem, MediaKind, Split

_GENERATED_FILENAMES = {
    "noise.scp",
    "rirs.scp",
    "reference.scp",
    "spk1.scp",
    "spk2gender",
    "spk2utt",
    "text",
    "utt2category",
    "utt2fs",
    "utt2spk",
    "wav.scp",
}
_EVALUATION_FILENAMES = {"reference.scp", "wav.scp"}
_EVALUATION_DIRECTORY = re.compile(r"open_testset_track[1-9][0-9]*_[A-Za-z0-9][A-Za-z0-9_-]*")
_TRACK = re.compile(r"track_([1-9][0-9]*)")
_CONDITION = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_GENDERS = {"f": "f", "female": "f", "m": "m", "male": "m"}


def export_kaldi(
    manifest: Path,
    output: Path,
    *,
    workspace: Path | None = None,
    relative_audio_paths: bool = False,
) -> dict[str, int]:
    """Atomically stream one ordered manifest into a Kaldi data directory."""

    manifest = manifest.expanduser().resolve()
    output = output.expanduser().resolve()
    data_root = (
        None
        if relative_audio_paths
        else (
            workspace.expanduser().resolve()
            if workspace is not None
            else _infer_workspace(manifest)
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _check_destination(output)

    work_parent = _work_parent(output)
    managed_work = work_parent != output.parent
    _prepare_work_parent(work_parent, managed=managed_work)
    if managed_work:
        _cleanup_managed_leftovers(work_parent, output.name)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".lrac-kaldi-{output.name}.staging-",
            dir=work_parent,
        )
    )
    backup: Path | None = None
    try:
        counts = _write_generation(staging, _read_ordered_manifest(manifest), data_root)
        if output.exists():
            backup = work_parent / f".lrac-kaldi-{output.name}.previous-{uuid.uuid4().hex}"
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
    if managed_work and not any(work_parent.iterdir()):
        work_parent.rmdir()
    return counts


def _write_generation(
    output: Path,
    records: Iterator[ManifestItem],
    data_root: Path | None,
) -> dict[str, int]:
    streams: dict[str, TextIO] = {}
    counts: dict[str, int] = {}
    previous_keys: dict[str, str] = {}
    database_path = output / ".kaldi-index.sqlite3"
    saw_record = False
    saw_standard_record = False

    with ExitStack() as stack, sqlite3.connect(database_path) as database:
        database.executescript(
            """
            PRAGMA journal_mode = OFF;
            PRAGMA synchronous = OFF;
            PRAGMA temp_store = FILE;
            CREATE TABLE memberships (
                speaker TEXT NOT NULL,
                utterance TEXT PRIMARY KEY
            );
            CREATE INDEX memberships_by_speaker
                ON memberships (speaker, utterance);
            CREATE TABLE speakers (
                speaker TEXT PRIMARY KEY,
                gender TEXT,
                gender_valid INTEGER NOT NULL
            );
            CREATE TABLE evaluation_pairs (
                directory TEXT NOT NULL,
                pair_id TEXT NOT NULL,
                role TEXT NOT NULL,
                PRIMARY KEY (directory, pair_id, role)
            );
            """
        )

        def stream(relative: str) -> TextIO:
            opened = streams.get(relative)
            if opened is not None:
                return opened
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            opened = stack.enter_context(path.open("w", encoding="utf-8", newline="\n"))
            streams[relative] = opened
            counts[relative] = 0
            return opened

        def write(relative: str, key: str, value: str) -> None:
            previous = previous_keys.get(relative)
            if previous is not None and key <= previous:
                raise ValueError(f"{relative}: keys are not strictly ordered at {key!r}")
            stream(relative).write(f"{key} {value}\n")
            counts[relative] += 1
            previous_keys[relative] = key

        def ensure_standard_files() -> None:
            for filename in ("wav.scp", "utt2fs", "utt2category"):
                stream(filename)

        for record in records:
            saw_record = True
            audio_path = _audio_path(record, data_root)
            if _write_evaluation_pair(record, audio_path, write, database):
                continue
            if not saw_standard_record:
                ensure_standard_files()
                saw_standard_record = True

            item_id = record.id
            if record.media_kind is MediaKind.NOISE:
                write("noise.scp", item_id, audio_path)
                continue
            if record.media_kind is MediaKind.RIR:
                write("rirs.scp", item_id, audio_path)
                continue

            speaker_id = record.speaker_id or f"unknown:{item_id}"
            _reject_newline(speaker_id, field="speaker ID", item_id=item_id)
            utterance_text = (
                (record.text or "<not-available>").replace("\r", " ").replace("\n", " ")
            )
            write("wav.scp", item_id, audio_path)
            if record.split is Split.EVALUATION:
                write("reference.scp", item_id, audio_path)
            write("spk1.scp", item_id, audio_path)
            write("utt2spk", item_id, speaker_id)
            write("text", item_id, utterance_text)
            write("utt2fs", item_id, str(record.sample_rate_hz))
            write(
                "utt2category",
                item_id,
                f"{record.channels}ch_{record.sample_rate_hz}Hz",
            )
            database.execute(
                "INSERT INTO memberships (speaker, utterance) VALUES (?, ?)",
                (speaker_id, item_id),
            )
            gender = _GENDERS.get(record.gender.casefold()) if record.gender else None
            gender_valid = int(record.gender is None or gender is not None)
            database.execute(
                """
                INSERT INTO speakers (speaker, gender, gender_valid)
                VALUES (?, ?, ?)
                ON CONFLICT (speaker) DO UPDATE SET
                    gender = CASE
                        WHEN speakers.gender IS NULL THEN excluded.gender
                        ELSE speakers.gender
                    END,
                    gender_valid = (
                        speakers.gender_valid
                        AND excluded.gender_valid
                        AND (
                            speakers.gender IS NULL
                            OR excluded.gender IS NULL
                            OR speakers.gender = excluded.gender
                        )
                    )
                """,
                (speaker_id, gender, gender_valid),
            )

        if not saw_record:
            ensure_standard_files()
        _require_complete_evaluation_pairs(database)
        if counts.get("utt2spk"):
            spk2utt = stream("spk2utt")
            memberships = database.execute(
                "SELECT speaker, utterance FROM memberships ORDER BY speaker, utterance"
            )
            for speaker, rows in groupby(memberships, key=lambda row: str(row[0])):
                spk2utt.write(speaker)
                for _speaker, utterance in rows:
                    spk2utt.write(f" {utterance}")
                spk2utt.write("\n")
                counts["spk2utt"] += 1

            incomplete_gender = database.execute(
                "SELECT 1 FROM speakers WHERE gender IS NULL OR NOT gender_valid LIMIT 1"
            ).fetchone()
            if incomplete_gender is None:
                for speaker, gender in database.execute(
                    "SELECT speaker, gender FROM speakers ORDER BY speaker"
                ):
                    write("spk2gender", str(speaker), str(gender))

    database_path.unlink()
    return counts


def _write_evaluation_pair(
    record: ManifestItem,
    audio_path: str,
    write: Callable[[str, str, str], None],
    database: sqlite3.Connection,
) -> bool:
    role = record.metadata.get("role")
    pair_id = record.metadata.get("pair_id")
    track = record.metadata.get("track")
    condition = record.metadata.get("condition")
    fields = (role, pair_id, track, condition)
    if all(value is None for value in fields):
        return False
    if record.media_kind is not MediaKind.SPEECH:
        raise ValueError(f"{record.id}: evaluation pair must contain speech")
    if role not in {"input", "reference"}:
        raise ValueError(f"{record.id}: evaluation role must be 'input' or 'reference'")
    if (
        not isinstance(pair_id, str)
        or not pair_id
        or any(character.isspace() for character in pair_id)
    ):
        raise ValueError(f"{record.id}: evaluation pair_id must not contain whitespace")
    if not isinstance(track, str) or (track_match := _TRACK.fullmatch(track)) is None:
        raise ValueError(f"{record.id}: unsafe evaluation track {track!r}")
    if not isinstance(condition, str) or _CONDITION.fullmatch(condition) is None:
        raise ValueError(f"{record.id}: unsafe evaluation condition {condition!r}")

    directory = f"open_testset_track{track_match.group(1)}_{condition}"
    filename = "wav.scp" if role == "input" else "reference.scp"
    write(f"{directory}/{filename}", pair_id, audio_path)
    try:
        database.execute(
            "INSERT INTO evaluation_pairs (directory, pair_id, role) VALUES (?, ?, ?)",
            (directory, pair_id, role),
        )
    except sqlite3.IntegrityError as error:
        raise ValueError(f"{record.id}: duplicate evaluation {role} for {pair_id!r}") from error
    return True


def _require_complete_evaluation_pairs(database: sqlite3.Connection) -> None:
    incomplete = database.execute(
        """
        SELECT directory, pair_id
        FROM evaluation_pairs
        GROUP BY directory, pair_id
        HAVING COUNT(*) != 2
        LIMIT 1
        """
    ).fetchone()
    if incomplete is not None:
        directory, pair_id = incomplete
        raise ValueError(f"{directory}: incomplete evaluation pair {pair_id!r}")


def _read_ordered_manifest(path: Path) -> Iterator[ManifestItem]:
    previous_id: str | None = None
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            context = f"{path}:{line_number}"
            if not line.strip():
                raise ManifestError(f"{context}: blank JSONL line")
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("expected a JSON object")
                record = ManifestItem.model_validate(value)
            except (json.JSONDecodeError, ValueError) as error:
                raise ManifestError(f"{context}: invalid manifest record: {error}") from error
            if previous_id is not None and record.id <= previous_id:
                raise ManifestError(f"{context}: manifest IDs are not strictly ordered")
            previous_id = record.id
            yield record


def _audio_path(record: ManifestItem, data_root: Path | None) -> str:
    if data_root is None:
        audio_path = record.audio_path.as_posix()
    else:
        resolved = (data_root / Path(record.audio_path)).resolve()
        try:
            resolved.relative_to(data_root)
        except ValueError as error:
            raise ValueError(
                f"manifest item {record.id!r} audio path escapes the data root"
            ) from error
        audio_path = str(resolved)
    _reject_newline(audio_path, field="audio path", item_id=record.id)
    return audio_path


def _infer_workspace(manifest: Path) -> Path:
    for parent in manifest.parents:
        if parent.name == "manifests":
            return parent.parent
    raise ValueError(
        "cannot infer workspace from a manifest outside a 'manifests' directory; "
        "pass workspace explicitly"
    )


def _check_destination(output: Path) -> None:
    if output.exists() and (not output.is_dir() or output.is_symlink()):
        raise ValueError(f"Kaldi export destination is not a directory: {output}")
    if not output.exists():
        return

    unexpected: list[str] = []
    for path in output.iterdir():
        if path.is_file() and not path.is_symlink() and path.name in _GENERATED_FILENAMES:
            continue
        if path.is_dir() and not path.is_symlink() and _EVALUATION_DIRECTORY.fullmatch(path.name):
            for child in path.iterdir():
                if (
                    not child.is_file()
                    or child.is_symlink()
                    or child.name not in _EVALUATION_FILENAMES
                ):
                    unexpected.append(child.relative_to(output).as_posix())
            continue
        unexpected.append(path.relative_to(output).as_posix())
    if unexpected:
        raise ValueError(
            f"Kaldi export destination contains files not owned by lrac-data: {sorted(unexpected)}"
        )


def _work_parent(output: Path) -> Path:
    """Isolate resumable release generation from published files."""

    release_staging = _managed_release_staging(output)
    if release_staging is not None:
        return release_staging / ".kaldi-work"
    return output.parent


def _prepare_work_parent(path: Path, *, managed: bool) -> None:
    if path.exists() and (not path.is_dir() or path.is_symlink()):
        raise ValueError(f"Kaldi work path is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if managed:
        for child in path.iterdir():
            if child.is_symlink():
                raise ValueError(f"Kaldi work path contains a symlink: {child}")


def _cleanup_managed_leftovers(work_parent: Path, output_name: str) -> None:
    prefixes = (
        f".lrac-kaldi-{output_name}.staging-",
        f".lrac-kaldi-{output_name}.previous-",
    )
    for path in work_parent.iterdir():
        if not path.name.startswith(prefixes):
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def _managed_release_staging(output: Path) -> Path | None:
    release_staging = output.parent.parent
    if output.parent.name == "kaldi" and (release_staging / "manifests").is_dir():
        return release_staging
    return None


def _reject_newline(value: str, *, field: str, item_id: str) -> None:
    if "\n" in value or "\r" in value:
        raise ValueError(f"manifest item {item_id!r} has a newline in {field}")
