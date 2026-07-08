"""Validation of published LRAC manifests and their materialized audio."""

from __future__ import annotations

import json
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .audio import AudioMetadata, probe
from .models import ManifestItem
from .state import sha256_file

_VALIDATION_BATCH_SIZE = 256


@dataclass(frozen=True)
class ValidationReport:
    manifests: int
    records: int
    audio_files: int
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class PublishedRunReport:
    manifests: tuple[Path, ...]
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class _AudioCheck:
    item: ManifestItem
    path: Path


def validate_published_run(group: Path, *, workspace: Path) -> PublishedRunReport:
    """Verify that one published directory matches its run metadata exactly."""

    workspace = workspace.expanduser().resolve()
    group = group.expanduser().resolve()
    actual = tuple(sorted(path.resolve() for path in group.glob("*.jsonl") if path.is_file()))
    errors: list[str] = []
    metadata_path = group / "run.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return PublishedRunReport(actual, (f"{group}: run.json is missing",))
    except (OSError, json.JSONDecodeError) as error:
        return PublishedRunReport(actual, (f"{metadata_path}: {error}",))
    if not isinstance(metadata, Mapping):
        return PublishedRunReport(actual, (f"{metadata_path}: expected a JSON object",))

    declarations = metadata.get("manifests")
    if not isinstance(declarations, Mapping) or not declarations:
        return PublishedRunReport(
            actual,
            (f"{metadata_path}: 'manifests' must be a non-empty object",),
        )

    declared: dict[Path, str] = {}
    for split, declaration in sorted(declarations.items(), key=lambda item: str(item[0])):
        if not isinstance(split, str) or not split:
            errors.append(f"{metadata_path}: manifest split names must be non-empty strings")
            continue
        if not isinstance(declaration, Mapping):
            errors.append(f"{metadata_path}: manifest {split!r} must be an object")
            continue
        path_value = declaration.get("path")
        digest = declaration.get("sha256")
        if not isinstance(path_value, str) or not path_value:
            errors.append(f"{metadata_path}: manifest {split!r} has no path")
            continue
        relative = Path(path_value)
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(f"{metadata_path}: manifest {split!r} has an unsafe path")
            continue
        candidate = (workspace / relative).resolve()
        if candidate.parent != group:
            errors.append(f"{metadata_path}: manifest {split!r} is outside its published directory")
            continue
        if candidate.name != f"{split}.jsonl":
            errors.append(f"{metadata_path}: manifest {split!r} must reference {split}.jsonl")
        if candidate in declared:
            errors.append(
                f"{metadata_path}: manifest path is declared more than once: {candidate.name}"
            )
            continue
        declared[candidate] = split
        if not isinstance(digest, str) or not digest:
            errors.append(f"{metadata_path}: manifest {split!r} has no SHA-256 digest")
            continue
        if not candidate.is_file():
            errors.append(f"{metadata_path}: referenced manifest is missing: {candidate}")
            continue
        try:
            actual_digest = sha256_file(candidate)
        except OSError as error:
            errors.append(f"{candidate}: {error}")
            continue
        if actual_digest != digest:
            errors.append(f"{candidate}: digest does not match run.json")

    actual_set = set(actual)
    declared_set = set(declared)
    for path in sorted(actual_set - declared_set):
        errors.append(f"{path}: manifest is not declared by run.json")
    for path in sorted(declared_set - actual_set):
        if path.is_file():
            errors.append(f"{path}: declared manifest is not part of the published set")
    return PublishedRunReport(actual, tuple(errors))


def validate_manifests(
    manifests: list[Path],
    *,
    workspace: Path,
    verify_checksums: bool = True,
    known_audio: Mapping[Path, AudioMetadata] | None = None,
    workers: int = 1,
) -> ValidationReport:
    if workers < 1:
        raise ValueError("validation workers must be positive")

    errors: list[str] = []
    seen: dict[str, Path] = {}
    records = 0
    audio_files: set[Path] = set()
    executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None

    try:
        for manifest in sorted(manifests):
            previous_id: str | None = None
            events: list[str | _AudioCheck] = []
            pending_checks = 0
            try:
                source = manifest.open(encoding="utf-8")
            except OSError as error:
                errors.append(f"{manifest}: {error}")
                continue
            with source:
                for line_number, line in enumerate(source, start=1):
                    if not line.strip():
                        events.append(f"{manifest}:{line_number}: blank JSONL line")
                        continue
                    try:
                        item = ManifestItem.model_validate(json.loads(line))
                    except (ValueError, json.JSONDecodeError) as error:
                        events.append(f"{manifest}:{line_number}: {error}")
                        continue
                    records += 1
                    if previous_id is not None and item.id <= previous_id:
                        events.append(
                            f"{manifest}:{line_number}: records are not strictly sorted by ID"
                        )
                    previous_id = item.id
                    if item.id in seen:
                        events.append(f"{item.id}: appears in both {seen[item.id]} and {manifest}")
                    else:
                        seen[item.id] = manifest

                    audio_path = workspace / Path(item.audio_path.as_posix())
                    audio_files.add(audio_path)
                    events.append(_AudioCheck(item, audio_path))
                    pending_checks += 1
                    if pending_checks >= _VALIDATION_BATCH_SIZE:
                        _flush_audio_checks(
                            events,
                            errors,
                            verify_checksums=verify_checksums,
                            known_audio=known_audio,
                            executor=executor,
                        )
                        events = []
                        pending_checks = 0
            _flush_audio_checks(
                events,
                errors,
                verify_checksums=verify_checksums,
                known_audio=known_audio,
                executor=executor,
            )
    finally:
        if executor is not None:
            executor.shutdown()

    return ValidationReport(
        manifests=len(manifests),
        records=records,
        audio_files=len(audio_files),
        errors=tuple(errors),
    )


def _flush_audio_checks(
    events: list[str | _AudioCheck],
    errors: list[str],
    *,
    verify_checksums: bool,
    known_audio: Mapping[Path, AudioMetadata] | None,
    executor: ThreadPoolExecutor | None,
) -> None:
    checks = [event for event in events if isinstance(event, _AudioCheck)]
    if executor is None:
        results = [
            _inspect_audio(
                check,
                verify_checksums=verify_checksums,
                known_audio=known_audio,
            )
            for check in checks
        ]
    else:
        futures = [
            executor.submit(
                _inspect_audio,
                check,
                verify_checksums=verify_checksums,
                known_audio=known_audio,
            )
            for check in checks
        ]
        results = [future.result() for future in futures]

    inspected = iter(results)
    for event in events:
        if isinstance(event, str):
            errors.append(event)
        else:
            errors.extend(next(inspected))


def _inspect_audio(
    check: _AudioCheck,
    *,
    verify_checksums: bool,
    known_audio: Mapping[Path, AudioMetadata] | None,
) -> tuple[str, ...]:
    item = check.item
    audio_path = check.path
    if not audio_path.is_file():
        return (f"{item.id}: audio is missing: {audio_path}",)

    metadata = known_audio.get(audio_path) if known_audio is not None else None
    if (
        metadata is None
        or not metadata.matches_stat(audio_path)
        or (verify_checksums and (not metadata.sha256 or not metadata.checksum_fresh))
    ):
        try:
            metadata = probe(audio_path, include_checksum=verify_checksums)
        except (OSError, RuntimeError, ValueError) as error:
            return (f"{item.id}: cannot inspect {audio_path}: {error}",)

    errors: list[str] = []
    if metadata.sample_rate_hz != item.sample_rate_hz:
        errors.append(f"{item.id}: sample-rate mismatch")
    if metadata.channels != item.channels:
        errors.append(f"{item.id}: channel-count mismatch")
    if metadata.num_frames != item.frame_count:
        errors.append(f"{item.id}: frame-count mismatch")
    if metadata.subtype != "PCM_16":
        errors.append(f"{item.id}: expected PCM_16 WAV audio")
    if verify_checksums and metadata.sha256 != item.checksum:
        errors.append(f"{item.id}: checksum mismatch")
    return tuple(errors)


def find_published_manifests(workspace: Path) -> list[Path]:
    root = workspace / "manifests"
    if not root.exists():
        return []
    return [
        path
        for path in sorted(root.rglob("*.jsonl"))
        if "inventory" not in path.parts and not path.name.startswith("inventory")
    ]
