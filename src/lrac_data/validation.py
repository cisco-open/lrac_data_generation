"""Validation of published LRAC manifests and their materialized audio."""

from __future__ import annotations

import json
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from .audio import AudioMetadata, probe
from .config import (
    ConfigError,
    LoadedEdition,
    load_recorded_edition_config,
    portable_config_path,
    portable_config_payload,
)
from .models import AudioFormat, ManifestItem, MediaKind, PublishedRunMetadata, Split
from .state import fingerprint, sha256_file

_VALIDATION_BATCH_SIZE = 256
_BASE_PARTITIONS = frozenset({"train", "validation", "evaluation"})
_PARTITION_SPLITS = {
    "train": Split.TRAIN,
    "validation": Split.VALIDATION,
    "evaluation": Split.EVALUATION,
    "open-evaluation": Split.EVALUATION,
}


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

    metadata: PublishedRunMetadata | None = None
    config: LoadedEdition | None = None

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class _AudioCheck:
    item: ManifestItem
    path: Path


def validate_published_run(
    group: Path,
    *,
    workspace: Path,
    config: LoadedEdition | None = None,
    repo_root: Path | None = None,
    edition_config: Path | None = None,
) -> PublishedRunReport:
    """Verify that one published directory matches its typed run contract exactly."""

    workspace = workspace.expanduser().resolve()
    group = group.expanduser().resolve()
    manifests_root = (workspace / "manifests").resolve()
    actual = tuple(sorted(path.resolve() for path in group.glob("*.jsonl") if path.is_file()))
    errors: list[str] = []
    metadata_path = group / "run.json"
    try:
        metadata = PublishedRunMetadata.model_validate_json(
            metadata_path.read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        return PublishedRunReport(actual, (f"{group}: run.json is missing",))
    except OSError as error:
        return PublishedRunReport(actual, (f"{metadata_path}: {error}",))
    except ValidationError as error:
        return PublishedRunReport(
            actual,
            (f"{metadata_path}: run metadata does not match schema: {error}",),
        )

    resolved_from_metadata = False
    if config is None:
        try:
            config = load_recorded_edition_config(
                metadata.config_path,
                repo_root=repo_root,
                selection=metadata.selection,
                edition_config=edition_config,
            )
        except (ConfigError, ValueError) as error:
            errors.append(f"{metadata_path}: cannot resolve recorded configuration: {error}")
        else:
            resolved_from_metadata = True

    location: tuple[str, str] | None = None
    try:
        relative_group = group.relative_to(manifests_root)
    except ValueError:
        errors.append(f"{group}: published directory is outside {manifests_root}")
    else:
        if len(relative_group.parts) != 2:
            errors.append(f"{group}: expected publication location manifests/<edition>/<selection>")
        else:
            location = (relative_group.parts[0], relative_group.parts[1])

    if location is not None:
        location_edition, location_selection = location
        if metadata.edition != location_edition:
            errors.append(
                f"{metadata_path}: edition {metadata.edition!r} does not match "
                f"publication directory {location_edition!r}"
            )
        if metadata.selection.value != location_selection:
            errors.append(
                f"{metadata_path}: selection {metadata.selection.value!r} does not match "
                f"publication directory {location_selection!r}"
            )

    if config is not None:
        if metadata.edition != config.config.edition:
            errors.append(
                f"{metadata_path}: edition {metadata.edition!r} does not match "
                f"resolved configuration {config.config.edition!r}"
            )
        expected_config_path = portable_config_path(config)
        if not resolved_from_metadata and metadata.config_path != expected_config_path:
            errors.append(
                f"{metadata_path}: config path {metadata.config_path!r} does not match "
                f"{expected_config_path!r}"
            )
        expected_config_fingerprint = fingerprint(portable_config_payload(config))
        if metadata.config_fingerprint != expected_config_fingerprint:
            errors.append(f"{metadata_path}: config fingerprint does not match resolved edition")

    required = set(_BASE_PARTITIONS)
    if config is not None:
        if config.config.public_evaluation is not None:
            required.add("open-evaluation")
    elif metadata.counts.open_evaluation is not None:
        required.add("open-evaluation")
    declared_names = set(metadata.manifests)
    for partition in sorted(required - declared_names):
        errors.append(f"{metadata_path}: required manifest {partition!r} is not declared")
    for partition in sorted(declared_names - required):
        errors.append(f"{metadata_path}: unexpected manifest {partition!r} is declared")

    declared: dict[Path, str] = {}
    for partition, declaration in sorted(metadata.manifests.items()):
        relative = Path(declaration.path.as_posix())
        candidate = (workspace / relative).resolve()
        if candidate.parent != group:
            errors.append(
                f"{metadata_path}: manifest {partition!r} is outside its published directory"
            )
            continue
        if candidate.name != f"{partition}.jsonl":
            errors.append(
                f"{metadata_path}: manifest {partition!r} must reference {partition}.jsonl"
            )
        if candidate in declared:
            errors.append(
                f"{metadata_path}: manifest path is declared more than once: {candidate.name}"
            )
            continue
        declared[candidate] = partition
        if not candidate.is_file():
            errors.append(f"{metadata_path}: referenced manifest is missing: {candidate}")
            continue
        try:
            actual_digest = sha256_file(candidate)
        except OSError as error:
            errors.append(f"{candidate}: {error}")
            continue
        if actual_digest != declaration.sha256:
            errors.append(f"{candidate}: digest does not match run.json")

    actual_set = set(actual)
    declared_set = set(declared)
    for path in sorted(actual_set - declared_set):
        errors.append(f"{path}: manifest is not declared by run.json")
    for path in sorted(declared_set - actual_set):
        if path.is_file():
            errors.append(f"{path}: declared manifest is not part of the published set")
    return PublishedRunReport(actual, tuple(errors), metadata, config)


def validate_manifests(
    manifests: list[Path],
    *,
    workspace: Path,
    verify_checksums: bool = True,
    known_audio: Mapping[Path, AudioMetadata] | None = None,
    expected_counts: Mapping[str, int] | None = None,
    target_audio: AudioFormat | None = None,
    workers: int = 1,
) -> ValidationReport:
    if workers < 1:
        raise ValueError("validation workers must be positive")

    errors: list[str] = []
    seen: dict[str, Path] = {}
    records = 0
    audio_files: set[Path] = set()
    manifest_counts: dict[str, int] = {}
    speakers: dict[str, set[tuple[str, str]]] = {"train": set(), "validation": set()}
    executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None

    try:
        for manifest in sorted(manifests):
            partition = manifest.stem
            expected_split = _PARTITION_SPLITS.get(partition)
            manifest_records = 0
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
                    manifest_records += 1
                    if expected_split is not None and item.split is not expected_split:
                        events.append(
                            f"{manifest}:{line_number}: split {item.split.value!r} does not match "
                            f"manifest partition {partition!r}"
                        )
                    if target_audio is not None:
                        events.extend(_audio_declaration_errors(item, target_audio))
                    if (
                        partition in speakers
                        and item.media_kind is MediaKind.SPEECH
                        and item.speaker_id is not None
                    ):
                        speakers[partition].add((item.dataset, item.speaker_id))
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
                            target_audio=target_audio,
                            executor=executor,
                        )
                        events = []
                        pending_checks = 0
            _flush_audio_checks(
                events,
                errors,
                verify_checksums=verify_checksums,
                known_audio=known_audio,
                target_audio=target_audio,
                executor=executor,
            )
            manifest_counts[partition] = manifest_records
    finally:
        if executor is not None:
            executor.shutdown()

    if expected_counts is not None:
        expected_names = set(expected_counts)
        actual_names = {manifest.stem for manifest in manifests}
        for partition in sorted(expected_names - actual_names):
            errors.append(f"required manifest {partition!r} is missing")
        for partition in sorted(actual_names - expected_names):
            errors.append(f"unexpected manifest {partition!r} was provided")
        for partition, expected in sorted(expected_counts.items()):
            actual_count = manifest_counts.get(partition, 0)
            if actual_count != expected:
                errors.append(
                    f"{partition}.jsonl: run.json records {expected} items, found {actual_count}"
                )
        if manifest_counts.get("train", 0) == 0:
            errors.append("train.jsonl: training partition must not be empty")

    leaked_speakers = speakers["train"] & speakers["validation"]
    for dataset, speaker_id in sorted(leaked_speakers):
        errors.append(
            f"{dataset}:{speaker_id}: speaker appears in both train and validation manifests"
        )

    return ValidationReport(
        manifests=len(manifests),
        records=records,
        audio_files=len(audio_files),
        errors=tuple(errors),
    )


def _audio_declaration_errors(
    item: ManifestItem,
    target: AudioFormat,
) -> tuple[str, ...]:
    errors: list[str] = []
    if item.sample_rate_hz != target.sample_rate_hz:
        errors.append(
            f"{item.id}: manifest declares {item.sample_rate_hz} Hz, "
            f"edition requires {target.sample_rate_hz} Hz"
        )
    if item.channels != target.channels:
        errors.append(
            f"{item.id}: manifest declares {item.channels} channels, "
            f"edition requires {target.channels}"
        )
    if item.audio_path.suffix.lower() != f".{target.container}":
        errors.append(f"{item.id}: edition requires {target.container} audio")
    return tuple(errors)


def _flush_audio_checks(
    events: list[str | _AudioCheck],
    errors: list[str],
    *,
    verify_checksums: bool,
    known_audio: Mapping[Path, AudioMetadata] | None,
    target_audio: AudioFormat | None,
    executor: ThreadPoolExecutor | None,
) -> None:
    checks = [event for event in events if isinstance(event, _AudioCheck)]
    if executor is None:
        results = [
            _inspect_audio(
                check,
                verify_checksums=verify_checksums,
                known_audio=known_audio,
                target_audio=target_audio,
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
                target_audio=target_audio,
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
    target_audio: AudioFormat | None,
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
    expected_sample_rate = target_audio.sample_rate_hz if target_audio else item.sample_rate_hz
    expected_channels = target_audio.channels if target_audio else item.channels
    if metadata.sample_rate_hz != expected_sample_rate:
        errors.append(f"{item.id}: sample-rate mismatch")
    if metadata.channels != expected_channels:
        errors.append(f"{item.id}: channel-count mismatch")
    if target_audio is not None and metadata.format.lower() != target_audio.container.lower():
        errors.append(f"{item.id}: container mismatch")
    if target_audio is not None and metadata.subtype != "PCM_16":
        errors.append(f"{item.id}: sample-format mismatch")
    if metadata.num_frames != item.frame_count:
        errors.append(f"{item.id}: frame-count mismatch")
    if metadata.format != "WAV" or metadata.subtype != "PCM_16":
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
