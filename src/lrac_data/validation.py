# Copyright 2026 Cisco Systems, Inc. and its affiliates
# Apache-2.0

"""Validation for a self-contained data release.

The release is produced in a local staging directory and is trusted not to be
modified while validation runs. Validation therefore uses ordinary ``Path``
operations: reject unsafe paths and symlinks, validate the JSON contracts, then
hash and inspect every declared file before publication.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import field_validator, model_validator

from .audio import AudioMetadata, is_float32_wav
from .exporters.kaldi import export_kaldi
from .models import (
    AudioFormat,
    ChannelMode,
    ContractModel,
    ManifestItem,
    MediaKind,
    NonEmptyText,
    NonNegativeInt,
    PathSegment,
    PositiveInt,
    SelectionMode,
    Sha256,
    Split,
)
from .state import fingerprint, sha256_file

_PARTITION_SPLITS = {
    "train": Split.TRAIN,
    "validation": Split.VALIDATION,
    "evaluation": Split.EVALUATION,
    "open-evaluation": Split.EVALUATION,
}
_BASE_PARTITIONS = frozenset({"train", "validation", "evaluation"})
_TEST_PARTITION = re.compile(r"test-[a-z0-9]+(?:-[a-z0-9]+)*")
_FIXED_PATHS = frozenset(
    {
        "README.md",
        "licenses/README.md",
        "metadata/datasets.json",
        "metadata/provenance.json",
        "release.json",
    }
)
_SHA256SUMS = "SHA256SUMS"
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_PROGRESS_INTERVAL = 10_000

DatasetCounts = dict[str, dict[str, dict[MediaKind, int]]]


def _partition_split(partition: str) -> Split | None:
    split = _PARTITION_SPLITS.get(partition)
    if split is not None:
        return split
    return Split.EVALUATION if _TEST_PARTITION.fullmatch(partition) else None


class _ManifestDeclaration(ContractModel):
    path: PurePosixPath
    records: NonNegativeInt
    sha256: Sha256


class _ReleaseMetadata(ContractModel):
    audio: AudioFormat
    counts: dict[str, NonNegativeInt]
    dataset_counts: dict[PathSegment, dict[str, dict[MediaKind, PositiveInt]]]
    dataset_index: Literal["metadata/datasets.json"]
    edition: NonEmptyText
    manifests: dict[str, _ManifestDeclaration]
    provenance: Literal["metadata/provenance.json"]
    release_fingerprint: Sha256
    selection: SelectionMode
    total_audio_bytes: NonNegativeInt
    total_audio_frames: NonNegativeInt

    @field_validator("audio", mode="before")
    @classmethod
    def require_complete_audio_contract(cls, value: object) -> object:
        required = {"sample_rate_hz", "channels", "sample_format", "container"}
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError("audio must declare the complete target format")
        return value

    @model_validator(mode="after")
    def validate_partitions(self) -> _ReleaseMetadata:
        partitions = set(self.counts)
        if not partitions >= _BASE_PARTITIONS or any(
            _partition_split(partition) is None for partition in partitions
        ):
            raise ValueError("counts must contain the supported base partitions")
        if self.counts["train"] == 0:
            raise ValueError("the training partition must not be empty")
        if set(self.manifests) != partitions:
            raise ValueError("manifests must match counted partitions")
        for partition, declaration in self.manifests.items():
            if declaration.path.as_posix() != f"manifests/{partition}.jsonl":
                raise ValueError(f"invalid path for {partition!r} manifest")
            if declaration.records != self.counts[partition]:
                raise ValueError(f"invalid record count for {partition!r} manifest")
        for dataset, by_partition in self.dataset_counts.items():
            if not by_partition or not set(by_partition) <= partitions:
                raise ValueError(f"invalid dataset counts for {dataset!r}")
            if any(not by_media for by_media in by_partition.values()):
                raise ValueError(f"empty dataset counts for {dataset!r}")
        return self


class _DatasetEntry(ContractModel):
    id: PathSegment
    license: NonEmptyText
    media_kinds: tuple[MediaKind, ...]
    release: NonEmptyText
    source_urls: tuple[str, ...]

    @model_validator(mode="after")
    def validate_lists(self) -> _DatasetEntry:
        kinds = [kind.value for kind in self.media_kinds]
        if not kinds or kinds != sorted(set(kinds)):
            raise ValueError("media_kinds must be nonempty, unique, and sorted")
        if list(self.source_urls) != sorted(set(self.source_urls)):
            raise ValueError("source_urls must be unique and sorted")
        if any(not _is_public_url(url) for url in self.source_urls):
            raise ValueError("source_urls must contain public HTTP(S) URLs")
        return self


class _DatasetIndex(ContractModel):
    datasets: tuple[_DatasetEntry, ...]

    @field_validator("datasets")
    @classmethod
    def validate_datasets(cls, datasets: tuple[_DatasetEntry, ...]) -> tuple[_DatasetEntry, ...]:
        ids = [dataset.id for dataset in datasets]
        if not ids or ids != sorted(set(ids)):
            raise ValueError("dataset IDs must be nonempty, unique, and sorted")
        return datasets


class _Provenance(ContractModel):
    config_fingerprint: Sha256
    dependency_lock_digest: Sha256 | None
    edition: NonEmptyText
    environment: dict[str, Any]
    implementation_fingerprint: Sha256
    input_fingerprint: Sha256
    inventory_digests: dict[PathSegment, Sha256]
    run_fingerprint: Sha256
    selected_source_digest: Sha256
    selection: SelectionMode
    source_artifacts_digest: Sha256

    @field_validator("inventory_digests")
    @classmethod
    def require_inventories(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("inventory_digests must not be empty")
        return value


@dataclass(frozen=True, slots=True)
class ReleaseValidationReport:
    """Result of validating one complete release."""

    release: Path
    counts: Mapping[str, int]
    total_audio_bytes: int
    total_audio_frames: int
    errors: tuple[str, ...]
    run_fingerprint: str | None = None
    release_fingerprint: str | None = None

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True, slots=True)
class _AudioItem:
    id: str
    checksum: str
    channels: int
    frame_count: int


@dataclass(frozen=True, slots=True)
class _ManifestResult:
    counts: dict[str, int]
    dataset_counts: DatasetCounts
    datasets: set[str]
    audio_items: dict[str, _AudioItem]
    total_audio_frames: int


class _ContractError(ValueError):
    """A structural error that prevents meaningful validation."""


def release_fingerprint(
    metadata: Mapping[str, Any],
    *,
    dataset_index_sha256: str,
    provenance_sha256: str,
) -> str:
    """Return the canonical fingerprint of a release's local contract."""

    return fingerprint(
        {
            "audio": metadata.get("audio"),
            "counts": metadata.get("counts"),
            "dataset_counts": metadata.get("dataset_counts"),
            "dataset_index_sha256": dataset_index_sha256,
            "edition": metadata.get("edition"),
            "manifests": metadata.get("manifests"),
            "provenance_sha256": provenance_sha256,
            "selection": metadata.get("selection"),
            "total_audio_bytes": metadata.get("total_audio_bytes"),
            "total_audio_frames": metadata.get("total_audio_frames"),
        }
    )


def validate_release(
    release: Path,
    *,
    workers: int = 8,
    progress: Callable[[str], None] | None = None,
    known_audio: Mapping[Path, AudioMetadata] | None = None,
    known_audio_root: Path | None = None,
) -> ReleaseValidationReport:
    """Validate a complete, locally staged release."""

    release = release.expanduser().absolute()
    errors: list[str] = []
    if workers < 1:
        return _empty_report(release, ["workers must be positive"])

    try:
        tree_files, tree_directories, tree_errors = _scan_tree(release)
        errors.extend(tree_errors)
        checksums = _read_checksum_index(release)
        release_metadata = _read_json_object(release, "release.json")
        contract = _parse_release_contract(release / "release.json", release_metadata)
        datasets_metadata = _read_json_object(release, "metadata/datasets.json")
        provenance_metadata = _read_json_object(release, "metadata/provenance.json")
        datasets = _parse_dataset_index(release / "metadata/datasets.json", datasets_metadata)
        provenance = _validate_provenance(
            release / "metadata/provenance.json",
            provenance_metadata,
            release_contract=contract,
            indexed_datasets=set(datasets),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, _ContractError) as error:
        errors.append(str(error))
        return _empty_report(release, errors)

    result = _validate_manifests(release, contract, checksums, datasets, errors, progress)
    kaldi_paths = _validate_kaldi_views(release, contract, checksums, errors, progress)
    fixed_paths = {
        *_FIXED_PATHS,
        *(declaration.path.as_posix() for declaration in contract.manifests.values()),
    }
    fixed_digests = _validate_fixed_files(release, checksums, errors)
    total_audio_bytes = _validate_audio_files(
        release,
        checksums,
        result.audio_items,
        contract.audio,
        workers,
        errors,
        progress,
        known_audio,
        known_audio_root,
    )
    _reconcile_metadata(contract, result, total_audio_bytes, errors)
    _reconcile_datasets(datasets, result.datasets, errors)
    _validate_checksum_coverage(
        checksums,
        fixed_paths | kaldi_paths,
        set(result.audio_items),
        errors,
    )
    _validate_tree(
        release,
        tree_files,
        tree_directories,
        fixed_paths | kaldi_paths | set(result.audio_items) | {_SHA256SUMS},
        errors,
        progress,
    )
    _validate_fingerprints(release_metadata, fixed_digests, errors)

    _emit(progress, f"Release validation complete ({sum(result.counts.values())} records)")
    return ReleaseValidationReport(
        release=release,
        counts=result.counts,
        total_audio_bytes=total_audio_bytes,
        total_audio_frames=result.total_audio_frames,
        errors=tuple(errors),
        run_fingerprint=provenance.run_fingerprint,
        release_fingerprint=contract.release_fingerprint,
    )


def _empty_report(release: Path, errors: list[str]) -> ReleaseValidationReport:
    return ReleaseValidationReport(release, {}, 0, 0, tuple(errors))


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _safe_relative(value: object, *, context: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise _ContractError(f"{context}: unsafe path {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path == PurePosixPath(".")
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or path.as_posix() != value
    ):
        raise _ContractError(f"{context}: unsafe path {value!r}")
    return path


def _regular_file(release: Path, relative: str) -> Path:
    relative_path = _safe_relative(relative, context=str(release))
    current = release
    parts = relative_path.parts
    for index, part in enumerate(parts):
        current /= part
        try:
            details = current.lstat()
        except OSError as error:
            raise _ContractError(f"{current}: cannot inspect path: {error}") from error
        expected = stat.S_ISREG if index == len(parts) - 1 else stat.S_ISDIR
        if not expected(details.st_mode):
            kind = "file" if index == len(parts) - 1 else "directory"
            raise _ContractError(f"{current}: expected a regular {kind}")
    return current


def _scan_tree(release: Path) -> tuple[set[str], set[str], list[str]]:
    try:
        root = release.lstat()
    except OSError as error:
        raise _ContractError(f"{release}: cannot inspect release root: {error}") from error
    if not stat.S_ISDIR(root.st_mode):
        raise _ContractError(f"{release}: release root is not a regular directory")

    files: set[str] = set()
    directories: set[str] = set()
    errors: list[str] = []

    def walk_error(error: OSError) -> None:
        errors.append(f"{release}: cannot walk release tree: {error}")

    for current, names, filenames in os.walk(
        release, topdown=True, followlinks=False, onerror=walk_error
    ):
        parent = Path(current)
        retained: list[str] = []
        for name in sorted(names):
            path = parent / name
            relative = path.relative_to(release).as_posix()
            try:
                details = path.lstat()
            except OSError as error:
                errors.append(f"{path}: cannot inspect directory: {error}")
                continue
            if stat.S_ISDIR(details.st_mode):
                directories.add(relative)
                retained.append(name)
            else:
                errors.append(f"{path}: release contains a symlink or special entry")
        names[:] = retained
        for name in sorted(filenames):
            path = parent / name
            relative = path.relative_to(release).as_posix()
            try:
                details = path.lstat()
            except OSError as error:
                errors.append(f"{path}: cannot inspect file: {error}")
                continue
            if stat.S_ISREG(details.st_mode):
                files.add(relative)
            else:
                errors.append(f"{path}: release contains a symlink or special entry")
    return files, directories, errors


def _read_checksum_index(release: Path) -> dict[str, str]:
    path = _regular_file(release, _SHA256SUMS)
    checksums: dict[str, str] = {}
    previous: str | None = None
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            digest, separator, relative = line.rstrip("\n").partition("  ")
            if not separator or not _is_sha256(digest) or "\r" in relative:
                raise _ContractError(f"{path}:{line_number}: invalid checksum line")
            relative = _safe_relative(relative, context=f"{path}:{line_number}").as_posix()
            if previous is not None and relative <= previous:
                raise _ContractError(
                    f"{path}:{line_number}: checksum paths are not strictly sorted"
                )
            checksums[relative] = digest
            previous = relative
    if not checksums:
        raise _ContractError(f"{path}: checksum index is empty")
    return checksums


def _read_json_object(release: Path, relative: str) -> dict[str, Any]:
    path = _regular_file(release, relative)
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise _ContractError(f"{path}: expected a JSON object")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _ContractError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _parse_release_contract(path: Path, value: dict[str, Any]) -> _ReleaseMetadata:
    try:
        return _ReleaseMetadata.model_validate(value)
    except ValueError as error:
        raise _ContractError(f"{path}: invalid release contract: {error}") from error


def _parse_dataset_index(path: Path, value: dict[str, Any]) -> dict[str, _DatasetEntry]:
    try:
        parsed = _DatasetIndex.model_validate(value)
    except ValueError as error:
        raise _ContractError(f"{path}: invalid dataset index: {error}") from error
    return {dataset.id: dataset for dataset in parsed.datasets}


def _validate_provenance(
    path: Path,
    value: dict[str, Any],
    *,
    release_contract: _ReleaseMetadata,
    indexed_datasets: set[str],
) -> _Provenance:
    try:
        parsed = _Provenance.model_validate(value)
    except ValueError as error:
        raise _ContractError(f"{path}: invalid provenance: {error}") from error
    if set(parsed.inventory_digests) != indexed_datasets:
        raise _ContractError(f"{path}: inventory_digests dataset IDs differ from dataset index")
    if parsed.edition != release_contract.edition:
        raise _ContractError(f"{path}: edition does not match release.json")
    if parsed.selection is not release_contract.selection:
        raise _ContractError(f"{path}: selection does not match release.json")
    return parsed


def _validate_manifests(
    release: Path,
    contract: _ReleaseMetadata,
    checksums: Mapping[str, str],
    datasets: Mapping[str, _DatasetEntry],
    errors: list[str],
    progress: Callable[[str], None] | None,
) -> _ManifestResult:
    counts: dict[str, int] = {}
    counter: Counter[tuple[str, str, MediaKind]] = Counter()
    represented_datasets: set[str] = set()
    audio_items: dict[str, _AudioItem] = {}
    seen_ids: set[str] = set()
    speakers: dict[str, set[tuple[str, str]]] = {"train": set(), "validation": set()}
    recordings: dict[str, set[tuple[str, str]]] = {"train": set(), "validation": set()}
    total_frames = 0
    total_records = 0

    for partition in contract.counts:
        expected_split = _partition_split(partition)
        assert expected_split is not None
        manifest_declaration = contract.manifests[partition]
        relative = manifest_declaration.path.as_posix()
        manifest = release / relative
        try:
            source = _regular_file(release, relative).open("rb")
        except (OSError, _ContractError) as error:
            errors.append(str(error))
            counts[partition] = 0
            continue
        count = 0
        digest = hashlib.sha256()
        previous_id: str | None = None
        with source:
            for line_number, encoded in enumerate(source, start=1):
                digest.update(encoded)
                context = f"{manifest}:{line_number}"
                try:
                    line = encoded.decode("utf-8")
                    if not line.strip():
                        raise ValueError("blank JSONL line")
                    item = ManifestItem.model_validate(
                        json.loads(line, object_pairs_hook=_unique_object)
                    )
                except (UnicodeError, json.JSONDecodeError, ValueError) as error:
                    errors.append(f"{context}: invalid manifest record: {error}")
                    continue
                count += 1
                total_records += 1
                total_frames += item.frame_count
                if total_records % _PROGRESS_INTERVAL == 0:
                    _emit(progress, f"Validated {total_records} manifest records")
                if previous_id is not None and item.id <= previous_id:
                    errors.append(f"{context}: IDs are not strictly ordered")
                previous_id = item.id
                if item.id in seen_ids:
                    errors.append(f"{context}: duplicate release ID {item.id!r}")
                seen_ids.add(item.id)
                if item.split is not expected_split:
                    errors.append(f"{context}: split does not match {partition!r}")
                dataset_declaration = datasets.get(item.dataset)
                if dataset_declaration is not None:
                    if item.source_release != dataset_declaration.release:
                        errors.append(f"{context}: source release differs from dataset index")
                    if item.media_kind not in dataset_declaration.media_kinds:
                        errors.append(f"{context}: media kind is not declared in dataset index")
                _validate_audio_declaration(item, contract.audio, context, errors)
                if (
                    partition in speakers
                    and item.media_kind is MediaKind.SPEECH
                    and item.speaker_id
                ):
                    speakers[partition].add((item.dataset, item.speaker_id))
                recording_id = item.metadata.get("recording_id")
                if partition in recordings and isinstance(recording_id, str) and recording_id:
                    recordings[partition].add((item.dataset, recording_id))

                relative_audio = item.audio_path.as_posix()
                if not _valid_audio_path(item, contract.audio):
                    errors.append(f"{context}: invalid release audio path {relative_audio!r}")
                elif relative_audio in audio_items:
                    errors.append(f"{context}: duplicate release audio path {relative_audio!r}")
                else:
                    audio_items[relative_audio] = _AudioItem(
                        item.id,
                        item.checksum,
                        item.channels,
                        item.frame_count,
                    )
                    expected = checksums.get(relative_audio)
                    if expected is None:
                        errors.append(f"{context}: SHA256SUMS does not declare {relative_audio}")
                    elif expected != item.checksum:
                        errors.append(f"{context}: SHA256SUMS differs from the manifest checksum")
                counter[(partition, item.dataset, item.media_kind)] += 1
                represented_datasets.add(item.dataset)
        counts[partition] = count
        actual_digest = digest.hexdigest()
        indexed_digest = checksums.get(relative)
        if indexed_digest is None:
            errors.append(f"{manifest}: SHA256SUMS entry is missing")
        elif actual_digest != indexed_digest:
            errors.append(f"{manifest}: checksum differs from SHA256SUMS")
        if actual_digest != manifest_declaration.sha256:
            errors.append(f"{manifest}: digest differs from release.json for {partition!r}")
        _emit(progress, f"Validated {partition}.jsonl ({count} records)")

    for dataset, speaker in sorted(speakers["train"] & speakers["validation"]):
        errors.append(
            f"{dataset}:{speaker}: speaker appears in both train and validation manifests"
        )
    for dataset, recording in sorted(recordings["train"] & recordings["validation"]):
        errors.append(
            f"{dataset}:{recording}: recording appears in both train and validation manifests"
        )
    return _ManifestResult(
        counts,
        _nested_counts(counter),
        represented_datasets,
        audio_items,
        total_frames,
    )

def _validate_audio_declaration(
    item: ManifestItem, target: AudioFormat, context: str, errors: list[str]
) -> None:
    if item.sample_rate_hz != target.sample_rate_hz:
        errors.append(f"{context}: {item.id} has the wrong sample rate")
    if (
        target.channels.for_media_kind(item.media_kind) is ChannelMode.DOWNMIX
        and item.channels != 1
    ):
        errors.append(f"{context}: {item.id} violates the downmix policy")


def _valid_audio_path(item: ManifestItem, target: AudioFormat) -> bool:
    value = item.audio_path.as_posix()
    try:
        _safe_relative(value, context=item.id)
    except _ContractError:
        return False
    parts = item.audio_path.parts
    return (
        len(parts) >= 3
        and parts[0] == "audio"
        and parts[1] == item.dataset
        and item.audio_path.suffix.lower() == f".{target.container}"
    )


def _nested_counts(counter: Counter[tuple[str, str, MediaKind]]) -> DatasetCounts:
    result: DatasetCounts = {}
    for (partition, dataset, kind), count in sorted(counter.items()):
        result.setdefault(dataset, {}).setdefault(partition, {})[kind] = count
    return result


def _validate_fixed_files(
    release: Path,
    checksums: Mapping[str, str],
    errors: list[str],
) -> dict[str, str]:
    actual: dict[str, str] = {}
    for relative in sorted(_FIXED_PATHS):
        expected = checksums.get(relative)
        if expected is None:
            errors.append(f"{release / relative}: SHA256SUMS entry is missing")
            continue
        try:
            digest = sha256_file(_regular_file(release, relative))
        except (OSError, _ContractError) as error:
            errors.append(str(error))
            continue
        actual[relative] = digest
        if digest != expected:
            errors.append(f"{release / relative}: checksum differs from SHA256SUMS")
    return actual


def _validate_kaldi_views(
    release: Path,
    contract: _ReleaseMetadata,
    checksums: Mapping[str, str],
    errors: list[str],
    progress: Callable[[str], None] | None,
) -> set[str]:
    expected_paths: set[str] = set()
    try:
        temporary_context = tempfile.TemporaryDirectory(prefix="lrac-kaldi-validation-")
    except OSError as error:
        errors.append(f"{release}: cannot create temporary Kaldi validation directory: {error}")
        return expected_paths
    with temporary_context as temporary_name:
        temporary = Path(temporary_name)
        for partition, declaration in contract.manifests.items():
            manifest = release / declaration.path
            try:
                expected = temporary / partition
                generated = export_kaldi(
                    manifest,
                    expected,
                    relative_audio_paths=True,
                )
            except (OSError, ValueError) as error:
                errors.append(f"{manifest}: cannot derive Kaldi view: {error}")
                continue

            for filename in generated:
                relative = f"kaldi/{partition}/{filename}"
                expected_paths.add(relative)
                try:
                    path = _regular_file(release, relative)
                    matches, digest = _compare_and_hash(path, expected / filename)
                except (OSError, _ContractError) as error:
                    errors.append(str(error))
                    continue
                if not matches:
                    errors.append(f"{path}: contents differ from {declaration.path}")
                indexed = checksums.get(relative)
                if indexed is None:
                    errors.append(f"{path}: SHA256SUMS entry is missing")
                elif digest != indexed:
                    errors.append(f"{path}: checksum differs from SHA256SUMS")
            _emit(progress, f"Validated kaldi/{partition}")
    return expected_paths


def _compare_and_hash(actual: Path, expected: Path) -> tuple[bool, str]:
    digest = hashlib.sha256()
    matches = True
    with actual.open("rb") as actual_stream, expected.open("rb") as expected_stream:
        while True:
            actual_chunk = actual_stream.read(1024 * 1024)
            expected_chunk = expected_stream.read(1024 * 1024)
            digest.update(actual_chunk)
            if actual_chunk != expected_chunk:
                matches = False
            if not actual_chunk and not expected_chunk:
                break
    return matches, digest.hexdigest()


def _reconcile_metadata(
    contract: _ReleaseMetadata,
    result: _ManifestResult,
    total_audio_bytes: int,
    errors: list[str],
) -> None:
    if result.counts != contract.counts:
        errors.append(
            "release.json: counts differ from manifests: "
            f"recorded={contract.counts}, found={result.counts}"
        )
    if result.dataset_counts != contract.dataset_counts:
        errors.append("release.json: dataset_counts differ from manifests")
    if total_audio_bytes != contract.total_audio_bytes:
        errors.append("release.json: total_audio_bytes differs from materialized audio")
    if result.total_audio_frames != contract.total_audio_frames:
        errors.append("release.json: total_audio_frames differs from manifests")


def _reconcile_datasets(
    datasets: Mapping[str, _DatasetEntry],
    represented: set[str],
    errors: list[str],
) -> None:
    for dataset in sorted(set(datasets) - represented):
        errors.append(f"metadata/datasets.json: dataset {dataset!r} has no manifest records")
    for dataset in sorted(represented - set(datasets)):
        errors.append(f"metadata/datasets.json: manifest dataset {dataset!r} is not indexed")


def _validate_checksum_coverage(
    checksums: Mapping[str, str], fixed: set[str], audio: set[str], errors: list[str]
) -> None:
    expected = fixed | audio
    for relative in sorted(set(checksums) - expected):
        errors.append(f"SHA256SUMS: unexpected entry {relative!r}")
    for relative in sorted(expected - set(checksums)):
        errors.append(f"SHA256SUMS: missing entry {relative!r}")


def _validate_audio_files(
    release: Path,
    checksums: Mapping[str, str],
    items: Mapping[str, _AudioItem],
    target: AudioFormat,
    workers: int,
    errors: list[str],
    progress: Callable[[str], None] | None,
    known_audio: Mapping[Path, AudioMetadata] | None,
    known_audio_root: Path | None,
) -> int:
    def validate(entry: tuple[str, _AudioItem]) -> tuple[int, list[str]]:
        relative, item = entry
        failures: list[str] = []
        path = release / relative
        size = 0
        try:
            path = _regular_file(release, relative)
            size = path.stat().st_size
            cached_path = (known_audio_root or release) / relative
            metadata = known_audio.get(cached_path) if known_audio is not None else None
            if metadata is not None and metadata.checksum_fresh and metadata.matches_stat(path):
                digest = metadata.sha256
                sample_rate = metadata.sample_rate_hz
                channels = metadata.channels
                frames = metadata.num_frames
                format_name = metadata.format
                subtype = metadata.subtype
            else:
                digest = sha256_file(path)
                import soundfile as sf

                info = sf.info(str(path))
                sample_rate = info.samplerate
                channels = info.channels
                frames = info.frames
                format_name = info.format
                subtype = info.subtype
        except (OSError, RuntimeError, ValueError, _ContractError) as error:
            return size, [f"{path}: cannot inspect audio: {error}"]
        if digest != checksums.get(relative):
            failures.append(f"{path}: audio checksum differs from SHA256SUMS")
        mismatches: list[str] = []
        if sample_rate != target.sample_rate_hz:
            mismatches.append("sample rate")
        if channels != item.channels:
            mismatches.append("channel count")
        if frames != item.frame_count:
            mismatches.append("frame count")
        if not is_float32_wav(format_name, subtype):
            mismatches.append("float32 WAV format")
        if mismatches:
            failures.append(f"{item.id}: audio differs in {', '.join(mismatches)}")
        return size, failures

    entries = sorted(items.items())
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="lrac-release-hash") as pool:
        for completed, (size, failures) in enumerate(pool.map(validate, entries), start=1):
            total_bytes += size
            errors.extend(failures)
            if completed % _PROGRESS_INTERVAL == 0:
                _emit(progress, f"Validated {completed}/{len(entries)} release audio files")
    _emit(progress, f"Validated all {len(entries)} release audio files")
    return total_bytes


def _validate_tree(
    release: Path,
    files: set[str],
    directories: set[str],
    expected_files: set[str],
    errors: list[str],
    progress: Callable[[str], None] | None,
) -> None:
    for relative in sorted(files - expected_files):
        errors.append(f"{release / relative}: unexpected release file")
    expected_directories: set[str] = set()
    for relative in expected_files:
        parent = PurePosixPath(relative).parent
        while parent != PurePosixPath("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    for relative in sorted(directories - expected_directories):
        errors.append(f"{release / relative}: unexpected release directory")
    _emit(progress, f"Checked complete release tree ({len(files)} files)")


def _validate_fingerprints(
    metadata: Mapping[str, Any], fixed_digests: Mapping[str, str], errors: list[str]
) -> None:
    dataset_digest = fixed_digests.get("metadata/datasets.json")
    provenance_digest = fixed_digests.get("metadata/provenance.json")
    if dataset_digest is None or provenance_digest is None:
        return
    expected = release_fingerprint(
        metadata,
        dataset_index_sha256=dataset_digest,
        provenance_sha256=provenance_digest,
    )
    if metadata.get("release_fingerprint") != expected:
        errors.append("release.json: release_fingerprint does not match the release contract")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_CHARACTERS for character in value)
    )


def _is_public_url(value: object) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlsplit(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc) and parsed.username is None
