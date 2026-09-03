# Copyright 2026 Cisco Systems, Inc. and its affiliates
# Apache-2.0

"""Publish a completed preparation run as one release directory."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePath
from typing import Any

from .audio import AudioMetadata
from .config import LoadedEdition
from .exporters.kaldi import export_kaldi
from .models import (
    ChannelMode,
    ManifestItem,
    MediaKind,
    PreparationRunMetadata,
)
from .state import FileIdentity, canonical_json, fingerprint, sha256_file
from .validation import release_fingerprint, validate_release

Progress = Callable[[str], None]
_BASE_PARTITIONS = ("train", "validation", "evaluation")


class ReleaseError(ValueError):
    """Raised when a preparation run cannot be published."""


@dataclass(frozen=True)
class ReleaseResult:
    """Summary of one release directory."""

    output: Path
    counts: Mapping[str, int]
    total_audio_bytes: int
    total_audio_frames: int
    release_fingerprint: str


@dataclass
class _Stats:
    total_audio_bytes: int = 0
    total_audio_frames: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    dataset_counts: Counter[tuple[str, str, str]] = field(default_factory=Counter)
    checksums: dict[str, str] = field(default_factory=dict)
    manifest_digests: dict[str, str] = field(default_factory=dict)
    audio_directories: set[Path] = field(default_factory=set)


def publish_release(
    *,
    output: Path,
    workspace: Path,
    loaded: LoadedEdition,
    manifests: Mapping[str, Path],
    run: PreparationRunMetadata,
    known_audio: dict[Path, AudioMetadata] | None = None,
    workers: int = 8,
    progress: Progress | None = None,
) -> ReleaseResult:
    """Link prepared audio into a staging directory, validate it, and publish it."""

    if workers < 1:
        raise ReleaseError("workers must be positive")
    destination, workspace = preflight_release_output(output, workspace)
    notify = progress or (lambda _message: None)
    expected_counts, expected_digests = _require_manifest_set(manifests, run)

    if destination.exists():
        return _reuse_release(destination, run, workers, notify)

    staging = destination.parent / f".{destination.name}.{run['run_fingerprint'][:16]}.staging"
    _prepare_staging(staging)
    (staging / "manifests").mkdir(parents=True, exist_ok=True)

    stats = _Stats()
    source_root = (workspace / "prepared").resolve()
    for partition in (*_BASE_PARTITIONS, *sorted(set(manifests) - set(_BASE_PARTITIONS))):
        source_manifest = manifests.get(partition)
        if source_manifest is None:
            continue
        count = _publish_manifest(
            source_manifest,
            staging / "manifests" / f"{partition}.jsonl",
            partition=partition,
            expected_digest=expected_digests[partition],
            source_root=source_root,
            staging=staging,
            known_audio=known_audio,
            stats=stats,
        )
        expected_count = expected_counts[partition]
        if count != expected_count:
            raise ReleaseError(f"{partition}: run declares {expected_count} records, found {count}")
        stats.counts[partition] = count
        stats.manifest_digests[partition] = expected_digests[partition]
        notify(f"Prepared {partition}.jsonl ({count} records)")
        export_kaldi(
            staging / "manifests" / f"{partition}.jsonl",
            staging / "kaldi" / partition,
            relative_audio_paths=True,
        )
        notify(f"Prepared kaldi/{partition}")

    _write_metadata(staging, loaded, run, stats)
    _write_checksums(staging, stats)

    notify("Validating complete release directory")
    report = validate_release(
        staging,
        workers=workers,
        progress=notify,
        known_audio=known_audio,
        known_audio_root=source_root,
    )
    if not report.ok:
        details = "\n".join(f"  {error}" for error in report.errors[:20])
        raise ReleaseError(f"release validation failed:\n{details}")
    assert report.release_fingerprint is not None
    if destination.exists():
        raise FileExistsError(f"release output appeared during preparation: {destination}")
    os.replace(staging, destination)
    _fsync_directory(destination.parent)
    notify(f"Published data release: {destination}")
    return ReleaseResult(
        output=destination,
        counts=report.counts,
        total_audio_bytes=report.total_audio_bytes,
        total_audio_frames=report.total_audio_frames,
        release_fingerprint=report.release_fingerprint,
    )


def preflight_release_output(output: Path, workspace: Path) -> tuple[Path, Path]:
    """Resolve the output and require hard-link-compatible placement."""

    expanded = output.expanduser()
    absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
    destination = absolute.parent.resolve() / absolute.name
    workspace = workspace.expanduser().resolve()
    if destination == workspace or destination.is_relative_to(workspace):
        raise ReleaseError("release output must be outside the build workspace")
    if os.path.lexists(destination) and destination.is_symlink():
        raise ReleaseError(f"release output must not be a symlink: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.stat().st_dev != workspace.stat().st_dev:
        raise ReleaseError("workspace and output must be on the same filesystem")
    return destination, workspace


def _require_manifest_set(
    manifests: Mapping[str, Path],
    run: PreparationRunMetadata,
) -> tuple[dict[str, int], dict[str, str]]:
    counts: dict[str, int] = {}
    for partition, key in {
        "train": "training",
        "validation": "validation",
        "evaluation": "evaluation",
    }.items():
        count = run["counts"].get(key)
        if count is None:
            raise ReleaseError(f"run metadata is missing the {key!r} count")
        counts[partition] = count
    open_evaluation = run["counts"]["open_evaluation"]
    if open_evaluation is not None:
        counts["open-evaluation"] = open_evaluation
    counts.update(
        (partition, count)
        for partition, count in run["counts"].items()
        if partition.startswith("test-") and count is not None
    )
    expected = set(counts)
    actual = set(manifests)
    if actual != expected:
        raise ReleaseError(
            "prepared manifests differ from run metadata; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    digests: dict[str, str] = {}
    for partition, path in manifests.items():
        declaration = run["manifests"].get(partition)
        if declaration is None:
            raise ReleaseError(f"run metadata does not declare {partition!r}")
        _require_regular_file(path, f"prepared {partition} manifest")
        digests[partition] = declaration["sha256"]
    return counts, digests


def _publish_manifest(
    source_manifest: Path,
    destination_manifest: Path,
    *,
    partition: str,
    expected_digest: str,
    source_root: Path,
    staging: Path,
    known_audio: dict[Path, AudioMetadata] | None,
    stats: _Stats,
) -> int:
    digest = hashlib.sha256()
    count = 0

    with source_manifest.open("rb") as source, destination_manifest.open("wb") as destination:
        for line_number, line in enumerate(source, start=1):
            digest.update(line)
            destination.write(line)
            context = f"{source_manifest}:{line_number}"
            if not line.strip():
                raise ReleaseError(f"{context}: blank JSONL line")
            try:
                item = ManifestItem.model_validate(
                    json.loads(line, object_pairs_hook=_reject_duplicate_keys)
                )
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                raise ReleaseError(f"{context}: invalid manifest record: {error}") from error

            relative = item.audio_path.as_posix()
            if (
                len(item.audio_path.parts) < 3
                or item.audio_path.parts[0] != "audio"
                or item.audio_path.parts[1] != item.dataset
            ):
                raise ReleaseError(f"{context}: invalid audio path {relative!r}")
            source_audio = _contained(source_root, item.audio_path)
            metadata = known_audio.get(source_audio) if known_audio is not None else None
            cached = (
                metadata is not None
                and metadata.checksum_fresh
                and metadata.matches_stat(source_audio)
            )
            destination_audio = _contained(staging, item.audio_path)
            if destination_audio.parent not in stats.audio_directories:
                destination_audio.parent.mkdir(parents=True, exist_ok=True)
                stats.audio_directories.add(destination_audio.parent)
            linked_identity = _link_audio(source_audio, destination_audio)
            if cached:
                assert metadata is not None
                metadata = replace(metadata, identity=linked_identity)
                assert known_audio is not None
                known_audio[source_audio] = metadata

            stats.total_audio_bytes += linked_identity.size
            stats.total_audio_frames += item.frame_count
            stats.dataset_counts[(item.dataset, partition, item.media_kind.value)] += 1
            stats.checksums[relative] = item.checksum
            count += 1

    actual_digest = digest.hexdigest()
    if actual_digest != expected_digest:
        raise ReleaseError(f"{source_manifest}: digest differs from run metadata")
    return count


def _write_metadata(
    staging: Path,
    loaded: LoadedEdition,
    run: PreparationRunMetadata,
    stats: _Stats,
) -> None:
    datasets = _dataset_index(loaded)
    metadata_dir = staging / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    datasets_path = metadata_dir / "datasets.json"
    datasets_path.write_text(f"{canonical_json({'datasets': datasets})}\n", encoding="utf-8")
    provenance = {
        "config_fingerprint": run["config_fingerprint"],
        "dependency_lock_digest": run["dependency_lock_digest"],
        "edition": run["edition"],
        "environment": run["environment"],
        "implementation_fingerprint": run["implementation_fingerprint"],
        "input_fingerprint": run["input_fingerprint"],
        "inventory_digests": run["inventory_digests"],
        "run_fingerprint": run["run_fingerprint"],
        "selected_source_digest": run["selected_source_digest"],
        "selection": run["selection"],
        "source_artifacts_digest": fingerprint(run["source_artifacts"]),
    }
    provenance_path = metadata_dir / "provenance.json"
    provenance_path.write_text(f"{canonical_json(provenance)}\n", encoding="utf-8")
    _write_licenses(staging, datasets)

    manifests = {
        partition: {
            "path": f"manifests/{partition}.jsonl",
            "records": count,
            "sha256": stats.manifest_digests[partition],
        }
        for partition, count in stats.counts.items()
    }
    dataset_counts: dict[str, dict[str, dict[str, int]]] = {}
    for (dataset, partition, kind), count in sorted(stats.dataset_counts.items()):
        dataset_counts.setdefault(dataset, {}).setdefault(partition, {})[kind] = count
    release: dict[str, Any] = {
        "audio": loaded.config.audio.model_dump(mode="json"),
        "counts": dict(stats.counts),
        "dataset_counts": dataset_counts,
        "dataset_index": "metadata/datasets.json",
        "edition": loaded.config.edition,
        "manifests": manifests,
        "provenance": "metadata/provenance.json",
        "selection": run["selection"],
        "total_audio_bytes": stats.total_audio_bytes,
        "total_audio_frames": stats.total_audio_frames,
    }
    release["release_fingerprint"] = release_fingerprint(
        release,
        dataset_index_sha256=sha256_file(datasets_path),
        provenance_sha256=sha256_file(provenance_path),
    )
    (staging / "release.json").write_text(f"{canonical_json(release)}\n", encoding="utf-8")
    _write_readme(staging, loaded, stats)


def _dataset_index(loaded: LoadedEdition) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = [
        {
            "id": dataset.id,
            "license": dataset.license,
            "media_kinds": sorted(kind.value for kind in dataset.media_kinds),
            "release": dataset.release,
            "source_urls": sorted({source.url for source in dataset.sources if source.url}),
        }
        for dataset in loaded.config.datasets
    ]
    public = loaded.config.public_evaluation
    if public is not None:
        entries.append(
            {
                "id": public.id,
                "license": "See the upstream evaluation repository",
                "media_kinds": ["speech"],
                "release": public.revision,
                "source_urls": [public.repository_url],
            }
        )
    return sorted(entries, key=lambda entry: str(entry["id"]))


def _write_licenses(staging: Path, datasets: list[dict[str, object]]) -> None:
    (staging / "licenses").mkdir(exist_ok=True)
    lines = [
        "# Dataset licensing index",
        "",
        "Each source remains governed by its upstream terms. Consult the upstream source",
        "before redistribution or use.",
        "",
        "| Dataset | Release | Declared terms | Sources |",
        "|---|---|---|---|",
    ]
    for dataset in datasets:
        urls = dataset["source_urls"]
        assert isinstance(urls, list)
        lines.append(
            f"| {_markdown(dataset['id'])} | {_markdown(dataset['release'])} | "
            f"{_markdown(dataset['license'])} | {_markdown(', '.join(urls))} |"
        )
    (staging / "licenses" / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_readme(staging: Path, loaded: LoadedEdition, stats: _Stats) -> None:
    rows = [
        f"| {partition.replace('-', ' ').title()} | {count:,} |"
        for partition, count in stats.counts.items()
    ]
    target = loaded.config.audio
    actions = {
        ChannelMode.PRESERVE: "preserves source channels",
        ChannelMode.DOWNMIX: "is downmixed to mono",
    }
    labels = {MediaKind.SPEECH: "speech", MediaKind.NOISE: "noise", MediaKind.RIR: "RIR"}
    policy = "; ".join(
        f"{labels[kind]} {actions[target.channels.for_media_kind(kind)]}" for kind in MediaKind
    )
    hours = stats.total_audio_frames / target.sample_rate_hz / 3600
    text = f"""# LRAC {loaded.config.edition} data release

This directory contains the complete data prepared from the LRAC
{loaded.config.edition} edition configuration. All manifest audio paths are relative to
this directory. Audio is {target.sample_rate_hz / 1000:g} kHz IEEE float32
{target.container.upper()}. Resampled samples are written without clipping,
normalization, limiting, or gain adjustment. Channel handling by media kind: {policy}.

## Partitions

| Partition | Records |
|---|---:|
{chr(10).join(rows)}
| **Total** | **{sum(stats.counts.values()):,}** |

The collection contains {hours:,.3f} hours of audio. Canonical JSONL manifests
are under `manifests/`; matching ESPnet/Kaldi data directories are under
`kaldi/`, with portable audio paths relative to this release root. Software that
requires absolute paths should regenerate the Kaldi directories with `export-kaldi`
as documented in the data-preparation repository. See `release.json` for the
machine-readable contract and `licenses/README.md` for upstream terms.
"""
    (staging / "README.md").write_text(text, encoding="utf-8")


def _write_checksums(staging: Path, stats: _Stats) -> None:
    checksums = dict(stats.checksums)
    fixed = (
        "README.md",
        "licenses/README.md",
        "metadata/datasets.json",
        "metadata/provenance.json",
        "release.json",
    )
    for relative in fixed:
        checksums[relative] = sha256_file(staging / relative)
    for partition, digest in stats.manifest_digests.items():
        checksums[f"manifests/{partition}.jsonl"] = digest
    for path in sorted((staging / "kaldi").rglob("*")):
        if path.is_file():
            checksums[path.relative_to(staging).as_posix()] = sha256_file(path)
    (staging / "SHA256SUMS").write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest in sorted(checksums.items())),
        encoding="utf-8",
    )


def _reuse_release(
    destination: Path,
    run: PreparationRunMetadata,
    workers: int,
    progress: Progress,
) -> ReleaseResult:
    report = validate_release(destination, workers=workers, progress=progress)
    if not report.ok:
        raise ReleaseError(f"existing output is invalid: {report.errors[0]}")
    if report.run_fingerprint != run["run_fingerprint"]:
        raise FileExistsError(f"release output already exists for another run: {destination}")
    assert report.release_fingerprint is not None
    return ReleaseResult(
        output=destination,
        counts=report.counts,
        total_audio_bytes=report.total_audio_bytes,
        total_audio_frames=report.total_audio_frames,
        release_fingerprint=report.release_fingerprint,
    )


def _prepare_staging(path: Path) -> None:
    if os.path.lexists(path):
        metadata = path.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise ReleaseError(f"release staging path is not a directory: {path}")
    else:
        path.mkdir()


def _link_audio(source: Path, destination: Path) -> FileIdentity:
    source_metadata = _require_regular_file(source, "prepared audio")
    if os.path.lexists(destination):
        existing = _require_regular_file(destination, "staged audio")
        if (existing.st_dev, existing.st_ino) == (
            source_metadata.st_dev,
            source_metadata.st_ino,
        ):
            return FileIdentity.from_stat(source_metadata)
        destination.unlink()
    os.link(source, destination)
    return FileIdentity.from_stat(source.stat())


def _require_regular_file(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ReleaseError(f"{label} is unavailable: {path}: {error}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ReleaseError(f"{label} is not a regular file: {path}")
    return metadata


def _contained(root: Path, relative: PurePath) -> Path:
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ReleaseError(f"path escapes managed directory: {root / relative}") from error
    return resolved


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _markdown(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
