"""Complete, resumable materialization of an LRAC challenge edition."""

from __future__ import annotations

import fcntl
import importlib.util
import inspect
import json
import os
import shutil
import tempfile
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, TypeVar

from . import __version__
from . import audio as audio_module
from .audio import (
    AudioMetadata,
    MaterializationTask,
    materialization_fingerprint,
    materialize_all,
    output_path,
)
from .config import LoadedEdition, load_edition_config
from .datasets import create_adapter
from .datasets.io import EXTRACTION_FORMAT_VERSION, trusted_file_sha256
from .manifests import read_jsonl, write_jsonl, write_manifest
from .models import (
    InventoryItem,
    ManifestItem,
    SelectionMode,
    SelectionResult,
    Split,
)
from .public_evaluation import fetch_public_evaluation, inventory_public_evaluation
from .selection import select_inventory
from .state import (
    FileIdentity,
    StateStore,
    atomic_write_text,
    canonical_json,
    environment_provenance,
    fingerprint,
    sha256_file,
)
from .validation import validate_manifests


@dataclass(frozen=True)
class WorkspaceLayout:
    root: Path
    downloads: Path
    extracted: Path
    prepared_audio: Path
    inventories: Path
    state: Path
    manifests: Path
    runs: Path

    @classmethod
    def at(cls, root: Path) -> WorkspaceLayout:
        resolved = root.expanduser().resolve()
        return cls(
            root=resolved,
            downloads=resolved / "downloads",
            extracted=resolved / "extracted",
            prepared_audio=resolved / "prepared" / "audio",
            inventories=resolved / "inventories",
            state=resolved / "state",
            manifests=resolved / "manifests",
            runs=resolved / "runs",
        )

    def create(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for path in (
            self.downloads,
            self.extracted,
            self.prepared_audio,
            self.inventories,
            self.state,
            self.manifests,
            self.runs,
        ):
            managed = _workspace_descendant(
                path,
                self.root,
                label=f"managed workspace directory {path.name!r}",
            )
            managed.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class PrepareResult:
    edition: str
    selection: str
    run_id: str
    workspace: Path
    manifests: dict[str, Path]
    counts: dict[str, int]
    resumed_datasets: tuple[str, ...]


ProgressCallback = Callable[[str], None]
SOURCE_ARTIFACT_FORMAT_VERSION = 1
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@contextmanager
def _workspace_prepare_lock(path: Path) -> Iterator[None]:
    """Allow only one preparation process to mutate a workspace at a time."""

    descriptor = os.open(
        path,
        os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
        0o600,
    )
    with os.fdopen(descriptor, "a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another preparation is already using this workspace: {path}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def prepare_edition(
    edition: str | Path,
    *,
    selection: SelectionMode | str = SelectionMode.CURATED,
    workspace: Path,
    repo_root: Path | None = None,
    workers: int = 1,
    progress: ProgressCallback | None = None,
) -> PrepareResult:
    """Fetch and materialize every configured source for one edition."""

    if workers <= 0:
        raise ValueError("workers must be positive")
    layout = WorkspaceLayout.at(workspace)
    layout.create()
    with _workspace_prepare_lock(layout.state / "prepare.lock"):
        return _prepare_edition_unlocked(
            edition,
            selection=selection,
            workspace=workspace,
            repo_root=repo_root,
            workers=workers,
            progress=progress,
        )


def _prepare_edition_unlocked(
    edition: str | Path,
    *,
    selection: SelectionMode | str,
    workspace: Path,
    repo_root: Path | None,
    workers: int,
    progress: ProgressCallback | None,
) -> PrepareResult:
    """Implementation guarded by the workspace-wide prepare lock."""

    emit = progress or (lambda _message: None)
    prepare_started = time.monotonic()
    mode = SelectionMode(selection)
    loaded = load_edition_config(edition, repo_root=repo_root, selection=mode)
    _require_preparation_requirements(loaded)
    layout = WorkspaceLayout.at(workspace)

    code_fingerprint = _implementation_fingerprint(loaded.repo_root)
    dependency_lock_digest = _optional_digest(loaded.repo_root / "uv.lock")
    environment = environment_provenance(loaded.repo_root)
    tool_versions = {name: environment[name] for name in ("python", "ffmpeg", "git", "zip")}
    audio_implementation_fingerprint = _audio_implementation_fingerprint(
        loaded.repo_root, environment["ffmpeg"]
    )
    config_payload = _portable_config_payload(loaded)
    local_source_inputs = {
        dataset.id: _local_source_fingerprints(dataset.sources)
        for dataset in loaded.config.datasets
    }
    run_seed_fingerprint = fingerprint(
        {
            "config": config_payload,
            "selection": mode.value,
            "local_source_inputs": local_source_inputs,
            "implementation": code_fingerprint,
            "dependency_lock": dependency_lock_digest,
            "tool_versions": tool_versions,
            "version": __version__,
        }
    )
    provisional_run_id = f"{loaded.config.edition}-{mode.value}-pending-{run_seed_fingerprint[:12]}"
    run_dir = layout.runs / provisional_run_id
    staging_dir = run_dir / "manifests"
    shared_state = StateStore(layout.state / "datasets")
    run_state = StateStore(run_dir / "state")
    run_started = run_state.mark_running("prepare", run_seed_fingerprint)
    resolved_config_path = run_dir / "resolved-config.json"
    atomic_write_text(resolved_config_path, f"{canonical_json(config_payload)}\n")

    inventory: list[InventoryItem] = []
    resumed: list[str] = []
    inventory_digests: dict[str, str] = {}
    inventory_counts: dict[str, dict[str, int]] = {}
    source_artifacts: dict[str, list[dict[str, str]]] = {}
    dataset_timings: dict[str, float] = {}
    for dataset in loaded.config.datasets:
        dataset_started = time.monotonic()
        emit(f"Dataset {dataset.id}: checking inventory")
        source_digest_cache = _workspace_descendant(
            layout.state / "source-checksums" / f"{dataset.id}.json",
            layout.state,
            label=f"source checksum cache for dataset {dataset.id!r}",
        )
        inventory_path = layout.inventories / f"{dataset.id}.jsonl"
        sources_path = layout.inventories / f"{dataset.id}.sources.json"
        stage_key = f"inventory-{dataset.id}"
        inventory_implementation_fingerprint = _inventory_implementation_fingerprint(
            loaded.repo_root, dataset.adapter
        )
        stage_fingerprint = fingerprint(
            {
                "dataset": dataset.model_dump(mode="json", exclude_none=True),
                "local_sources": local_source_inputs[dataset.id],
                "implementation": inventory_implementation_fingerprint,
            }
        )
        reusable = (
            shared_state.is_complete(
                stage_key,
                stage_fingerprint,
                verify_paths=(inventory_path, sources_path),
            )
            and inventory_path.is_file()
            and sources_path.is_file()
        )
        artifact_data = _read_source_artifact_records(sources_path) if reusable else None
        reusable = (
            reusable
            and artifact_data is not None
            and _source_artifacts_match(
                artifact_data,
                repo_root=loaded.repo_root,
                workspace=layout.root,
                workers=workers,
            )
        )
        items = list(read_jsonl(inventory_path, InventoryItem)) if reusable else []
        reusable = reusable and _inventory_sources_are_valid(
            items,
            workers=workers,
            cache_path=source_digest_cache,
        )
        if reusable:
            resumed.append(dataset.id)
            emit(f"Dataset {dataset.id}: reusing inventory ({len(items)} items)")
        else:
            started = shared_state.mark_running(stage_key, stage_fingerprint)
            try:
                adapter = create_adapter(
                    dataset,
                    loaded.repo_root,
                    layout.root,
                    workers=workers,
                )
                extracted_dir = _workspace_descendant(
                    adapter.extracted_dir,
                    layout.extracted,
                    label=f"dataset {dataset.id!r} extraction directory",
                )
                if extracted_dir.exists():
                    shutil.rmtree(extracted_dir)
                adapter.fetch()
                items = adapter.inventory()
                if any(item.dataset != dataset.id for item in items):
                    raise ValueError(
                        f"adapter {dataset.adapter!r} returned an item for another dataset"
                    )
                items = _attach_source_checksums(
                    items,
                    workers=workers,
                    cache_path=source_digest_cache,
                )
                _validate_inventory_completeness(dataset, items)
                write_jsonl(inventory_path, items)
                artifact_paths = _dataset_artifact_paths(adapter.download_dir, dataset.sources)
                artifact_records = _artifact_records(
                    artifact_paths,
                    repo_root=loaded.repo_root,
                    workspace=layout.root,
                    workers=workers,
                )
                atomic_write_text(sources_path, f"{canonical_json(artifact_records)}\n")
                artifact_data = artifact_records
                shared_state.mark_complete(
                    stage_key,
                    stage_fingerprint,
                    [inventory_path, sources_path],
                    started_at=started.started_at,
                )
            except BaseException as error:
                shared_state.mark_failed(
                    stage_key,
                    stage_fingerprint,
                    error,
                    started_at=started.started_at,
                )
                raise
            emit(f"Dataset {dataset.id}: prepared inventory ({len(items)} items)")
        inventory.extend(items)
        inventory_digests[dataset.id] = _inventory_digest(
            items,
            repo_root=loaded.repo_root,
            workspace=layout.root,
        )
        dataset_counts = Counter(item.media_kind.value for item in items)
        inventory_counts[dataset.id] = dict(sorted(dataset_counts.items()))
        if artifact_data is None:
            artifact_data = _read_source_artifact_records(sources_path)
        if artifact_data is None:
            raise ValueError(f"invalid source artifact manifest: {sources_path}")
        source_artifacts[dataset.id] = artifact_data
        dataset_timings[dataset.id] = time.monotonic() - dataset_started

    selection_started = time.monotonic()
    selected = select_inventory(
        inventory,
        selection=mode,
        exclusions=loaded.config.exclusions,
        curations=loaded.config.curations,
    )
    emit(
        "Selection complete: "
        f"{len(selected.training)} training, {len(selected.validation)} validation, "
        f"{len(selected.evaluation)} evaluation"
    )
    selected_source_digest = _selected_source_digest(selected)
    selection_seconds = time.monotonic() - selection_started
    emit("Selected source checksums complete")
    materialization_started = time.monotonic()
    emit(f"Materializing audio with {workers} workers")
    staged, public_evaluation_count, known_audio = _materialize_selection(
        selected,
        loaded=loaded,
        layout=layout,
        staging_dir=staging_dir,
        workers=workers,
        implementation_fingerprint=audio_implementation_fingerprint,
    )
    materialization_seconds = time.monotonic() - materialization_started
    emit("Audio materialization complete")
    if loaded.config.public_evaluation is not None:
        public_spec = loaded.config.public_evaluation
        public_inventory_path = layout.inventories / f"{public_spec.id}.jsonl"
        public_items = read_jsonl(public_inventory_path, InventoryItem)
        inventory_digests[public_spec.id] = _inventory_digest(
            public_items,
            repo_root=loaded.repo_root,
            workspace=layout.root,
        )
        public_counts = Counter(item.media_kind.value for item in public_items)
        inventory_counts[public_spec.id] = dict(sorted(public_counts.items()))
        source_artifacts[public_spec.id] = [
            {
                "path": f"git:{public_spec.repository_url}",
                "revision": public_spec.revision,
            }
        ]

    input_fingerprint = fingerprint(
        {
            "inventories": inventory_digests,
            "source_artifacts": source_artifacts,
            "selected_sources": selected_source_digest,
        }
    )
    run_fingerprint = fingerprint({"run_seed": run_seed_fingerprint, "inputs": input_fingerprint})
    run_id = f"{loaded.config.edition}-{mode.value}-{run_fingerprint[:12]}"
    run_dir, staged = _finalize_run_directory(
        run_dir,
        destination=layout.runs / run_id,
        staged=staged,
        resolved_config=config_payload,
    )
    resolved_config_path = run_dir / "resolved-config.json"
    run_state = StateStore(run_dir / "state")
    run_state.mark_running("prepare", run_fingerprint)

    validation_started = time.monotonic()
    validation = validate_manifests(
        list(staged.values()),
        workspace=layout.root,
        verify_checksums=True,
        known_audio=known_audio,
        workers=workers,
    )
    if not validation.ok:
        details = "\n".join(validation.errors[:20])
        raise RuntimeError(f"prepared manifests failed validation:\n{details}")
    validation_seconds = time.monotonic() - validation_started
    emit("Manifest validation complete")

    counts = dict(selected.counts)
    if public_evaluation_count:
        counts["open_evaluation"] = public_evaluation_count
    run_metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "edition": loaded.config.edition,
        "selection": mode.value,
        "selection_policy": mode.policy_name,
        "config_path": _portable_path(loaded.path, loaded.repo_root, layout.root),
        "config_fingerprint": fingerprint(config_payload),
        "implementation_fingerprint": code_fingerprint,
        "dependency_lock_digest": dependency_lock_digest,
        "input_fingerprint": input_fingerprint,
        "run_fingerprint": run_fingerprint,
        "counts": counts,
        "inventory_digests": inventory_digests,
        "selected_source_digest": selected_source_digest,
        "inventory_counts": inventory_counts,
        "source_artifacts": source_artifacts,
        "manifests": {
            split: {
                "path": (layout.manifests / loaded.config.edition / mode.value / f"{split}.jsonl")
                .relative_to(layout.root)
                .as_posix(),
                "sha256": sha256_file(path),
            }
            for split, path in sorted(staged.items())
        },
        "timings_seconds": {
            "datasets": {key: round(value, 3) for key, value in sorted(dataset_timings.items())},
            "selection_and_source_hashing": round(selection_seconds, 3),
            "materialization": round(materialization_seconds, 3),
            "validation": round(validation_seconds, 3),
            "total": round(time.monotonic() - prepare_started, 3),
        },
        "environment": environment,
    }
    run_metadata_path = run_dir / "run.json"
    atomic_write_text(run_metadata_path, f"{canonical_json(run_metadata)}\n")
    published = _publish_manifest_set(
        staged,
        run_metadata,
        destination=layout.manifests / loaded.config.edition / mode.value,
        run_id=run_id,
    )
    run_state.mark_complete(
        "prepare",
        run_fingerprint,
        [run_metadata_path, resolved_config_path, *staged.values()],
        started_at=run_started.started_at,
    )
    return PrepareResult(
        edition=loaded.config.edition,
        selection=mode.value,
        run_id=run_id,
        workspace=layout.root,
        manifests=published,
        counts=counts,
        resumed_datasets=tuple(sorted(resumed)),
    )


def workspace_status(workspace: Path) -> list[dict[str, Any]]:
    layout = WorkspaceLayout.at(workspace)
    if not layout.runs.exists():
        return []
    reports: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in layout.runs.iterdir() if path.is_dir()):
        metadata_path = run_dir / "run.json"
        states = StateStore(run_dir / "state").all()
        prepare_state = next((stage for stage in states if stage.key == "prepare"), None)
        state_store = StateStore(run_dir / "state")
        reports.append(
            {
                "run_id": run_dir.name,
                "complete": metadata_path.is_file()
                and prepare_state is not None
                and state_store.is_complete("prepare", prepare_state.fingerprint),
                "stages": {stage.key: stage.status for stage in states},
                "metadata": (
                    json.loads(metadata_path.read_text(encoding="utf-8"))
                    if metadata_path.is_file()
                    else None
                ),
            }
        )
    return reports


def _materialize_selection(
    selected: SelectionResult,
    *,
    loaded: LoadedEdition,
    layout: WorkspaceLayout,
    staging_dir: Path,
    workers: int,
    implementation_fingerprint: str,
) -> tuple[dict[str, Path], int, dict[Path, AudioMetadata]]:
    partitions: dict[str, tuple[Split, tuple[InventoryItem, ...]]] = {
        Split.TRAIN.value: (Split.TRAIN, selected.training),
        Split.VALIDATION.value: (Split.VALIDATION, selected.validation),
        Split.EVALUATION.value: (Split.EVALUATION, selected.evaluation),
    }
    public_evaluation_count = 0
    if loaded.config.public_evaluation is not None:
        public_root = fetch_public_evaluation(loaded.config.public_evaluation, layout.root)
        public_items = tuple(
            _attach_source_checksums(
                inventory_public_evaluation(loaded.config.public_evaluation, public_root),
                workers=workers,
                cache_path=_workspace_descendant(
                    layout.state
                    / "source-checksums"
                    / f"{loaded.config.public_evaluation.id}.json",
                    layout.state,
                    label="public evaluation source checksum cache",
                ),
            )
        )
        write_jsonl(
            layout.inventories / f"{loaded.config.public_evaluation.id}.jsonl",
            public_items,
        )
        partitions["open-evaluation"] = (Split.EVALUATION, public_items)
        public_evaluation_count = len(public_items)

    all_items = tuple(item for _split, items in partitions.values() for item in items)
    tasks = [
        MaterializationTask(
            source=item.source_path,
            destination=_prepared_path(
                item,
                loaded=loaded,
                layout=layout,
                implementation_fingerprint=implementation_fingerprint,
            ),
            sample_rate_hz=loaded.config.audio.sample_rate_hz,
            channels=loaded.config.audio.channels,
            source_release=item.source_release,
            implementation_fingerprint=implementation_fingerprint,
            source_sha256=item.source_checksum,
        )
        for item in all_items
    ]
    metadata_by_path = {
        metadata.path: metadata for metadata in materialize_all(tasks, workers=workers)
    }

    manifests: dict[str, Path] = {}
    for name, (split, items) in partitions.items():
        records: list[ManifestItem] = []
        for item in items:
            destination = _prepared_path(
                item,
                loaded=loaded,
                layout=layout,
                implementation_fingerprint=implementation_fingerprint,
            )
            metadata = metadata_by_path[destination]
            records.append(
                ManifestItem.from_inventory(
                    item,
                    audio_path=destination.relative_to(layout.root).as_posix(),
                    split=split,
                    sample_rate_hz=metadata.sample_rate_hz,
                    channels=metadata.channels,
                    frame_count=metadata.num_frames,
                    checksum=metadata.sha256,
                )
            )
        path = staging_dir / f"{name}.jsonl"
        write_manifest(path, records)
        manifests[name] = path
    return manifests, public_evaluation_count, metadata_by_path


def _prepared_path(
    item: InventoryItem,
    *,
    loaded: LoadedEdition,
    layout: WorkspaceLayout,
    implementation_fingerprint: str,
) -> Path:
    if item.source_checksum is None:
        raise ValueError(f"inventory item {item.id!r} has no source checksum")
    key = materialization_fingerprint(
        source_sha256=item.source_checksum,
        source_release=item.source_release,
        sample_rate_hz=loaded.config.audio.sample_rate_hz,
        channels=loaded.config.audio.channels,
        implementation_fingerprint=implementation_fingerprint,
    )
    return _workspace_descendant(
        output_path(
            layout.prepared_audio,
            item.dataset,
            item.source_id,
            materialization_key=key,
        ),
        layout.prepared_audio,
        label=f"prepared audio path for {item.id!r}",
    )


def _implementation_fingerprint(repo_root: Path) -> str:
    source_root = _source_root(repo_root)
    payload = [
        (path.relative_to(source_root).as_posix(), sha256_file(path))
        for path in sorted(source_root.rglob("*.py"))
    ]
    return fingerprint(payload)


def _inventory_implementation_fingerprint(repo_root: Path, adapter: str) -> str:
    """Fingerprint code that can change normalized source inventory semantics."""

    component_paths = [
        "models.py",
        "manifests.py",
        "datasets/base.py",
        "datasets/common.py",
        "datasets/inventory.py",
        f"datasets/{adapter}.py",
    ]
    return fingerprint(
        {
            "schema_version": 1,
            "extraction_format": EXTRACTION_FORMAT_VERSION,
            "source_artifact_format": SOURCE_ARTIFACT_FORMAT_VERSION,
            "pipeline": _callable_fingerprint(
                _portable_path,
                _dataset_artifact_paths,
                _inventory_digest,
                _validate_inventory_completeness,
            ),
            "code": _component_fingerprint(repo_root, component_paths),
        }
    )


def _audio_implementation_fingerprint(_repo_root: Path, ffmpeg_version: Any) -> str:
    """Fingerprint only code and tooling that can change prepared WAV bytes."""

    return fingerprint(
        {
            "schema_version": 1,
            "conversion": _callable_fingerprint(
                audio_module.probe,
                audio_module._ffmpeg_command,
                audio_module._needs_explicit_channel_average,
                audio_module._materialize_with_ffmpeg,
                audio_module._materialize_with_soundfile,
            ),
            "ffmpeg": ffmpeg_version,
        }
    )


def _component_fingerprint(repo_root: Path, relative_paths: Iterable[str]) -> str:
    source_root = _source_root(repo_root)
    payload = []
    for relative_path in sorted(set(relative_paths)):
        path = source_root / relative_path
        payload.append((relative_path, sha256_file(path) if path.is_file() else None))
    return fingerprint(payload)


def _callable_fingerprint(*functions: Any) -> str:
    return fingerprint(
        [(function.__qualname__, inspect.getsource(function)) for function in functions]
    )


def _source_root(repo_root: Path) -> Path:
    source_root = repo_root / "src" / "lrac_data"
    return source_root if source_root.is_dir() else Path(__file__).resolve().parent


def _workspace_descendant(path: Path, root: Path, *, label: str) -> Path:
    """Resolve a generated path and require it to remain below its workspace root."""

    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{label} escapes {resolved_root}: {resolved}") from error
    if relative == Path("."):
        raise ValueError(f"{label} must not be the workspace root: {resolved_root}")
    return resolved


def _optional_digest(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def _require_preparation_requirements(loaded: LoadedEdition) -> None:
    missing: list[str] = []
    for executable in ("ffmpeg", "zip"):
        if shutil.which(executable) is None:
            missing.append(f"executable {executable}")
    if loaded.config.public_evaluation is not None and shutil.which("git") is None:
        missing.append("executable git")
    for package in ("numpy", "pyarrow", "scipy", "soundfile"):
        if importlib.util.find_spec(package) is None:
            missing.append(f"Python package {package}")
    if missing:
        raise RuntimeError("missing preparation requirements: " + ", ".join(missing))


def _local_source_fingerprints(sources: Iterable[Any]) -> list[tuple[str, Any]]:
    fingerprints: list[tuple[str, Any]] = []
    for source in sources:
        path = source.path
        if path is None:
            continue
        if path.is_file():
            value: Any = sha256_file(path)
        elif path.is_dir():
            value = [
                (child.relative_to(path).as_posix(), sha256_file(child))
                for child in sorted(path.rglob("*"))
                if child.is_file()
            ]
        else:
            value = None
        fingerprints.append((source.name, value))
    return fingerprints


def _portable_config_payload(loaded: LoadedEdition) -> dict[str, Any]:
    payload = loaded.config.model_dump(mode="python", exclude_none=True)

    def normalize(value: Any) -> Any:
        if isinstance(value, Path):
            return _portable_path(value, loaded.repo_root, None)
        if isinstance(value, PurePath):
            return value.as_posix()
        if isinstance(value, dict):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return getattr(value, "value", value)

    normalized = normalize(payload)
    assert isinstance(normalized, dict)
    return normalized


def _portable_path(path: Path, repo_root: Path, workspace: Path | None) -> str:
    resolved = path.resolve()
    for prefix, root in (("repo", repo_root), ("workspace", workspace)):
        if root is None:
            continue
        try:
            relative = resolved.relative_to(root.resolve())
        except ValueError:
            continue
        return f"{prefix}:{relative.as_posix()}"
    return f"external:{resolved.name}"


def _dataset_artifact_paths(download_dir: Path, sources: Iterable[Any]) -> list[Path]:
    artifacts = [
        path
        for path in sorted(download_dir.rglob("*"))
        if path.is_file()
        and not path.name.endswith(".part")
        and not path.name.endswith(".download.json")
        and not path.name.endswith(".inputs.json")
        and not path.with_suffix(f"{path.suffix}.inputs.json").is_file()
    ]
    for source in sources:
        path = source.path
        if path is None:
            continue
        if path.is_file():
            artifacts.append(path)
        elif path.is_dir():
            artifacts.extend(child for child in sorted(path.rglob("*")) if child.is_file())
    return list(dict.fromkeys(path.resolve() for path in artifacts))


def _artifact_records(
    artifacts: Iterable[Path],
    *,
    repo_root: Path,
    workspace: Path,
    workers: int,
) -> list[dict[str, str]]:
    resolved_paths = [path.resolve() for path in artifacts]
    hashes = _bounded_thread_map(
        trusted_file_sha256,
        resolved_paths,
        workers=workers,
        thread_name="lrac-artifact-hash",
    )
    records: list[dict[str, str]] = []
    for resolved, digest in zip(resolved_paths, hashes, strict=True):
        records.append(
            {
                "path": _portable_path(resolved, repo_root, workspace),
                "sha256": digest,
            }
        )
    return records


def _read_source_artifact_records(path: Path) -> list[dict[str, str]] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(value, list):
        return None
    records: list[dict[str, str]] = []
    for record in value:
        if not isinstance(record, dict):
            return None
        portable_path = record.get("path")
        digest = record.get("sha256")
        if (
            not isinstance(portable_path, str)
            or not isinstance(digest, str)
            or not _is_sha256(digest)
        ):
            return None
        records.append({"path": portable_path, "sha256": digest})
    return records


def _source_artifacts_match(
    records: Iterable[dict[str, str]],
    *,
    repo_root: Path,
    workspace: Path,
    workers: int,
) -> bool:
    expected: list[tuple[Path, str]] = []
    for record in records:
        portable = record["path"]
        if portable.startswith("external:"):
            # External configured inputs are already covered by local-source fingerprints.
            continue
        prefix, separator, relative = portable.partition(":")
        root = {"repo": repo_root, "workspace": workspace}.get(prefix)
        if not separator or root is None:
            return False
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            return False
        expected.append((candidate, record["sha256"]))

    def digest_or_none(path: Path) -> str | None:
        try:
            return trusted_file_sha256(path) if path.is_file() else None
        except OSError:
            return None

    actual = _bounded_thread_map(
        digest_or_none,
        [path for path, _digest in expected],
        workers=workers,
        thread_name="lrac-artifact-verify",
    )
    return all(
        digest == expected_digest
        for (_path, expected_digest), digest in zip(expected, actual, strict=True)
    )


def _inventory_digest(
    items: Iterable[InventoryItem],
    *,
    repo_root: Path,
    workspace: Path,
) -> str:
    records: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda candidate: candidate.id):
        record = item.model_dump(mode="json", exclude_none=True)
        record["source_path"] = _portable_path(item.source_path, repo_root, workspace)
        records.append(record)
    return fingerprint(records)


def _validate_inventory_completeness(dataset: Any, items: list[InventoryItem]) -> None:
    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"dataset {dataset.id!r} inventory contains duplicate IDs")
    counts = Counter(item.media_kind for item in items)
    missing = [kind.value for kind in dataset.media_kinds if counts[kind] == 0]
    if missing:
        raise ValueError(f"dataset {dataset.id!r} inventory has no items for: {', '.join(missing)}")
    mismatches = [
        f"{kind.value}: expected {expected}, found {counts[kind]}"
        for kind, expected in dataset.expected_inventory.items()
        if counts[kind] != expected
    ]
    if mismatches:
        raise ValueError(
            f"dataset {dataset.id!r} inventory count mismatch: " + "; ".join(mismatches)
        )


def _selected_source_digest(selected: SelectionResult) -> str:
    items = (*selected.training, *selected.validation, *selected.evaluation)
    return fingerprint(
        [(item.id, item.source_checksum) for item in sorted(items, key=lambda item: item.id)]
    )


def _attach_source_checksums(
    items: Iterable[InventoryItem],
    *,
    workers: int,
    cache_path: Path,
) -> list[InventoryItem]:
    materialized = list(items)
    unique_paths = list(dict.fromkeys(item.source_path.resolve() for item in materialized))
    identities = _path_identities(unique_paths, workers=workers)
    missing = [
        path for path, identity in zip(unique_paths, identities, strict=True) if identity is None
    ]
    if missing:
        raise FileNotFoundError(missing[0])
    stored_cache = _read_source_digest_cache(cache_path)
    active_keys = {str(path) for path in unique_paths}
    cache = {key: value for key, value in stored_cache.items() if key in active_keys}
    cache_changed = len(cache) != len(stored_cache)
    digest_by_path: dict[Path, str] = {}
    misses: list[Path] = []
    for path, identity in zip(unique_paths, identities, strict=True):
        assert identity is not None
        record = cache.get(str(path))
        cached_digest = _cached_source_digest(record, identity)
        if cached_digest is not None:
            digest_by_path[path] = cached_digest
        else:
            misses.append(path)
    if misses:
        hashes = _bounded_thread_map(
            _stable_source_digest,
            misses,
            workers=workers,
            thread_name="lrac-source-hash",
        )
        for path, (digest, identity) in zip(misses, hashes, strict=True):
            digest_by_path[path] = digest
            cache[str(path)] = _source_cache_record(digest, identity)
    if misses or cache_changed:
        atomic_write_text(cache_path, f"{canonical_json(cache)}\n")
    return [
        item.model_copy(update={"source_checksum": digest_by_path[item.source_path.resolve()]})
        for item in materialized
    ]


def _inventory_sources_are_valid(
    items: Iterable[InventoryItem],
    *,
    workers: int,
    cache_path: Path,
) -> bool:
    expected_by_path: dict[Path, str] = {}
    for item in items:
        if item.source_checksum is None:
            return False
        path = item.source_path.resolve()
        previous = expected_by_path.setdefault(path, item.source_checksum)
        if previous != item.source_checksum:
            return False
    paths = list(expected_by_path)
    identities = _path_identities(paths, workers=workers)
    if any(identity is None for identity in identities):
        return False

    stored_cache = _read_source_digest_cache(cache_path)
    active_keys = {str(path) for path in paths}
    cache = {key: value for key, value in stored_cache.items() if key in active_keys}
    cache_changed = len(cache) != len(stored_cache)
    recovery_paths: list[Path] = []
    for path, identity in zip(paths, identities, strict=True):
        assert identity is not None
        if _cached_source_digest(cache.get(str(path)), identity) != expected_by_path[path]:
            recovery_paths.append(path)

    if recovery_paths:
        recovered = _bounded_thread_map(
            _stable_source_digest,
            recovery_paths,
            workers=workers,
            thread_name="lrac-source-recovery",
        )
        for path, (digest, identity) in zip(recovery_paths, recovered, strict=True):
            if digest != expected_by_path[path]:
                return False
            cache[str(path)] = _source_cache_record(digest, identity)

    if recovery_paths or cache_changed:
        atomic_write_text(cache_path, f"{canonical_json(cache)}\n")
    return True


def _cached_source_digest(record: Any, identity: FileIdentity) -> str | None:
    if not isinstance(record, dict):
        return None
    digest = record.get("sha256")
    cached_identity = FileIdentity.from_dict(record.get("identity"))
    if cached_identity is None:
        cached_identity = FileIdentity.from_dict(record)
    if cached_identity != identity or not _is_sha256(digest):
        return None
    assert isinstance(digest, str)
    return digest


def _stable_source_digest(path: Path) -> tuple[str, FileIdentity]:
    before = FileIdentity.from_stat(path.stat())
    digest = sha256_file(path)
    after = FileIdentity.from_stat(path.stat())
    if before != after:
        raise RuntimeError(f"source changed while hashing: {path}")
    return digest, after


def _source_cache_record(digest: str, identity: FileIdentity) -> dict[str, Any]:
    return {"identity": identity.as_dict(), "sha256": digest}


def _path_identities(paths: Iterable[Path], *, workers: int) -> list[FileIdentity | None]:
    def identity_or_none(path: Path) -> FileIdentity | None:
        try:
            return FileIdentity.from_stat(path.stat())
        except OSError:
            return None

    return _bounded_thread_map(
        identity_or_none,
        list(paths),
        workers=workers,
        thread_name="lrac-source-stat",
    )


def _bounded_thread_map(
    function: Callable[[InputT], OutputT],
    items: list[InputT],
    *,
    workers: int,
    thread_name: str,
) -> list[OutputT]:
    if not items:
        return []
    chunk_size = max(workers * 8, 1)
    results: list[OutputT] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=thread_name) as pool:
        for start in range(0, len(items), chunk_size):
            results.extend(pool.map(function, items[start : start + chunk_size]))
    return results


def _read_source_digest_cache(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _publish_manifest_set(
    staged: dict[str, Path],
    run_metadata: dict[str, Any],
    *,
    destination: Path,
    run_id: str,
) -> dict[str, Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup_pattern = f".{destination.name}.*.previous"
    existing_backups = sorted(destination.parent.glob(backup_pattern))
    if not destination.exists() and existing_backups:
        newest = max(existing_backups, key=lambda path: path.stat().st_mtime_ns)
        newest.replace(destination)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.{run_id}.", dir=destination.parent)
    )
    backup = destination.parent / f".{destination.name}.{run_id}.{uuid.uuid4().hex}.previous"
    try:
        for split, path in sorted(staged.items()):
            atomic_write_text(temporary / f"{split}.jsonl", path.read_text(encoding="utf-8"))
        atomic_write_text(temporary / "run.json", f"{canonical_json(run_metadata)}\n")
        if destination.exists():
            destination.replace(backup)
        try:
            temporary.replace(destination)
        except BaseException:
            if backup.exists() and not destination.exists():
                backup.replace(destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        for stale_backup in destination.parent.glob(backup_pattern):
            shutil.rmtree(stale_backup)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return {split: destination / f"{split}.jsonl" for split in staged}


def _finalize_run_directory(
    provisional: Path,
    *,
    destination: Path,
    staged: dict[str, Path],
    resolved_config: dict[str, Any],
) -> tuple[Path, dict[str, Path]]:
    if provisional != destination:
        if destination.exists():
            final_staging = destination / "manifests"
            for split, source in sorted(staged.items()):
                atomic_write_text(
                    final_staging / f"{split}.jsonl",
                    source.read_text(encoding="utf-8"),
                )
            shutil.rmtree(provisional)
        else:
            provisional.replace(destination)
    atomic_write_text(
        destination / "resolved-config.json",
        f"{canonical_json(resolved_config)}\n",
    )
    return destination, {split: destination / "manifests" / f"{split}.jsonl" for split in staged}
