# Copyright 2026 Cisco Systems, Inc. and its affiliates
# Apache-2.0

"""Complete, resumable materialization of an LRAC challenge edition."""

from __future__ import annotations

import fcntl
import hashlib
import inspect
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from heapq import merge
from itertools import groupby
from pathlib import Path
from typing import Any, TypeVar

from .audio import (
    AudioMetadata,
    MaterializationTask,
    materialize_all,
    output_path,
    refresh_materialization_cache,
)
from .config import LoadedEdition, load_edition_config, portable_config_payload
from .datasets import create_adapter
from .datasets.io import trusted_file_sha256, verify_checksum
from .manifests import read_jsonl, write_jsonl, write_ordered_manifest
from .models import (
    InventoryItem,
    ManifestItem,
    PreparationRunMetadata,
    SelectionMode,
    SelectionResult,
    Split,
)
from .planner import collect_preparation_readiness
from .public_evaluation import (
    fetch_public_evaluation,
    inventory_kaldi_testsets,
    inventory_public_evaluation,
)
from .release import preflight_release_output, publish_release
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


@dataclass(frozen=True)
class WorkspaceLayout:
    root: Path
    downloads: Path
    extracted: Path
    prepared_audio: Path
    inventories: Path
    state: Path
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
    release: Path
    manifests: dict[str, Path]
    counts: dict[str, int]
    resumed_datasets: tuple[str, ...]


ProgressCallback = Callable[[str], None]
_MATERIALIZATION_CHUNK_SIZE = 4096
_FMA_RECOVERED_FRAME_WARNING = re.compile(
    rb"\[src/libmpg123/layer3\.c:INT123_do_layer3\(\):\d+\] error: dequantization failed!"
)
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class _IncompleteInventoryError(ValueError):
    """An inventory is structurally valid but appears incompletely extracted."""


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
    output: Path,
    repo_root: Path | None = None,
    workers: int = 8,
    low_storage: bool = False,
    progress: ProgressCallback | None = None,
) -> PrepareResult:
    """Prepare and publish the complete configured release."""

    if workers < 1:
        raise ValueError("workers must be positive")
    mode = SelectionMode(selection)
    loaded = load_edition_config(edition, repo_root=repo_root, selection=mode)
    layout = WorkspaceLayout.at(workspace)
    _require_preparation_requirements(loaded, layout.root)
    layout.create()
    preflight_release_output(output, layout.root)
    with _workspace_prepare_lock(layout.state / "prepare.lock"):
        return _prepare_edition_unlocked(
            loaded,
            mode=mode,
            layout=layout,
            output=output,
            workers=workers,
            low_storage=low_storage,
            progress=progress,
        )


def _prepare_edition_unlocked(
    loaded: LoadedEdition,
    *,
    mode: SelectionMode,
    layout: WorkspaceLayout,
    output: Path,
    workers: int,
    low_storage: bool,
    progress: ProgressCallback | None,
) -> PrepareResult:
    """Implementation guarded by the workspace-wide prepare lock."""

    emit = progress or (lambda _message: None)
    prepare_started = time.monotonic()

    code_fingerprint = _implementation_fingerprint(loaded.repo_root)
    lock_path = loaded.repo_root / "uv.lock"
    dependency_lock_digest = sha256_file(lock_path) if lock_path.is_file() else None
    environment = environment_provenance(loaded.repo_root)
    requires_zip = any(dataset.id == "fsd50k" for dataset in loaded.config.datasets)
    execution_names = ["python", "git", "packages"]
    if requires_zip:
        execution_names.append("zip")
    execution_identity = {name: environment[name] for name in execution_names}
    audio_implementation_fingerprint = _audio_implementation_fingerprint(
        loaded.repo_root, environment
    )
    config_payload = portable_config_payload(loaded)
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
            "execution_identity": execution_identity,
        }
    )
    shared_state = StateStore(layout.state / "datasets")

    selected_training: list[InventoryItem] = []
    selected_validation: list[InventoryItem] = []
    selected_evaluation: list[InventoryItem] = []
    selection_counts: Counter[str] = Counter()
    selected_source_hash = hashlib.sha256()
    selection_seconds = 0.0
    materialization_seconds = 0.0
    known_audio: dict[Path, AudioMetadata] = {}
    resumed: list[str] = []
    inventory_digests: dict[str, str] = {}
    inventory_counts: dict[str, dict[str, int]] = {}
    source_artifacts: dict[str, list[dict[str, str]]] = {}
    dataset_timings: dict[str, float] = {}
    for dataset in sorted(loaded.config.datasets, key=lambda value: value.id):
        dataset_started = time.monotonic()
        emit(f"Dataset {dataset.id}: checking inventory")
        adapter = create_adapter(
            dataset,
            layout.root,
            workers=workers,
        )
        inventory_path = layout.inventories / f"{dataset.id}.jsonl"
        sources_path = layout.inventories / f"{dataset.id}.sources.json"
        stage_key = f"inventory-{dataset.id}"
        adapter_source = inspect.getsourcefile(type(adapter))
        if adapter_source is None:
            raise RuntimeError(f"cannot locate implementation for dataset {dataset.id!r}")
        inventory_implementation_fingerprint = _inventory_implementation_fingerprint(
            loaded.repo_root,
            dataset.id,
            environment,
            Path(adapter_source),
        )
        stage_fingerprint = fingerprint(
            {
                "dataset": dataset.model_dump(mode="json", exclude_none=True),
                "local_sources": local_source_inputs[dataset.id],
                "implementation": inventory_implementation_fingerprint,
            }
        )
        reusable = shared_state.is_complete(
            stage_key,
            stage_fingerprint,
            verify_paths=(inventory_path, sources_path),
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
                allow_missing_workspace=low_storage,
            )
        )
        items = list(read_jsonl(inventory_path, InventoryItem)) if reusable else []
        inventory_reused = reusable
        if reusable:
            resumed.append(dataset.id)
            emit(f"Dataset {dataset.id}: reusing inventory ({len(items)} items)")
        else:
            previous = shared_state.read(stage_key)
            started = shared_state.mark_running(stage_key, stage_fingerprint)
            try:
                extracted_dir = _workspace_descendant(
                    adapter.extracted_dir,
                    layout.extracted,
                    label=f"dataset {dataset.id!r} extraction directory",
                )
                repair_incomplete = (
                    extracted_dir.exists()
                    and previous is not None
                    and previous.status != "complete"
                    and previous.fingerprint == stage_fingerprint
                )
                if extracted_dir.exists() and (
                    previous is None
                    or previous.status == "complete"
                    or previous.fingerprint != stage_fingerprint
                ):
                    adapter.clear_extracted()
                _verify_local_sources(dataset.sources, workers=workers)
                try:
                    adapter.fetch()
                    items = adapter.inventory()
                    _validate_inventory_completeness(dataset, items)
                except (FileNotFoundError, _IncompleteInventoryError):
                    if not repair_incomplete:
                        raise
                    emit(f"Dataset {dataset.id}: repairing incomplete extraction")
                    adapter.clear_extracted()
                    adapter.fetch()
                    items = adapter.inventory()
                    _validate_inventory_completeness(dataset, items)
                if any(item.dataset != dataset.id for item in items):
                    raise ValueError(f"dataset {dataset.id!r} returned an item for another dataset")
                write_jsonl(inventory_path, items)
                artifact_records = _artifact_records(
                    adapter.provenance_artifacts(),
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
        items.sort(key=lambda item: item.id)
        inventory_digests[dataset.id] = _inventory_digest(
            items,
            repo_root=loaded.repo_root,
            workspace=layout.root,
        )
        dataset_counts = Counter(item.media_kind.value for item in items)
        inventory_counts[dataset.id] = dict(sorted(dataset_counts.items()))
        assert artifact_data is not None
        source_artifacts[dataset.id] = artifact_data
        dataset_timings[dataset.id] = time.monotonic() - dataset_started
        selection_started = time.monotonic()
        dataset_selection = _select_dataset_inventory(
            items,
            dataset_id=dataset.id,
            loaded=loaded,
            mode=mode,
        )
        selected_training.extend(dataset_selection.training)
        selected_validation.extend(dataset_selection.validation)
        selected_evaluation.extend(dataset_selection.evaluation)
        selection_counts.update(dataset_selection.counts)
        dataset_items = tuple(
            merge(
                dataset_selection.training,
                dataset_selection.validation,
                dataset_selection.evaluation,
                key=lambda item: item.id,
            )
        )
        adapter.fetch_selected(dataset_items)
        if inventory_reused and dataset_items:
            refetched_artifacts = _artifact_records(
                adapter.provenance_artifacts(),
                repo_root=loaded.repo_root,
                workspace=layout.root,
                workers=workers,
            )
            if refetched_artifacts != source_artifacts[dataset.id]:
                raise RuntimeError(
                    f"dataset {dataset.id!r} source artifacts changed between "
                    "inventory and materialization"
                )
        selection_seconds += time.monotonic() - selection_started

        materialization_started = time.monotonic()
        _materialize_items(
            dataset_items,
            loaded=loaded,
            layout=layout,
            workers=workers,
            implementation_fingerprint=audio_implementation_fingerprint,
            metadata_by_path=known_audio,
            source_digest=selected_source_hash,
        )
        materialization_seconds += time.monotonic() - materialization_started
        if low_storage:
            adapter.clear_extracted()
            adapter.clear_downloads()
            emit(f"Dataset {dataset.id}: selected, materialized, and cleared source caches")
        else:
            emit(f"Dataset {dataset.id}: selected and materialized")

    selected = SelectionResult(
        training=tuple(selected_training),
        validation=tuple(selected_validation),
        evaluation=tuple(selected_evaluation),
    )
    counts: dict[str, int | None] = dict(selection_counts)
    selected_source_digest = selected_source_hash.hexdigest()
    del items, selected_training, selected_validation, selected_evaluation
    emit(
        "Selection complete: "
        f"{len(selected.training)} training, {len(selected.validation)} validation, "
        f"{len(selected.evaluation)} evaluation"
    )

    public_items: tuple[InventoryItem, ...] = ()
    test_items: dict[str, tuple[InventoryItem, ...]] = {}
    if loaded.config.public_evaluation is not None:
        public_spec = loaded.config.public_evaluation
        public_root = fetch_public_evaluation(public_spec, layout.root)
        public_items = tuple(
            sorted(
                inventory_public_evaluation(public_spec, public_root),
                key=lambda item: item.id,
            )
        )
        test_items = inventory_kaldi_testsets(public_spec, public_root)
        write_jsonl(layout.inventories / f"{public_spec.id}.jsonl", public_items)
        evaluation_items = tuple(
            merge(public_items, *test_items.values(), key=lambda item: item.id)
        )
        inventory_digests[public_spec.id] = _inventory_digest(
            evaluation_items,
            repo_root=loaded.repo_root,
            workspace=layout.root,
        )
        public_counts = Counter(item.media_kind.value for item in evaluation_items)
        inventory_counts[public_spec.id] = dict(sorted(public_counts.items()))
        for test_partition, test_partition_items in test_items.items():
            inventory_id = f"{public_spec.id}-{test_partition}"
            write_jsonl(layout.inventories / f"{inventory_id}.jsonl", test_partition_items)
        source_artifacts[public_spec.id] = [
            {
                "path": f"git:{public_spec.repository_url}",
                "revision": public_spec.revision,
            }
        ]
        if low_storage:
            materialization_started = time.monotonic()
            _materialize_items(
                evaluation_items,
                loaded=loaded,
                layout=layout,
                workers=workers,
                implementation_fingerprint=audio_implementation_fingerprint,
                metadata_by_path=known_audio,
            )
            materialization_seconds += time.monotonic() - materialization_started
            public_download = _workspace_descendant(
                layout.downloads / public_spec.id,
                layout.downloads,
                label="public evaluation download",
            )
            _remove_tree(public_download)

    input_fingerprint = fingerprint(
        {
            "inventories": inventory_digests,
            "source_artifacts": source_artifacts,
            "selected_sources": selected_source_digest,
        }
    )
    run_fingerprint = fingerprint({"run_seed": run_seed_fingerprint, "inputs": input_fingerprint})
    run_id = f"{loaded.config.edition}-{mode.value}-{run_fingerprint[:12]}"
    run_dir = _workspace_descendant(
        layout.runs / run_id,
        layout.runs,
        label="run directory",
    )
    staging_dir = run_dir / "manifests"
    staging_dir.mkdir(parents=True, exist_ok=True)
    run_state = StateStore(run_dir / "state")
    run_started = run_state.mark_running("prepare", run_fingerprint)
    resolved_config_path = run_dir / "resolved-config.json"
    atomic_write_text(resolved_config_path, f"{canonical_json(config_payload)}\n")

    manifest_started = time.monotonic()
    staged, known_audio = _materialize_selection(
        selected,
        public_items=(public_items if loaded.config.public_evaluation is not None else None),
        test_items=test_items,
        loaded=loaded,
        layout=layout,
        staging_dir=staging_dir,
        workers=workers,
        implementation_fingerprint=audio_implementation_fingerprint,
        known_audio=known_audio,
    )
    materialization_seconds += time.monotonic() - manifest_started
    emit("Audio materialization complete")

    counts["open_evaluation"] = (
        len(public_items) if loaded.config.public_evaluation is not None else None
    )
    counts.update({partition: len(items) for partition, items in test_items.items()})
    del selected, public_items, test_items
    run_metadata: PreparationRunMetadata = {
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
                "path": path.relative_to(layout.root).as_posix(),
                "sha256": sha256_file(path),
            }
            for split, path in sorted(staged.items())
        },
        "timings_seconds": {
            "datasets": {key: round(value, 3) for key, value in sorted(dataset_timings.items())},
            "selection_and_source_hashing": round(selection_seconds, 3),
            "materialization": round(materialization_seconds, 3),
            "total": round(time.monotonic() - prepare_started, 3),
        },
        "environment": environment,
    }
    run_metadata_path = run_dir / "run.json"
    atomic_write_text(run_metadata_path, f"{canonical_json(run_metadata)}\n")
    release = publish_release(
        output=output,
        workspace=layout.root,
        loaded=loaded,
        manifests=staged,
        run=run_metadata,
        known_audio=known_audio,
        workers=workers,
        progress=emit,
    )
    refresh_materialization_cache(layout.state / "audio.sqlite3", known_audio)
    run_state.mark_complete(
        "prepare",
        run_fingerprint,
        [run_metadata_path, resolved_config_path, *staged.values()],
        started_at=run_started.started_at,
        known_digests={
            path: run_metadata["manifests"][partition]["sha256"]
            for partition, path in staged.items()
        },
    )
    return PrepareResult(
        edition=loaded.config.edition,
        selection=mode.value,
        run_id=run_id,
        workspace=layout.root,
        release=release.output,
        manifests={
            partition: release.output / "manifests" / f"{partition}.jsonl" for partition in staged
        },
        counts={key: value for key, value in counts.items() if value is not None},
        resumed_datasets=tuple(sorted(resumed)),
    )


def workspace_status(workspace: Path) -> list[dict[str, Any]]:
    layout = WorkspaceLayout.at(workspace)
    reports: list[dict[str, Any]] = []
    if not layout.runs.exists():
        return reports
    for run_dir in sorted(path for path in layout.runs.iterdir() if path.is_dir()):
        metadata_path = run_dir / "run.json"
        state_store = StateStore(run_dir / "state")
        prepare_state = state_store.read("prepare")
        reports.append(
            {
                "run_id": run_dir.name,
                "complete": metadata_path.is_file()
                and prepare_state is not None
                and state_store.is_complete("prepare", prepare_state.fingerprint),
                "stages": (
                    {prepare_state.key: prepare_state.status} if prepare_state is not None else {}
                ),
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
    public_items: tuple[InventoryItem, ...] | None,
    test_items: dict[str, tuple[InventoryItem, ...]],
    loaded: LoadedEdition,
    layout: WorkspaceLayout,
    staging_dir: Path,
    workers: int,
    implementation_fingerprint: str,
    known_audio: dict[Path, AudioMetadata] | None = None,
) -> tuple[dict[str, Path], dict[Path, AudioMetadata]]:
    metadata_by_path = known_audio if known_audio is not None else {}
    partitions: dict[str, tuple[Split, tuple[InventoryItem, ...]]] = {
        Split.TRAIN.value: (Split.TRAIN, selected.training),
        Split.VALIDATION.value: (Split.VALIDATION, selected.validation),
        Split.EVALUATION.value: (Split.EVALUATION, selected.evaluation),
    }
    if public_items is not None:
        partitions["open-evaluation"] = (Split.EVALUATION, public_items)
    partitions.update(
        (partition, (Split.EVALUATION, items)) for partition, items in test_items.items()
    )

    manifests: dict[str, Path] = {}
    for name, (split, items) in partitions.items():
        _materialize_items(
            items,
            loaded=loaded,
            layout=layout,
            workers=workers,
            implementation_fingerprint=implementation_fingerprint,
            metadata_by_path=metadata_by_path,
        )
        path = staging_dir / f"{name}.jsonl"
        records = _manifest_records(
            items,
            split=split,
            loaded=loaded,
            layout=layout,
            implementation_fingerprint=implementation_fingerprint,
            metadata_by_path=metadata_by_path,
        )
        write_ordered_manifest(path, records)
        manifests[name] = path
    return manifests, metadata_by_path


def _select_dataset_inventory(
    items: Iterable[InventoryItem],
    *,
    dataset_id: str,
    loaded: LoadedEdition,
    mode: SelectionMode,
) -> SelectionResult:
    """Resolve one dataset's independently scoped edition policy."""

    return select_inventory(
        items,
        selection=mode,
        exclusions=(
            exclusion
            for exclusion in loaded.config.exclusions
            if exclusion.dataset == dataset_id
        ),
        curations=(
            curation for curation in loaded.config.curations if curation.dataset == dataset_id
        ),
    )


def _materialize_items(
    items: tuple[InventoryItem, ...],
    *,
    loaded: LoadedEdition,
    layout: WorkspaceLayout,
    workers: int,
    implementation_fingerprint: str,
    metadata_by_path: dict[Path, AudioMetadata],
    source_digest: Any | None = None,
) -> None:
    for start in range(0, len(items), _MATERIALIZATION_CHUNK_SIZE):
        chunk = items[start : start + _MATERIALIZATION_CHUNK_SIZE]
        destinations = [
            _prepared_path(
                item,
                loaded=loaded,
                layout=layout,
                implementation_fingerprint=implementation_fingerprint,
            )
            for item in chunk
        ]
        chunk_identities = (
            _source_identities(chunk, workers=workers)
            if source_digest is not None
            else None
        )
        if source_digest is not None:
            assert chunk_identities is not None
            _update_selected_source_digest(
                source_digest,
                chunk,
                source_identities=chunk_identities,
            )
        pending = [
            (item, destination)
            for item, destination in zip(chunk, destinations, strict=True)
            if destination not in metadata_by_path
        ]
        if not pending:
            continue
        pending_items = tuple(item for item, _destination in pending)
        if chunk_identities is None:
            chunk_identities = _source_identities(pending_items, workers=workers)
        for suppress_mpeg_warning, batch_iterator in groupby(
            pending,
            key=lambda pair: pair[0].dataset == "fma",
        ):
            batch = tuple(batch_iterator)
            tasks = [
                MaterializationTask(
                    source=item.source_path,
                    destination=destination,
                    sample_rate_hz=loaded.config.audio.sample_rate_hz,
                    channel_mode=loaded.config.audio.channels.for_media_kind(item.media_kind),
                    source_release=item.source_release,
                    implementation_fingerprint=implementation_fingerprint,
                    source_sha256=item.source_checksum,
                    source_identity=chunk_identities[item.source_path.resolve()],
                    source_segment=item.source_segment,
                )
                for item, destination in batch
            ]
            with _filter_fma_mpeg_stderr(suppress_mpeg_warning):
                chunk_metadata = {
                    metadata.path: metadata
                    for metadata in materialize_all(
                        tasks,
                        workers=workers,
                        checkpoint=layout.state / "audio.sqlite3",
                    )
                }
            metadata_by_path.update(chunk_metadata)


@contextmanager
def _filter_fma_mpeg_stderr(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    # A few files in the checksummed FMA archive contain damaged MPEG frames that
    # mpg123 recovers. Suppress its diagnostic here; normal audio validation still runs.
    sys.stderr.flush()
    with tempfile.TemporaryFile() as captured:
        saved_stderr = os.dup(2)
        try:
            os.dup2(captured.fileno(), 2)
            yield
        finally:
            sys.stderr.flush()
            os.dup2(saved_stderr, 2)
            os.close(saved_stderr)
            captured.seek(0)
            for line in captured:
                if _FMA_RECOVERED_FRAME_WARNING.fullmatch(line.rstrip(b"\r\n")):
                    continue
                remaining = memoryview(line)
                while remaining:
                    remaining = remaining[os.write(2, remaining) :]


def _manifest_records(
    items: tuple[InventoryItem, ...],
    *,
    split: Split,
    loaded: LoadedEdition,
    layout: WorkspaceLayout,
    implementation_fingerprint: str,
    metadata_by_path: dict[Path, AudioMetadata],
) -> Iterator[ManifestItem]:
    for item in items:
        destination = _prepared_path(
            item,
            loaded=loaded,
            layout=layout,
            implementation_fingerprint=implementation_fingerprint,
        )
        metadata = metadata_by_path[destination]
        yield ManifestItem.from_inventory(
            item,
            audio_path=destination.relative_to(layout.prepared_audio.parent).as_posix(),
            split=split,
            sample_rate_hz=metadata.sample_rate_hz,
            channels=metadata.channels,
            frame_count=metadata.num_frames,
            checksum=metadata.sha256,
        )


def _prepared_path(
    item: InventoryItem,
    *,
    loaded: LoadedEdition,
    layout: WorkspaceLayout,
    implementation_fingerprint: str,
) -> Path:
    key = fingerprint(
        {
            "id": item.id,
            "source_release": item.source_release,
            "sample_rate_hz": loaded.config.audio.sample_rate_hz,
            "channels": loaded.config.audio.channels.for_media_kind(item.media_kind).value,
            "source_segment": (
                item.source_segment.model_dump(mode="json")
                if item.source_segment is not None
                else None
            ),
            "implementation": implementation_fingerprint,
        }
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
    relative_paths = (
        path.relative_to(source_root).as_posix() for path in source_root.rglob("*.py")
    )
    return _code_digest(repo_root, *relative_paths)


def _inventory_implementation_fingerprint(
    repo_root: Path,
    dataset_id: str,
    environment: dict[str, Any],
    adapter_source: Path,
) -> str:
    """Identify inventory behavior without coupling reuse to unrelated source edits."""

    runtime: dict[str, Any] = {"python": environment["python"]}
    if dataset_id == "globe":
        runtime["pyarrow"] = environment["packages"]["pyarrow"]
    elif dataset_id == "fsd50k":
        runtime["zip"] = environment["zip"]
    return fingerprint(
        {
            "code": _code_digest(
                repo_root,
                "datasets/__init__.py",
                "datasets/base.py",
                "datasets/common.py",
                "datasets/inventory.py",
                "datasets/io.py",
                "manifests.py",
                "models.py",
            ),
            "adapter": sha256_file(adapter_source),
            "dataset": dataset_id,
            "runtime": runtime,
        }
    )


def _audio_implementation_fingerprint(
    repo_root: Path,
    environment: dict[str, Any],
) -> str:
    """Identify conversion behavior and only the libraries that implement it."""

    return fingerprint(
        {
            "code": _code_digest(repo_root, "audio.py", "models.py", "state.py"),
            "python": environment["python"],
            "packages": {
                name: environment["packages"][name]
                for name in ("libsndfile", "numpy", "scipy", "soundfile")
            },
        }
    )


def _code_digest(repo_root: Path, *relative_paths: str) -> str:
    root = _source_root(repo_root)
    return fingerprint(
        [(relative, sha256_file(root / relative)) for relative in sorted(relative_paths)]
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


def _remove_tree(path: Path) -> None:
    try:
        details = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(details.st_mode):
        raise RuntimeError(f"managed cache directory must not be a symlink: {path}")
    shutil.rmtree(path)


def _require_preparation_requirements(loaded: LoadedEdition, workspace: Path) -> None:
    readiness = collect_preparation_readiness(
        loaded,
        workspace=workspace,
    )
    if readiness.unresolved:
        raise RuntimeError("preparation is not ready: " + "; ".join(readiness.unresolved))


def _local_source_fingerprints(sources: Iterable[Any]) -> list[tuple[str, Any]]:
    fingerprints: list[tuple[str, Any]] = []
    for source in sources:
        path = source.path
        if path is None:
            continue
        if path.is_file():
            value: Any = trusted_file_sha256(path)
        elif path.is_dir():
            digest = hashlib.sha256()
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    relative = child.relative_to(path).as_posix()
                    record = (relative, trusted_file_sha256(child))
                    digest.update(f"{canonical_json(record)}\n".encode())
            value = digest.hexdigest()
        else:
            value = None
        fingerprints.append((source.name, value))
    return fingerprints


def _verify_local_sources(sources: Iterable[Any], *, workers: int) -> None:
    """Authenticate configured local files before an adapter consumes them."""

    configured = [
        (source.path, source.checksum)
        for source in sources
        if source.path is not None and source.path.is_file() and source.checksum
    ]

    def verify(configured_source: tuple[Path, str]) -> None:
        path, checksum = configured_source
        verify_checksum(path, checksum)

    _bounded_thread_map(
        verify,
        configured,
        workers=workers,
        thread_name="lrac-local-source-hash",
    )


def _portable_path(path: Path, repo_root: Path, workspace: Path) -> str:
    resolved = path.resolve()
    for prefix, root in (("repo", repo_root), ("workspace", workspace)):
        try:
            relative = resolved.relative_to(root.resolve())
        except ValueError:
            continue
        return f"{prefix}:{relative.as_posix()}"
    return f"external:{resolved.name}"


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
    return [
        {
            "path": _portable_path(resolved, repo_root, workspace),
            "sha256": digest,
        }
        for resolved, digest in zip(resolved_paths, hashes, strict=True)
    ]


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
    allow_missing_workspace: bool = False,
) -> bool:
    expected: list[tuple[Path, str, bool]] = []
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
        expected.append(
            (candidate, record["sha256"], allow_missing_workspace and prefix == "workspace")
        )

    def artifact_status(path: Path) -> tuple[str, str | None]:
        try:
            details = path.stat()
        except FileNotFoundError:
            return ("missing", None)
        if not stat.S_ISREG(details.st_mode):
            return ("invalid", None)
        try:
            return ("present", trusted_file_sha256(path))
        except OSError:
            return ("invalid", None)

    actual = _bounded_thread_map(
        artifact_status,
        [path for path, _digest, _allow_missing in expected],
        workers=workers,
        thread_name="lrac-artifact-verify",
    )
    return all(
        (status == "present" and digest == expected_digest)
        or (allow_missing and status == "missing")
        for (_path, expected_digest, allow_missing), (status, digest) in zip(
            expected,
            actual,
            strict=True,
        )
    )


def _inventory_digest(
    items: Iterable[InventoryItem],
    *,
    repo_root: Path,
    workspace: Path,
) -> str:
    digest = hashlib.sha256()
    for item in items:
        record = item.model_dump(mode="json", exclude_none=True)
        record["source_path"] = _portable_path(item.source_path, repo_root, workspace)
        digest.update(f"{canonical_json(record)}\n".encode())
    return digest.hexdigest()


def _validate_inventory_completeness(dataset: Any, items: list[InventoryItem]) -> None:
    ids: set[str] = set()
    counts: Counter[Any] = Counter()
    for item in items:
        if item.id in ids:
            raise ValueError(f"dataset {dataset.id!r} inventory contains duplicate IDs")
        ids.add(item.id)
        counts[item.media_kind] += 1
    missing = [kind.value for kind in dataset.media_kinds if counts[kind] == 0]
    if missing:
        raise _IncompleteInventoryError(
            f"dataset {dataset.id!r} inventory has no items for: {', '.join(missing)}"
        )
    mismatches = [
        f"{kind.value}: expected {expected}, found {counts[kind]}"
        for kind, expected in dataset.expected_inventory.items()
        if counts[kind] != expected
    ]
    if mismatches:
        raise _IncompleteInventoryError(
            f"dataset {dataset.id!r} inventory count mismatch: " + "; ".join(mismatches)
        )


def _update_selected_source_digest(
    digest: Any,
    items: Iterable[InventoryItem],
    *,
    source_identities: Mapping[Path, FileIdentity],
) -> None:
    for item in items:
        source = item.source_checksum or source_identities[item.source_path.resolve()].as_dict()
        segment = (
            item.source_segment.model_dump(mode="json")
            if item.source_segment is not None
            else None
        )
        digest.update(f"{canonical_json((item.id, source, segment))}\n".encode())


def _source_identities(
    items: Iterable[InventoryItem],
    *,
    workers: int,
) -> dict[Path, FileIdentity]:
    paths = list(dict.fromkeys(item.source_path.resolve() for item in items))

    def identity(path: Path) -> FileIdentity:
        return FileIdentity.from_stat(path.stat())

    identities = _bounded_thread_map(
        identity,
        paths,
        workers=workers,
        thread_name="lrac-source-stat",
    )
    return dict(zip(paths, identities, strict=True))


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


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
