# Copyright 2026 Cisco Systems, Inc. and its affiliates
# Apache-2.0

"""Read-only planning for a complete LRAC edition build."""

from __future__ import annotations

import importlib.util
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from .audio import missing_soundfile_formats
from .config import LoadedEdition, load_edition_config
from .datasets import ADAPTERS
from .datasets.io import (
    cached_download_checksum,
    huggingface_auth_header,
    is_huggingface_resolver_url,
)
from .models import DatasetConfig, MediaKind, SelectionMode, SourceSpec


@dataclass(frozen=True)
class DatasetPlan:
    id: str
    release: str
    media_kinds: tuple[str, ...]
    sources: int
    curation_targets: int
    exclusion_targets: int


@dataclass(frozen=True)
class RemoteCheck:
    dataset: str
    source: str
    url: str
    status: int | None
    ok: bool
    detail: str | None = None


@dataclass(frozen=True)
class PlanReport:
    edition: str
    selection: str
    policy: str
    config_path: str
    target_audio: dict[str, Any]
    datasets: tuple[DatasetPlan, ...]
    stages: tuple[str, ...]
    requirements: dict[str, bool]
    unresolved: tuple[str, ...]
    public_evaluation: dict[str, Any] | None = None
    remote_checks: tuple[RemoteCheck, ...] = ()


@dataclass(frozen=True)
class PreparationReadiness:
    requirements: dict[str, bool]
    unresolved: tuple[str, ...]


def collect_preparation_readiness(
    loaded: LoadedEdition,
    *,
    workspace: Path | None = None,
) -> PreparationReadiness:
    """Collect every local preparation blocker without network or filesystem writes."""

    config = loaded.config
    unresolved: list[str] = []
    commonvoice_sources = False
    commonvoice_archives_reusable = True

    for dataset in config.datasets:
        if dataset.id not in ADAPTERS:
            unresolved.append(
                f"unknown dataset {dataset.id!r}; available: {', '.join(sorted(ADAPTERS))}"
            )
        remote_commonvoice = [
            (index, source)
            for index, source in enumerate(dataset.sources)
            if dataset.id == "commonvoice_v26" and source.path is None
        ]
        if remote_commonvoice:
            commonvoice_sources = True
            if workspace is None or not _commonvoice_archives_cached(
                dataset, remote_commonvoice, workspace
            ):
                commonvoice_archives_reusable = False

        for source in dataset.sources:
            if source.path is not None:
                local_path = source.path
                if dataset.id == "commonvoice_v26" and not local_path.is_file():
                    unresolved.append(
                        f"{dataset.id}/{source.name}: bound archive does not exist or "
                        f"is not a file: {local_path}"
                    )
                elif not local_path.exists():
                    unresolved.append(
                        f"{dataset.id}/{source.name}: local source does not exist: {local_path}"
                    )

    soundfile_installed = importlib.util.find_spec("soundfile") is not None
    missing_audio_formats = missing_soundfile_formats() if soundfile_installed else ()
    requirements = {
        "git": shutil.which("git") is not None,
        "zip": shutil.which("zip") is not None,
        "numpy": importlib.util.find_spec("numpy") is not None,
        "scipy": importlib.util.find_spec("scipy") is not None,
        "soundfile": soundfile_installed and not missing_audio_formats,
        "pyarrow": importlib.util.find_spec("pyarrow") is not None,
    }
    if commonvoice_sources:
        requirements["MDC_API_KEY"] = bool(os.environ.get("MDC_API_KEY")) or (
            commonvoice_archives_reusable
        )
        if not requirements["MDC_API_KEY"]:
            unresolved.append(
                "MDC_API_KEY is not set; Common Voice downloads require an MDC API key "
                "after one-time terms acceptance"
            )
    if any(
        source.url is not None and is_huggingface_resolver_url(source.url)
        for dataset in config.datasets
        for source in dataset.sources
    ):
        requirements["HF_TOKEN (optional)"] = bool(os.environ.get("HF_TOKEN", "").strip())
    if config.public_evaluation is not None and not requirements["git"]:
        unresolved.append("required executable is not installed: git")
    if any(dataset.id == "fsd50k" for dataset in config.datasets) and not requirements["zip"]:
        unresolved.append("required executable is not installed: zip")
    for package in ("numpy", "scipy"):
        if not requirements[package]:
            unresolved.append(f"required preparation package is not installed: {package}")
    if any(dataset.id == "globe" for dataset in config.datasets) and not requirements["pyarrow"]:
        unresolved.append("required preparation package is not installed: pyarrow")
    if not soundfile_installed:
        unresolved.append("required preparation package is not installed: soundfile")
    elif missing_audio_formats:
        unresolved.append(
            "SoundFile/libsndfile does not support required formats: "
            + ", ".join(missing_audio_formats)
        )

    return PreparationReadiness(
        requirements=requirements,
        unresolved=tuple(sorted(set(unresolved))),
    )


def build_plan(
    edition: str | Path,
    *,
    selection: SelectionMode | str = SelectionMode.CURATED,
    repo_root: Path | None = None,
    check_remote: bool = False,
) -> PlanReport:
    """Resolve and validate an edition without writing or downloading anything."""

    mode = SelectionMode(selection)
    loaded = load_edition_config(
        edition,
        repo_root=repo_root,
        selection=mode,
    )
    config = loaded.config

    datasets: list[DatasetPlan] = []
    for dataset in config.datasets:
        curation_targets = sum(
            len(rule.source_ids) for rule in config.curations if rule.dataset == dataset.id
        )
        exclusion_targets = sum(
            len(rule.source_ids) + len(rule.speaker_ids)
            for rule in config.exclusions
            if rule.dataset == dataset.id
        )
        datasets.append(
            DatasetPlan(
                id=dataset.id,
                release=dataset.release,
                media_kinds=tuple(kind.value for kind in dataset.media_kinds),
                sources=len(dataset.sources),
                curation_targets=curation_targets,
                exclusion_targets=exclusion_targets,
            )
        )
    readiness = collect_preparation_readiness(loaded)
    checks = _check_remotes(loaded) if check_remote else ()
    unresolved = list(readiness.unresolved)
    unresolved.extend(
        f"{check.dataset}/{check.source}: {check.detail or 'remote check failed'}"
        for check in checks
        if not check.ok
    )
    stages = [
        "resolve configuration",
        "fetch source metadata and fixed archives",
        "build normalized inventory",
        "apply mandatory validation/evaluation exclusions",
        (
            "apply quality curation"
            if mode is SelectionMode.CURATED
            else "select all eligible inventory"
        ),
        "fetch selected item-level audio",
    ]
    if config.public_evaluation is not None:
        stages.append("fetch pinned public evaluation data")
    channel_modes = ", ".join(
        f"{kind.value}={config.audio.channels.for_media_kind(kind).value}" for kind in MediaKind
    )
    stages.extend(
        (
            f"materialize 24 kHz float32 WAV (channels: {channel_modes})",
            "publish JSONL and Kaldi views after complete release validation",
        )
    )

    return PlanReport(
        edition=config.edition,
        selection=mode.value,
        policy=mode.policy_name,
        config_path=str(loaded.path),
        target_audio=config.audio.model_dump(mode="json"),
        datasets=tuple(datasets),
        stages=tuple(stages),
        requirements=readiness.requirements,
        unresolved=tuple(sorted(set(unresolved))),
        public_evaluation=(
            config.public_evaluation.model_dump(mode="json")
            if config.public_evaluation is not None
            else None
        ),
        remote_checks=tuple(checks),
    )


def _check_remotes(loaded: LoadedEdition) -> tuple[RemoteCheck, ...]:
    requests: list[tuple[str, str, str]] = []
    for dataset in loaded.config.datasets:
        for source in dataset.sources:
            if source.url is None or dataset.id == "commonvoice_v26":
                continue
            artifact_name, check_url = _remote_artifact(source)
            requests.append((dataset.id, artifact_name, check_url))
    if loaded.config.public_evaluation is not None:
        spec = loaded.config.public_evaluation
        requests.append((spec.id, "git-repository", spec.repository_url))

    if not requests:
        return ()

    with httpx.Client(follow_redirects=True, timeout=15.0) as client:

        def check(request: tuple[str, str, str]) -> RemoteCheck:
            dataset, source, url = request
            for attempt in range(3):
                try:
                    auth_header = huggingface_auth_header(url)
                    headers = dict([auth_header]) if auth_header is not None else None
                    response = client.request("HEAD", url, headers=headers)
                except httpx.HTTPError as error:
                    return RemoteCheck(
                        dataset=dataset,
                        source=source,
                        url=url,
                        status=None,
                        ok=False,
                        detail=str(error),
                    )
                retryable = response.status_code == 429 or response.status_code >= 500
                if retryable and attempt < 2:
                    retry_after = response.headers.get("Retry-After")
                    delay = (
                        int(retry_after)
                        if retry_after and retry_after.isdecimal()
                        else 2**attempt
                    )
                    time.sleep(min(delay, 30))
                    continue
                ok = response.is_success
                return RemoteCheck(
                    dataset=dataset,
                    source=source,
                    url=url,
                    status=response.status_code,
                    ok=ok,
                    detail=None if ok else f"HTTP {response.status_code}",
                )
            raise AssertionError("remote retry loop did not return")

        with ThreadPoolExecutor(max_workers=min(4, len(requests))) as executor:
            return tuple(executor.map(check, requests))


def _commonvoice_archives_cached(
    dataset: DatasetConfig,
    sources: list[tuple[int, SourceSpec]],
    workspace: Path,
) -> bool:
    download_dir = workspace.expanduser().resolve() / "downloads" / dataset.id
    for index, source in sources:
        parsed = urlsplit(source.url or "")
        dataset_id = parsed.path.rstrip("/").rsplit("/", 1)[-1]
        api_url = f"https://mozilladatacollective.com/api/datasets/{dataset_id}/download"
        archive = download_dir / f"source-{index:03d}.tar.gz"
        if cached_download_checksum(archive, state_url=api_url) is None:
            return False
    return True


def _remote_artifact(source: SourceSpec) -> tuple[str, str]:
    assert source.url is not None
    url = source.url
    artifact_name = next(iter(source.artifact_checksums), None)
    if artifact_name is None:
        return source.name, url
    label = f"{source.name}:{artifact_name}"
    if "{suffix}" in url:
        filename_prefix = urlsplit(url).path.rsplit("/", 1)[-1].split("{suffix}", 1)[0]
        return label, url.format(suffix=artifact_name.removeprefix(filename_prefix))
    if "{speaker}" in url:
        return label, url.format(speaker=Path(artifact_name).stem)
    if "{index" in url:
        return label, url.format(index=int(Path(artifact_name).stem))
    if "{path}" in url:
        return label, url.format(path=artifact_name)
    return source.name, url
