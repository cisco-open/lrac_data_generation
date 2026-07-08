"""Read-only planning for a complete LRAC edition build."""

from __future__ import annotations

import importlib.util
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from .config import LoadedEdition, load_edition_config
from .datasets import ADAPTERS
from .models import SelectionMode


@dataclass(frozen=True)
class DatasetPlan:
    id: str
    adapter: str
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
    integrity_warnings: tuple[str, ...] = ()
    public_evaluation: dict[str, Any] | None = None
    remote_checks: tuple[RemoteCheck, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_plan(
    edition: str | Path,
    *,
    selection: SelectionMode | str = SelectionMode.CURATED,
    repo_root: Path | None = None,
    check_remote: bool = False,
) -> PlanReport:
    """Resolve and validate an edition without writing or downloading anything."""

    mode = SelectionMode(selection)
    loaded = load_edition_config(edition, repo_root=repo_root, selection=mode)
    config = loaded.config

    datasets: list[DatasetPlan] = []
    unresolved: list[str] = []
    integrity_warnings: list[str] = []
    for dataset in config.datasets:
        if dataset.adapter not in ADAPTERS:
            unresolved.append(
                f"{dataset.id}: unknown adapter {dataset.adapter!r}; "
                f"available: {', '.join(sorted(ADAPTERS))}"
            )
        missing_inventory_counts = sorted(
            set(dataset.media_kinds) - set(dataset.expected_inventory)
        )
        if missing_inventory_counts:
            integrity_warnings.append(
                f"{dataset.id}: canonical inventory counts are not yet pinned for "
                + ", ".join(kind.value for kind in missing_inventory_counts)
            )
        curation_targets = sum(
            len(rule.source_ids) for rule in config.curations if rule.dataset == dataset.id
        )
        exclusion_targets = sum(
            len(rule.source_ids) + len(rule.speaker_ids)
            for rule in config.exclusions
            if rule.dataset in (None, dataset.id)
        )
        datasets.append(
            DatasetPlan(
                id=dataset.id,
                adapter=dataset.adapter,
                release=dataset.release,
                media_kinds=tuple(kind.value for kind in dataset.media_kinds),
                sources=len(dataset.sources),
                curation_targets=curation_targets,
                exclusion_targets=exclusion_targets,
            )
        )
        for source in dataset.sources:
            if source.path is not None:
                local_path = _resolve_local_source(source.path, loaded)
                if not local_path.exists():
                    unresolved.append(
                        f"{dataset.id}/{source.name}: local source does not exist: {local_path}"
                    )
            if (
                source.url is not None
                and source.checksum is None
                and not source.options.get("checksums")
            ):
                integrity_warnings.append(
                    f"{dataset.id}/{source.name}: no upstream archive checksum; "
                    "the run will record the downloaded SHA-256"
                )

    checks = _check_remotes(loaded) if check_remote else ()
    unresolved.extend(
        f"{check.dataset}/{check.source}: {check.detail or 'remote check failed'}"
        for check in checks
        if not check.ok
    )
    requirements = {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "git": shutil.which("git") is not None,
        "zip": shutil.which("zip") is not None,
        "numpy": importlib.util.find_spec("numpy") is not None,
        "scipy": importlib.util.find_spec("scipy") is not None,
        "soundfile": importlib.util.find_spec("soundfile") is not None,
        "pyarrow": importlib.util.find_spec("pyarrow") is not None,
    }
    if not requirements["ffmpeg"]:
        unresolved.append("required executable is not installed: ffmpeg")
    if config.public_evaluation is not None and not requirements["git"]:
        unresolved.append("required executable is not installed: git")
    if not requirements["zip"]:
        unresolved.append("required executable is not installed: zip")
    for package in ("numpy", "pyarrow", "scipy", "soundfile"):
        if not requirements[package]:
            unresolved.append(f"required preparation package is not installed: {package}")

    stages = [
        "resolve configuration",
        "fetch complete sources",
        "build normalized inventory",
        "apply mandatory validation/evaluation exclusions",
        (
            "apply quality curation"
            if mode is SelectionMode.CURATED
            else "select all eligible inventory"
        ),
    ]
    if config.public_evaluation is not None:
        stages.append("fetch pinned public evaluation data")
    stages.extend(("materialize mono PCM16 WAV", "validate and publish manifests"))

    return PlanReport(
        edition=config.edition,
        selection=mode.value,
        policy=mode.policy_name,
        config_path=str(loaded.path),
        target_audio=config.audio.model_dump(mode="json"),
        datasets=tuple(datasets),
        stages=tuple(stages),
        requirements=requirements,
        unresolved=tuple(sorted(set(unresolved))),
        integrity_warnings=tuple(sorted(set(integrity_warnings))),
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
            if source.url is None:
                continue
            requests.extend(
                (dataset.id, artifact_name, check_url)
                for artifact_name, check_url in _remote_artifacts(dataset, source)
            )
    if loaded.config.public_evaluation is not None:
        spec = loaded.config.public_evaluation
        requests.append((spec.id, "git-repository", spec.repository_url))

    with httpx.Client(follow_redirects=True, timeout=15.0) as client:

        def check(request: tuple[str, str, str]) -> RemoteCheck:
            dataset, source, url = request
            try:
                response = client.request("HEAD", url)
                ok = response.is_success
                return RemoteCheck(
                    dataset=dataset,
                    source=source,
                    url=url,
                    status=response.status_code,
                    ok=ok,
                    detail=None if ok else f"HTTP {response.status_code}",
                )
            except httpx.HTTPError as error:
                return RemoteCheck(
                    dataset=dataset,
                    source=source,
                    url=url,
                    status=None,
                    ok=False,
                    detail=str(error),
                )

        workers = min(16, len(requests))
        if workers == 0:
            return ()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return tuple(executor.map(check, requests))


def _resolve_local_source(path: Path, loaded: LoadedEdition) -> Path:
    return path if path.is_absolute() else loaded.repo_root / path


def _remote_artifacts(dataset: Any, source: Any) -> list[tuple[str, str]]:
    assert source.url is not None
    url = source.url
    if "{suffix}" in url:
        suffixes = dataset.options.get("speech_part_suffixes") or tuple(
            source.options.get("checksums", {})
        )
        return [(f"{source.name}:{suffix}", url.format(suffix=suffix)) for suffix in suffixes]
    if "{speaker}" in url:
        first = int(source.options.get("first", 1))
        last = int(source.options.get("last", first))
        return [
            (f"{source.name}:p{index:03d}", url.format(speaker=f"p{index:03d}"))
            for index in range(first, last + 1)
        ]
    if "{index" in url:
        first = int(source.options.get("first", 0))
        last = int(source.options.get("last", first))
        return [
            (f"{source.name}:{index:04d}", url.format(index=index))
            for index in range(first, last + 1)
        ]
    if "{path}" in url and dataset.adapter == "mls":
        metadata = next(
            (
                candidate.path
                for candidate in dataset.sources
                if candidate.name == "source_metadata"
            ),
            None,
        )
        if metadata is not None:
            artifacts: list[tuple[str, str]] = []
            for language in dataset.options.get("languages", []):
                shard_list = metadata / f"mls_{language}_train_track1_data.txt"
                for remote_path in shard_list.read_text(encoding="utf-8").splitlines():
                    if remote_path.strip():
                        artifacts.append(
                            (
                                f"{source.name}:{language}:{remote_path.strip()}",
                                url.format(path=remote_path.strip()),
                            )
                        )
            return artifacts
    return [
        (
            source.name,
            str(source.options.get("remote_check_url", url)),
        )
    ]
