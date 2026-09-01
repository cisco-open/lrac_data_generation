# Copyright 2026 Cisco Systems, Inc. and its affiliates
# Apache-2.0

"""Load and validate repository-relative LRAC YAML configuration."""

from __future__ import annotations

import csv
import sysconfig
from collections import defaultdict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any

import yaml
from pydantic import ValidationError

from lrac_data.models import (
    CurationSpec,
    DatasetConfig,
    EditionConfig,
    ExclusionPartition,
    ExclusionSpec,
    SelectionMode,
)


class ConfigError(ValueError):
    """Raised when edition or dataset configuration cannot be resolved."""


@dataclass(frozen=True)
class LoadedEdition:
    """A validated edition plus the paths used to resolve it."""

    config: EditionConfig
    path: Path
    repo_root: Path


def portable_config_payload(loaded: LoadedEdition) -> dict[str, Any]:
    """Return the path-stable resolved configuration used for run fingerprints."""

    payload = loaded.config.model_dump(mode="python", exclude_none=True)

    def normalize(value: Any) -> Any:
        if isinstance(value, Path):
            return _portable_repo_path(value, loaded.repo_root)
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


def _portable_repo_path(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(repo_root.resolve())
    except ValueError:
        return f"external:{resolved.name}"
    return f"repo:{relative.as_posix()}"


def load_edition_config(
    edition: str | Path,
    *,
    repo_root: Path | None = None,
    selection: SelectionMode | str | None = None,
) -> LoadedEdition:
    """Load an edition and all referenced dataset YAML files.

    A simple edition name such as ``"2026"`` resolves to
    ``configs/editions/2026.yaml``. Dataset names resolve to
    ``configs/datasets/<name>.yaml``.

    No paths are inspected until this function is called, so importing the package
    has no file-system side effects.
    """

    root = _resolve_repo_root(repo_root)
    edition_path = _resolve_config_path(edition, root, "editions", "edition configuration")
    raw_edition = _load_mapping(edition_path, "edition")

    raw_datasets = raw_edition.get("datasets")
    if not isinstance(raw_datasets, list):
        raise ConfigError(f"{edition_path}: 'datasets' must be a YAML list")

    datasets = tuple(
        _load_dataset_reference(reference, root, edition_path) for reference in raw_datasets
    )
    resolved = dict(raw_edition)
    resolved["datasets"] = datasets
    mode = SelectionMode(selection) if selection is not None else SelectionMode.CURATED
    resolved["curations"] = (
        _resolve_curations(resolved, root, edition_path) if mode is SelectionMode.CURATED else ()
    )
    resolved["exclusions"] = _resolve_exclusions(resolved, root, edition_path)
    resolved.pop("curation_files", None)
    resolved.pop("exclusion_files", None)

    try:
        config = EditionConfig.model_validate(resolved)
    except ValidationError as error:
        raise ConfigError(f"{edition_path}: invalid edition configuration: {error}") from error
    return LoadedEdition(config=config, path=edition_path, repo_root=root)


def _load_dataset_reference(
    reference: Any,
    root: Path,
    edition_path: Path,
) -> DatasetConfig:
    if not isinstance(reference, str):
        raise ConfigError(
            f"{edition_path}: dataset references must be names, got {type(reference).__name__}"
        )
    path = _resolve_config_path(reference, root, "datasets", "dataset configuration")
    return _validate_dataset(_load_mapping(path, "dataset"), root, path)


def _validate_dataset(
    raw_dataset: Mapping[str, Any],
    root: Path,
    source_path: Path,
) -> DatasetConfig:
    resolved = dict(raw_dataset)
    sources = resolved.get("sources", [])
    if not isinstance(sources, list):
        raise ConfigError(f"{source_path}: dataset 'sources' must be a YAML list")

    resolved_sources: list[Any] = []
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise ConfigError(
                f"{source_path}: sources[{index}] must be a mapping, got {type(source).__name__}"
            )
        resolved_source = dict(source)
        local_path = resolved_source.get("path")
        if local_path is not None:
            candidate = Path(local_path).expanduser()
            resolved_source["path"] = (
                candidate if candidate.is_absolute() else root / candidate
            ).resolve()
        resolved_sources.append(resolved_source)
    resolved["sources"] = resolved_sources

    try:
        return DatasetConfig.model_validate(resolved)
    except ValidationError as error:
        raise ConfigError(f"{source_path}: invalid dataset configuration: {error}") from error


def _resolve_curations(
    raw_edition: Mapping[str, Any],
    root: Path,
    edition_path: Path,
) -> tuple[CurationSpec, ...]:
    references = raw_edition.get("curation_files", [])
    if not isinstance(references, list):
        raise ConfigError(f"{edition_path}: 'curation_files' must be a YAML list")
    curations: list[CurationSpec] = []
    for index, reference in enumerate(references):
        if not isinstance(reference, Mapping):
            raise ConfigError(f"{edition_path}: curation_files[{index}] must be a mapping")
        curations.append(_load_curation_file(dict(reference), root, edition_path, index))
    return tuple(curations)


def _load_curation_file(
    reference: dict[str, Any],
    root: Path,
    edition_path: Path,
    index: int,
) -> CurationSpec:
    expected = {"dataset", "media_kind", "path"}
    if set(reference) != expected:
        missing = sorted(expected - set(reference))
        unexpected = sorted(set(reference) - expected)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ConfigError(
            f"{edition_path}: curation_files[{index}] must contain dataset, media_kind, and path "
            f"({'; '.join(details)})"
        )

    path = _resolve_policy_path(reference["path"], root, "curation file")
    source_ids: list[str] = []
    seen: set[str] = set()
    with _open_csv(path) as (header, rows):
        if header != ["source_id"]:
            raise ConfigError(f"{path}: curation header must be exactly source_id")
        for row_number, row in rows:
            source_id = _clean_cell(row.get("source_id"))
            if not source_id:
                raise ConfigError(f"{path}:{row_number}: blank curation source ID")
            if any(character.isspace() for character in source_id):
                raise ConfigError(
                    f"{path}:{row_number}: source ID may not contain whitespace: {source_id!r}"
                )
            if source_id in seen:
                raise ConfigError(
                    f"{path}:{row_number}: duplicate curation source ID {source_id!r}"
                )
            seen.add(source_id)
            source_ids.append(source_id)

    if not source_ids:
        raise ConfigError(f"{path}: curation file contains no data rows")
    try:
        return CurationSpec(
            dataset=reference["dataset"],
            media_kind=reference["media_kind"],
            source_ids=tuple(source_ids),
        )
    except (ValidationError, ValueError) as error:
        raise ConfigError(f"{edition_path}: invalid curation_files[{index}]: {error}") from error


def _resolve_exclusions(
    raw_edition: Mapping[str, Any],
    root: Path,
    edition_path: Path,
) -> tuple[ExclusionSpec, ...]:
    references = raw_edition.get("exclusion_files", [])
    if not isinstance(references, list):
        raise ConfigError(f"{edition_path}: 'exclusion_files' must be a YAML list")

    grouped: defaultdict[tuple[str, str, str], dict[str, list[str]]] = defaultdict(
        lambda: {"source_ids": [], "speaker_ids": []}
    )
    for index, reference in enumerate(references):
        if not isinstance(reference, (str, Path)):
            raise ConfigError(f"{edition_path}: exclusion_files[{index}] must be a path string")
        path = _resolve_policy_path(reference, root, "exclusion file")
        with _open_csv(path) as (header, rows):
            required_header = ("name", "partition", "dataset", "source_id", "speaker_id")
            if tuple(header) != required_header:
                raise ConfigError(
                    f"{path}: exclusion header must be exactly " + ",".join(required_header)
                )
            for row_number, row in rows:
                name = _required_cell(row, "name", path, row_number)
                partition = _required_cell(row, "partition", path, row_number)
                dataset = _required_cell(row, "dataset", path, row_number)
                source_id = _clean_cell(row.get("source_id"))
                speaker_id = _clean_cell(row.get("speaker_id"))
                if bool(source_id) == bool(speaker_id):
                    raise ConfigError(
                        f"{path}:{row_number}: set exactly one of source_id or speaker_id"
                    )
                key = (name, partition, dataset)
                target_key = "source_ids" if source_id else "speaker_ids"
                target = source_id or speaker_id
                assert target is not None
                if target in grouped[key][target_key]:
                    raise ConfigError(
                        f"{path}:{row_number}: duplicate {target_key[:-1]} {target!r} "
                        f"for exclusion {name!r}"
                    )
                grouped[key][target_key].append(target)

    exclusions: list[ExclusionSpec] = []
    for (name, partition, dataset), targets in grouped.items():
        try:
            exclusions.append(
                ExclusionSpec(
                    name=name,
                    partition=ExclusionPartition(partition),
                    dataset=dataset,
                    source_ids=tuple(targets["source_ids"]),
                    speaker_ids=tuple(targets["speaker_ids"]),
                )
            )
        except (ValidationError, ValueError) as error:
            raise ConfigError(f"invalid exclusion group {name!r}: {error}") from error

    seen_source: set[tuple[str, str]] = set()
    seen_speaker: set[tuple[str, str]] = set()
    for exclusion in exclusions:
        for target in exclusion.source_ids:
            target_identity = (exclusion.dataset, target)
            if target_identity in seen_source:
                raise ConfigError(
                    f"duplicate source exclusion {target!r} in dataset {exclusion.dataset!r}"
                )
            seen_source.add(target_identity)
        for target in exclusion.speaker_ids:
            target_identity = (exclusion.dataset, target)
            if target_identity in seen_speaker:
                raise ConfigError(
                    f"duplicate speaker exclusion {target!r} in dataset {exclusion.dataset!r}"
                )
            seen_speaker.add(target_identity)
    return tuple(exclusions)


def _resolve_policy_path(value: Any, root: Path, description: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise ConfigError(f"{description} path must be a string")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return _require_file(candidate.resolve(), description)


@contextmanager
def _open_csv(
    path: Path,
) -> Iterator[tuple[list[str], Iterator[tuple[int, dict[str, str | None]]]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ConfigError(f"{path}: CSV header is missing")
        header = [field.strip() for field in reader.fieldnames]
        if any(not field for field in header):
            raise ConfigError(f"{path}: CSV header contains a blank column")
        if len(header) != len(set(header)):
            raise ConfigError(f"{path}: CSV header contains duplicate columns")

        def rows() -> Iterator[tuple[int, dict[str, str | None]]]:
            for row_number, row in enumerate(reader, start=2):
                if None in row:
                    raise ConfigError(f"{path}:{row_number}: row has too many columns")
                normalized = {
                    str(key).strip(): value for key, value in row.items() if key is not None
                }
                if not any(_clean_cell(value) for value in normalized.values()):
                    raise ConfigError(f"{path}:{row_number}: blank CSV row")
                yield row_number, normalized

        yield header, rows()


def _required_cell(
    row: Mapping[str, str | None],
    column: str,
    path: Path,
    row_number: int,
) -> str:
    value = _clean_cell(row.get(column))
    if not value:
        raise ConfigError(f"{path}:{row_number}: blank required column {column!r}")
    return value


def _clean_cell(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _resolve_repo_root(repo_root: Path | None) -> Path:
    if repo_root is not None:
        root = Path(repo_root).expanduser().resolve()
        if not root.is_dir():
            raise ConfigError(f"repository root does not exist or is not a directory: {root}")
        return root

    for starting_point in (Path.cwd(), Path(__file__).resolve()):
        discovered = _discover_repo_root(starting_point)
        if discovered is not None:
            return discovered
    installed = Path(sysconfig.get_path("data")) / "share" / "lrac-data"
    if (installed / "configs" / "editions").is_dir():
        return installed.resolve()
    raise ConfigError("could not discover repository root; pass repo_root explicitly")


def _discover_repo_root(starting_point: Path) -> Path | None:
    start = starting_point.resolve()
    if start.is_file():
        start = start.parent
    for candidate in (start, *start.parents):
        if (candidate / "configs").is_dir() and (
            (candidate / "pyproject.toml").is_file() or (candidate / ".git").exists()
        ):
            return candidate
    return None


def _resolve_config_path(
    reference: str | Path,
    root: Path,
    directory: str,
    description: str,
) -> Path:
    value = Path(reference).expanduser()
    if value.is_absolute():
        candidate = value
    elif value.suffix.lower() in {".yaml", ".yml"} or len(value.parts) > 1:
        candidate = root / value
    else:
        candidate = root / "configs" / directory / f"{value}.yaml"
    return _require_file(candidate.resolve(), description)


def _require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise ConfigError(f"{description} not found: {path}")
    return path


def _load_mapping(path: Path, description: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = yaml.load(stream, Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as error:
        raise ConfigError(f"{path}: invalid YAML: {error}") from error
    if not isinstance(value, dict):
        raise ConfigError(f"{path}: {description} configuration must be a YAML mapping")
    return value


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects silently overwritten mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)
