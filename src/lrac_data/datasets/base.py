"""Dataset adapter contract and shared path conventions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Any

from lrac_data.models import (
    DatasetConfig,
    InventoryItem,
    MediaKind,
    SourceSpec,
    qualify_id,
)
from lrac_data.state import fingerprint, sha256_file

from . import io as dataset_io


class DatasetAdapter(ABC):
    """Fetch a corpus and describe its complete, unfiltered source inventory.

    Adapters deliberately do not apply edition curation or validation/evaluation
    exclusions.  Those policies operate on the normalized inventory returned here.
    """

    def __init__(
        self,
        config: DatasetConfig,
        repo_root: Path,
        workspace: Path,
        *,
        workers: int = 4,
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be at least 1")
        self.config = config
        self.repo_root = Path(repo_root).resolve()
        self.workspace = Path(workspace).resolve()
        self.workers = workers

    @property
    def download_dir(self) -> Path:
        return self._cache_dir("downloads")

    @property
    def extracted_dir(self) -> Path:
        return self._cache_dir("extracted")

    def _cache_dir(self, category: str) -> Path:
        category_root = (self.workspace / category).resolve()
        _require_descendant(
            category_root,
            self.workspace,
            f"dataset {category} root",
        )
        dataset_root = category_root / self.config.id
        candidate = (
            dataset_root / self.cache_namespace
            if self.cache_namespace is not None
            else dataset_root
        ).resolve()
        return _require_descendant(
            candidate,
            category_root,
            f"dataset {self.config.id!r} {category} directory",
        )

    @cached_property
    def cache_namespace(self) -> str | None:
        """Scope production caches to immutable source configuration and local inputs."""

        if not self.config.sources:
            return None
        sources: list[dict[str, Any]] = []
        for source in self.config.sources:
            data = source.model_dump(mode="json", exclude={"path"}, exclude_none=True)
            if source.path is not None:
                path = source.path
                if path.is_file():
                    data["local_input"] = sha256_file(path)
                elif path.is_dir():
                    data["local_input"] = [
                        (child.relative_to(path).as_posix(), sha256_file(child))
                        for child in sorted(path.rglob("*"))
                        if child.is_file()
                    ]
                else:
                    data["local_input"] = None
            sources.append(data)
        return fingerprint(
            {
                "dataset": self.config.id,
                "adapter": self.config.adapter,
                "release": self.config.release,
                "sources": sources,
            }
        )[:16]

    def option(self, name: str, default: Any = None) -> Any:
        return self.config.options.get(name, default)

    def source(self, name: str) -> SourceSpec:
        for source in self.config.sources:
            if source.name == name:
                return source
        raise KeyError(f"Dataset {self.config.id!r} has no source named {name!r}")

    def remote_source(self, name: str) -> tuple[SourceSpec, str, str]:
        """Return a source and its required remote URL and local filename."""

        source = self.source(name)
        if source.url is None or source.filename is None:
            raise ValueError(
                f"Dataset {self.config.id!r} source {name!r} must define url and filename"
            )
        return source, source.url, source.filename

    def download_remote_sources(self, *names: str) -> list[Path]:
        """Download named fixed-file sources concurrently in the requested order."""

        root = self.download_dir.resolve()
        requests: list[dataset_io.DownloadRequest] = []
        for name in names:
            source, url, filename = self.remote_source(name)
            destination = (root / filename).resolve()
            if destination == root or not destination.is_relative_to(root):
                raise ValueError(
                    f"Dataset {self.config.id!r} source {name!r} filename escapes "
                    f"its download directory: {filename!r}"
                )
            requests.append(
                dataset_io.DownloadRequest(
                    url=url,
                    destination=destination,
                    checksum=source.checksum,
                )
            )
        return dataset_io.download_many(
            requests,
            max_workers=self.workers,
            downloader=dataset_io.download_file,
        )

    def item(
        self,
        source_id: str,
        media_kind: MediaKind,
        source_path: Path,
        **kwargs: Any,
    ) -> InventoryItem:
        """Build an inventory item with the repository-wide stable ID scheme."""

        return InventoryItem(
            id=qualify_id(self.config.id, source_id),
            dataset=self.config.id,
            source_id=source_id,
            source_release=self.config.release,
            media_kind=media_kind,
            source_path=source_path.resolve(),
            **kwargs,
        )

    def ensure_expected_files(self) -> None:
        missing = [
            pattern
            for pattern in self.config.expected_files
            if not any(self.extracted_dir.glob(pattern))
        ]
        if missing:
            formatted = ", ".join(repr(pattern) for pattern in missing)
            raise FileNotFoundError(
                f"Dataset {self.config.id!r} is incomplete; no files match {formatted} "
                f"under {self.extracted_dir}"
            )

    @abstractmethod
    def fetch(self) -> Path:
        """Download and extract the complete configured corpus."""

    @abstractmethod
    def inventory(self) -> list[InventoryItem]:
        """Return every eligible source item before edition selection."""


def _require_descendant(path: Path, root: Path, label: str) -> Path:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes {root}: {path}") from error
    if relative == Path("."):
        raise ValueError(f"{label} must not be its root: {root}")
    return path
