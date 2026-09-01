# Copyright 2026 Cisco Systems, Inc. and its affiliates
# Apache-2.0

"""Dataset adapter contract and shared path conventions."""

from __future__ import annotations

import shutil
import stat
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from lrac_data.models import (
    DatasetConfig,
    InventoryItem,
    MediaKind,
    SourceSpec,
    qualify_id,
)

from . import io as dataset_io


class DatasetAdapter(ABC):
    """Fetch a corpus and describe its complete, unfiltered source inventory.

    Adapters deliberately do not apply edition curation or validation/evaluation
    exclusions. Those policies operate on the normalized inventory returned here.
    """

    def __init__(
        self,
        config: DatasetConfig,
        workspace: Path,
        *,
        workers: int = 4,
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be at least 1")
        self.config = config
        self.workspace = Path(workspace).resolve()
        self.workers = workers

    @property
    def download_dir(self) -> Path:
        return self.workspace / "downloads" / self.config.id

    @property
    def extracted_dir(self) -> Path:
        return self.workspace / "extracted" / self.config.id

    def source(self, name: str) -> SourceSpec:
        for source in self.config.sources:
            if source.name == name:
                return source
        raise KeyError(f"Dataset {self.config.id!r} has no source named {name!r}")

    def download_remote_sources(self, *names: str) -> list[Path]:
        """Download fixed, checksummed sources in the requested order."""

        root = self.download_dir.resolve()
        requests: list[dataset_io.DownloadRequest] = []
        for name in names:
            source = self.source(name)
            if source.url is None or source.filename is None or source.checksum is None:
                raise ValueError(
                    f"Dataset {self.config.id!r} source {name!r} requires a URL, "
                    "filename, and checksum"
                )
            url, filename = source.url, source.filename
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

    @abstractmethod
    def fetch(self) -> Path:
        """Download and extract the complete configured corpus."""

    @abstractmethod
    def inventory(self) -> list[InventoryItem]:
        """Return every eligible source item before edition selection."""

    def fetch_selected(self, items: tuple[InventoryItem, ...]) -> None:
        """Fetch item-level sources for adapters that can defer large downloads."""

        missing = _missing_regular_files(item.source_path for item in items)
        if not missing:
            return
        self.clear_extracted()
        self.fetch()
        remaining = _missing_regular_files(missing)
        if remaining:
            raise FileNotFoundError(remaining[0])

    def clear_extracted(self) -> None:
        """Remove a damaged extraction and its archive completion markers."""

        _remove_tree(self.extracted_dir)
        _remove_tree(self.extracted_dir.parent / f".{self.extracted_dir.name}.lrac-extract")

    def clear_downloads(self) -> None:
        """Remove files downloaded into this adapter's managed cache."""

        _remove_tree(self.download_dir)

    def provenance_artifacts(self) -> tuple[Path, ...]:
        """Return immutable inputs that define the normalized inventory."""

        artifacts = [
            path
            for path in sorted(self.download_dir.rglob("*"))
            if path.is_file()
            and not path.name.endswith(".part")
            and not path.name.endswith(".download.json")
        ]
        for source in self.config.sources:
            if source.path is None:
                continue
            if source.path.is_file():
                artifacts.append(source.path)
            elif source.path.is_dir():
                artifacts.extend(path for path in sorted(source.path.rglob("*")) if path.is_file())
        return tuple(dict.fromkeys(path.resolve() for path in artifacts))


def _remove_tree(path: Path) -> None:
    try:
        details = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(details.st_mode):
        raise RuntimeError(f"managed cache directory must not be a symlink: {path}")
    shutil.rmtree(path)


def _missing_regular_files(paths: Iterable[Path]) -> list[Path]:
    missing: list[Path] = []
    for path in dict.fromkeys(paths):
        try:
            details = path.stat()
        except FileNotFoundError:
            missing.append(path)
            continue
        if not stat.S_ISREG(details.st_mode):
            missing.append(path)
    return missing
