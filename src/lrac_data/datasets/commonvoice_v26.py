# Copyright 2026 Cisco Systems, Inc. and its affiliates
# Apache-2.0

"""Common Voice 26.0 download and inventory adapter."""

from __future__ import annotations

import csv
import os
import re
import tarfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from lrac_data.models import InventoryItem, MediaKind, SourceSpec

from . import io as dataset_io
from .base import DatasetAdapter

_MDC_API_BASE = "https://mozilladatacollective.com/api"
_MDC_API_KEY_ENV = "MDC_API_KEY"
_MDC_DATASET_ID = re.compile(r"[A-Za-z0-9_-]+")
_MDC_SHA256 = re.compile(r"(?:sha256:)?([0-9a-fA-F]{64})")
_METADATA_TABLES = ("validated", "other", "invalidated")
_ARCHIVE_LOCALES = {"zh": "zh-CN"}


@dataclass(frozen=True, slots=True)
class _Corpus:
    metadata_tables: tuple[tuple[str, Path], ...]
    clips_root: Path
    archive_locale: str
    source_locale: str


class CommonVoiceV26Adapter(DatasetAdapter):
    """Download and inventory the configured Common Voice locale archives.

    Mozilla Data Collective requires one-time terms acceptance and an API key.
    The adapter exchanges ``MDC_API_KEY`` for temporary archive URLs, then uses
    the repository's ordinary resumable, checksum-verifying downloader.
    """

    def fetch(self) -> Path:
        roots: list[tuple[SourceSpec, Path]] = []
        for index, (source, archive) in enumerate(
            zip(self.config.sources, self._download_remote_archives(), strict=True)
        ):
            destination = (self.extracted_dir / f"source-{index:03d}").resolve()
            try:
                dataset_io.safe_extract_tar(archive, destination)
            except tarfile.TarError as error:
                raise ValueError(
                    f"Common Voice source {source.name!r} is not a readable tar archive: {archive}"
                ) from error
            roots.append((source, destination))
        self._corpora(roots)
        return self.extracted_dir

    def inventory(self) -> list[InventoryItem]:
        roots = [
            (source, (self.extracted_dir / f"source-{index:03d}").resolve())
            for index, source in enumerate(self.config.sources)
        ]
        corpora = self._corpora(roots)
        records: list[InventoryItem] = []
        source_ids: dict[str, Path] = {}

        for corpus in corpora:
            for source_subset, metadata_path in corpus.metadata_tables:
                for line_number, row in _read_metadata_tsv(metadata_path):
                    records.append(
                        self._inventory_item(
                            corpus,
                            source_subset,
                            metadata_path,
                            line_number,
                            row,
                            source_ids,
                        )
                    )

        return sorted(records, key=lambda item: item.id)

    def _inventory_item(
        self,
        corpus: _Corpus,
        source_subset: str,
        metadata_path: Path,
        line_number: int,
        row: dict[str, str],
        source_ids: dict[str, Path],
    ) -> InventoryItem:
        clip_name = _safe_clip_name(row["path"], metadata_path, line_number)
        clip_path = _resolve_clip(
            corpus.clips_root,
            clip_name,
            metadata_path,
            line_number,
        )
        source_id = f"{corpus.archive_locale}/{clip_path.stem}"
        previous = source_ids.get(source_id)
        if previous is not None:
            raise ValueError(
                f"Duplicate Common Voice source ID {source_id!r}: {previous} and {clip_path}"
            )
        source_ids[source_id] = clip_path

        locale = _optional_identifier(
            row.get("locale"),
            field="locale",
            path=metadata_path,
            line_number=line_number,
        )
        if locale is not None and locale != corpus.archive_locale:
            raise ValueError(
                f"{metadata_path}:{line_number}: locale {locale!r} does not "
                f"match archive locale {corpus.archive_locale!r}"
            )
        locale = locale or corpus.archive_locale
        speaker_id = _optional_identifier(
            row.get("client_id"),
            field="client_id",
            path=metadata_path,
            line_number=line_number,
        )
        gender = _optional_identifier(
            row.get("gender"),
            field="gender",
            path=metadata_path,
            line_number=line_number,
        )
        metadata = {
            "locale": locale,
            "archive_locale": corpus.archive_locale,
            "source_subset": source_subset,
            "source_locale": corpus.source_locale,
        }

        return self.item(
            source_id,
            MediaKind.SPEECH,
            clip_path,
            speaker_id=speaker_id,
            text=row.get("sentence") or None,
            language=locale,
            gender=gender,
            metadata=metadata,
        )

    def _download_remote_archives(self) -> list[Path]:
        api_key = os.environ.get(_MDC_API_KEY_ENV)

        def download(entry: tuple[int, SourceSpec]) -> Path:
            index, source = entry
            dataset_id = _mdc_dataset_id(source)
            api_url = f"{_MDC_API_BASE}/datasets/{dataset_id}/download"
            destination = self._download_path(index)
            cached_checksum = dataset_io.cached_download_checksum(
                destination,
                state_url=api_url,
            )
            if cached_checksum is not None:
                return destination
            if not api_key:
                raise RuntimeError(
                    "Common Voice downloads require MDC_API_KEY; create an API key in "
                    "Mozilla Data Collective after accepting each dataset's terms"
                )
            download_url, download_checksum = _request_download_session(source, api_url, api_key)
            return dataset_io.download_file(
                download_url,
                destination,
                checksum=download_checksum,
                state_url=api_url,
            )

        sources = list(enumerate(self.config.sources))
        workers = min(self.workers, len(sources))
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="lrac-commonvoice",
        ) as executor:
            return list(executor.map(download, sources))

    def _download_path(self, index: int) -> Path:
        return (self.download_dir / f"source-{index:03d}.tar.gz").resolve()

    def _corpora(self, roots: list[tuple[SourceSpec, Path]]) -> list[_Corpus]:
        corpora: list[_Corpus] = []
        for source, root in roots:
            resolved_root = root.resolve()
            matches = sorted(
                resolved_root.rglob("validated.tsv"),
                key=lambda path: path.relative_to(resolved_root).as_posix(),
            )
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Common Voice source {source.name!r} expected one validated.tsv under "
                    f"{resolved_root}, found {len(matches)}"
                )

            locale_root = matches[0].resolve().parent
            archive_locale = locale_root.name
            expected_locale = _ARCHIVE_LOCALES.get(source.name, source.name)
            _validate_identifier(expected_locale, "archive locale")
            if archive_locale != expected_locale:
                raise ValueError(
                    f"Common Voice source {source.name!r} expected archive locale "
                    f"{expected_locale!r}, found {archive_locale!r}"
                )

            clips_root = (locale_root / "clips").resolve()
            if not clips_root.is_dir():
                raise FileNotFoundError(
                    f"Common Voice clips directory is missing for {archive_locale!r}: {clips_root}"
                )
            tables = tuple(
                (name, (locale_root / f"{name}.tsv").resolve()) for name in _METADATA_TABLES
            )
            missing = [path for _, path in tables if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Common Voice metadata table is missing: {missing[0]}")
            corpora.append(
                _Corpus(
                    metadata_tables=tables,
                    clips_root=clips_root,
                    archive_locale=archive_locale,
                    source_locale=source.name,
                )
            )

        return sorted(corpora, key=lambda corpus: corpus.archive_locale)


def _mdc_dataset_id(source: SourceSpec) -> str:
    parsed = urlsplit(source.url or "")
    parts = parsed.path.rstrip("/").split("/")
    dataset_id = parts[-1]
    if (
        parsed.scheme != "https"
        or parsed.netloc != "mozilladatacollective.com"
        or len(parts) < 3
        or parts[-2] != "datasets"
        or _MDC_DATASET_ID.fullmatch(dataset_id) is None
    ):
        raise ValueError(f"Common Voice source {source.name!r} requires an MDC dataset landing URL")
    return dataset_id


def _request_download_session(
    source: SourceSpec,
    api_url: str,
    api_key: str,
) -> tuple[str, str]:
    try:
        response = httpx.post(
            api_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )
    except httpx.HTTPError as error:
        raise RuntimeError(
            f"Mozilla Data Collective could not create a download for "
            f"Common Voice source {source.name!r}"
        ) from error

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        raise RuntimeError(
            f"Common Voice source {source.name!r}: MDC download request failed"
        ) from error

    try:
        payload = response.json()
    except ValueError as error:
        raise RuntimeError(
            f"Common Voice source {source.name!r}: MDC returned invalid JSON"
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Common Voice source {source.name!r}: MDC returned an invalid download response"
        )

    download_url = payload.get("downloadUrl")
    parsed_url = urlsplit(download_url) if isinstance(download_url, str) else None
    if (
        parsed_url is None
        or parsed_url.scheme != "https"
        or not parsed_url.netloc
        or parsed_url.username is not None
        or parsed_url.password is not None
    ):
        raise RuntimeError(
            f"Common Voice source {source.name!r}: MDC returned an invalid download URL"
        )
    checksum = payload.get("checksum")
    checksum_match = _MDC_SHA256.fullmatch(checksum) if isinstance(checksum, str) else None
    if checksum_match is None:
        raise RuntimeError(
            f"Common Voice source {source.name!r}: MDC returned an invalid SHA-256 checksum"
        )
    assert isinstance(download_url, str)
    return download_url, f"sha256:{checksum_match.group(1).lower()}"


def _read_metadata_tsv(path: Path) -> Iterator[tuple[int, dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(
            stream,
            delimiter="\t",
            quoting=csv.QUOTE_NONE,
            strict=True,
        )
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"Common Voice metadata is empty: {path}") from error
        except csv.Error as error:
            raise ValueError(f"Invalid Common Voice TSV header in {path}: {error}") from error

        if not header or any(not name for name in header):
            raise ValueError(f"Common Voice TSV has an empty column name: {path}")
        if len(header) != len(set(header)):
            duplicates = sorted(name for name in set(header) if header.count(name) > 1)
            raise ValueError(
                f"Common Voice TSV has duplicate columns in {path}: {', '.join(duplicates)}"
            )
        required = {"client_id", "path", "sentence"}
        missing = sorted(required.difference(header))
        if missing:
            raise ValueError(
                f"Common Voice TSV is missing required columns in {path}: {', '.join(missing)}"
            )

        seen_paths: dict[str, int] = {}
        try:
            for values in reader:
                line_number = reader.line_num
                if len(values) != len(header):
                    raise ValueError(
                        f"{path}:{line_number}: expected {len(header)} TSV fields, "
                        f"found {len(values)}"
                    )
                row = dict(zip(header, values, strict=True))
                clip_name = row["path"]
                previous_line = seen_paths.get(clip_name)
                if previous_line is not None:
                    raise ValueError(
                        f"Duplicate Common Voice clip path {clip_name!r} in {path}: "
                        f"lines {previous_line} and {line_number}"
                    )
                seen_paths[clip_name] = line_number
                yield line_number, row
        except csv.Error as error:
            raise ValueError(
                f"Invalid Common Voice TSV record in {path} near line {reader.line_num}: {error}"
            ) from error


def _safe_clip_name(value: str, path: Path, line_number: int) -> str:
    if not value:
        raise ValueError(f"{path}:{line_number}: Common Voice clip path is blank")
    clip_path = Path(value)
    if (
        value != value.strip()
        or any(character.isspace() for character in value)
        or value in {".", ".."}
        or clip_path.is_absolute()
        or len(clip_path.parts) != 1
        or "/" in value
        or "\\" in value
        or clip_path.suffix != ".mp3"
    ):
        raise ValueError(
            f"{path}:{line_number}: unsafe Common Voice clip path {value!r}; "
            "expected one clips/*.mp3 filename"
        )
    return value


def _resolve_clip(clips_root: Path, name: str, metadata_path: Path, line_number: int) -> Path:
    candidate = clips_root / name
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"{metadata_path}:{line_number}: Common Voice clip is missing: {candidate}"
        ) from error
    if not resolved.is_relative_to(clips_root):
        raise ValueError(
            f"{metadata_path}:{line_number}: Common Voice clip escapes clips directory: {candidate}"
        )
    if not resolved.is_file():
        raise FileNotFoundError(
            f"{metadata_path}:{line_number}: Common Voice clip is not a file: {candidate}"
        )
    return resolved


def _optional_identifier(
    value: str | None,
    *,
    field: str,
    path: Path,
    line_number: int,
) -> str | None:
    if value is None or value == "":
        return None
    try:
        _validate_identifier(value, field)
    except ValueError as error:
        raise ValueError(f"{path}:{line_number}: {error}") from error
    return value


def _validate_identifier(value: str, field: str) -> None:
    if not value or value != value.strip() or any(character.isspace() for character in value):
        raise ValueError(f"Common Voice {field} must be a non-empty value without whitespace")
