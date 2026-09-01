# Copyright 2026 Cisco Systems, Inc. and its affiliates
# Apache-2.0

"""Network and archive helpers used by dataset adapters.

Nothing in this module performs I/O at import time.  Downloads use a sibling
``.part`` file and archives are extracted member-by-member after path validation.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import IO, Any
from urllib.parse import urlsplit

from lrac_data.state import FileIdentity, atomic_write_text, canonical_json


class ChecksumError(ValueError):
    """Raised when a downloaded artifact does not match its declared digest."""


class UnsafeArchiveError(ValueError):
    """Raised when an archive member could escape its extraction root."""


@dataclass(frozen=True, slots=True)
class DownloadRequest:
    """One independently downloadable remote artifact."""

    url: str
    destination: Path
    checksum: str | None = None


_CONTENT_RANGE = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", re.IGNORECASE)


def is_huggingface_resolver_url(url: str) -> bool:
    """Return whether a URL is an authenticated Hugging Face resolver endpoint."""

    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme.lower() == "https"
        and parsed.hostname == "huggingface.co"
        and port in (None, 443)
        and "/resolve/" in parsed.path
    )


def huggingface_auth_header(url: str) -> tuple[str, str] | None:
    """Return optional resolver authentication without exposing the token elsewhere."""

    token = os.environ.get("HF_TOKEN", "").strip()
    if token and is_huggingface_resolver_url(url):
        return "Authorization", f"Bearer {token}"
    return None


def _checksum_parts(checksum: str) -> tuple[str, str]:
    if ":" in checksum:
        algorithm, expected = checksum.split(":", 1)
    elif len(checksum) == 40:
        algorithm, expected = "sha1", checksum
    elif len(checksum) == 64:
        algorithm, expected = "sha256", checksum
    else:
        raise ValueError("Checksums must use '<algorithm>:<digest>' or be a SHA-1/SHA-256 digest")
    return algorithm.lower(), expected.lower()


def _file_checksum(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=256)
def _stat_bound_checksum(
    path: str,
    algorithm: str,
    identity: FileIdentity,
) -> str:
    del identity
    return _file_checksum(Path(path), algorithm)


def _trusted_checksum(path: Path, algorithm: str) -> str:
    """Hash a file once per process while its filesystem identity is unchanged."""

    resolved = Path(path).resolve()
    before = FileIdentity.from_stat(resolved.stat())
    digest = _stat_bound_checksum(str(resolved), algorithm, before)
    if not before.matches_stat(resolved.stat()):
        raise OSError(f"File changed while computing its checksum: {resolved}")
    return digest


def trusted_file_sha256(path: Path) -> str:
    """Return a cached download SHA-256 when its file stat still matches.

    Files without a valid download sidecar are hashed normally.  The sidecar is
    deliberately only a stat-keyed cache: callers that need independent deep
    validation should continue to hash the file directly.
    """

    path = Path(path)
    state = _read_download_state(path)
    if state is not None and _download_state_matches_file(path, state):
        digest = _state_sha256(state)
        if digest is not None:
            return digest
    return _trusted_checksum(path, "sha256")


def cached_download_checksum(path: Path, *, state_url: str) -> str | None:
    """Return the verified checksum for an unchanged completed download."""

    path = Path(path)
    state = _read_download_state(path)
    if (
        state is None
        or state.get("url") != state_url
        or not _download_state_matches_file(path, state)
    ):
        return None
    checksum = state.get("checksum")
    if not isinstance(checksum, str):
        return None
    try:
        normalized = _normalize_checksum(checksum)
    except ValueError:
        return None
    algorithm, expected = _checksum_parts(normalized)
    recorded_sha256 = _state_sha256(state)
    if recorded_sha256 is None or (algorithm == "sha256" and expected != recorded_sha256):
        return None
    return normalized


def verify_checksum(path: Path, checksum: str | None) -> None:
    if not checksum:
        return
    algorithm, expected = _checksum_parts(_normalize_checksum(checksum))
    actual = _trusted_checksum(path, algorithm)
    if actual != expected:
        raise ChecksumError(
            f"Checksum mismatch for {path}: expected {algorithm}:{expected}, "
            f"got {algorithm}:{actual}"
        )


def require_checksum_map(
    value: object,
    expected_keys: Iterable[str],
    *,
    label: str,
) -> dict[str, str]:
    """Validate and normalize checksums for a templated remote source."""

    ordered_keys = tuple(expected_keys)
    expected = set(ordered_keys)
    if len(expected) != len(ordered_keys):
        raise ValueError(f"{label} contains duplicate artifact keys")
    if not isinstance(value, dict):
        raise ValueError(f"{label} requires a checksum for every artifact")

    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} checksum map mismatch: expected {len(expected)}, got {len(actual)} "
            f"(missing {len(expected - actual)}, unexpected {len(actual - expected)})"
        )

    normalized: dict[str, str] = {}
    for key in ordered_keys:
        checksum = value[key]
        if not isinstance(checksum, str):
            raise ValueError(f"{label} checksum for {key!r} must be a string")
        normalized[key] = _normalize_checksum(checksum)
    return normalized


def download_file(
    url: str,
    destination: Path,
    *,
    checksum: str | None = None,
    attempts: int = 4,
    timeout: float = 60.0,
    state_url: str | None = None,
) -> Path:
    """Stream a URL atomically and retain its already-computed SHA-256.

    A checksummed partial response is resumed only when the server confirms the
    requested byte range. Unpinned downloads restart because a changed remote
    entity cannot otherwise be detected. ``state_url`` provides a stable, safe
    identity for temporary credential-bearing URLs written to resume metadata.
    """

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if attempts < 1:
        raise ValueError("attempts must be at least 1")

    normalized_checksum = _normalize_checksum(checksum) if checksum else None
    persisted_url = state_url or url
    configured_algorithm = (
        _checksum_parts(normalized_checksum)[0] if normalized_checksum is not None else "sha256"
    )
    state_path = _download_state_path(destination)

    if destination.is_symlink():
        destination.unlink()
        state_path.unlink(missing_ok=True)
    elif destination.is_file() and destination.stat().st_size > 0:
        state = _read_download_state(destination)
        state_matches = state is not None and _download_state_matches_file(destination, state)
        state_sha256 = _state_sha256(state) if state_matches and state is not None else None
        if (
            state_matches
            and state is not None
            and state.get("url") == persisted_url
            and state_sha256 is not None
            and (
                normalized_checksum is None or state.get("checksum") == normalized_checksum
            )
        ):
            return destination

        # A newly pinned SHA-256 can authenticate a stat-bound cache entry
        # without rereading the artifact.  This also upgrades its sidecar so
        # subsequent calls take the exact-match path above.
        if normalized_checksum is not None and state_sha256 is not None:
            configured_algorithm, configured_digest = _checksum_parts(normalized_checksum)
            if configured_algorithm == "sha256" and configured_digest == state_sha256:
                _write_download_state(
                    destination,
                    url=persisted_url,
                    checksum=normalized_checksum,
                    sha256=state_sha256,
                )
                return destination

        # A configured checksum can authenticate an otherwise untrusted cache
        # with one read.  Without one, only the exact stat/URL/sidecar match
        # above is trustworthy; an arbitrary pre-existing file must be fetched.
        if normalized_checksum is None:
            destination.unlink()
            state_path.unlink(missing_ok=True)
        else:
            digests = _file_digests(destination, configured_algorithm)
            try:
                _verify_digest(destination, normalized_checksum, digests)
            except ChecksumError:
                destination.unlink()
                state_path.unlink(missing_ok=True)
            else:
                _write_download_state(
                    destination,
                    url=persisted_url,
                    checksum=normalized_checksum,
                    sha256=digests["sha256"].hexdigest(),
                )
                return destination
    elif destination.exists():
        destination.unlink()
        state_path.unlink(missing_ok=True)

    partial = destination.with_name(f"{destination.name}.part")
    if partial.is_symlink() or (partial.exists() and not partial.is_file()):
        partial.unlink()
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        if normalized_checksum is None:
            partial.unlink(missing_ok=True)
        offset = partial.stat().st_size if partial.exists() else 0
        digests = (
            _file_digests(partial, configured_algorithm)
            if offset
            else _new_digests(configured_algorithm)
        )
        headers = {"User-Agent": "lrac-data/2026"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        if auth_header := huggingface_auth_header(url):
            request.add_unredirected_header(*auth_header)
        try:
            with closing(urllib.request.urlopen(request, timeout=timeout)) as response:
                status = getattr(response, "status", response.getcode())
                append = bool(offset and status == 206)
                response_range = _response_range(response)
                if status == 206 and (response_range is None or response_range[0] != offset):
                    partial.unlink(missing_ok=True)
                    raise OSError(f"Server returned an invalid Content-Range while resuming {url}")
                if not append:
                    digests = _new_digests(configured_algorithm)
                    offset = 0
                expected_total = _response_total(response, response_range)
                mode = "ab" if append else "wb"
                with partial.open(mode) as output:
                    received = 0
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        received += len(chunk)
                        for digest in digests.values():
                            digest.update(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                _validate_response_length(
                    response,
                    response_range=response_range,
                    received=received,
                    resulting_size=offset + received,
                    expected_total=expected_total,
                    url=url,
                )
            final_digests = {name: digest.hexdigest() for name, digest in digests.items()}
            if normalized_checksum is not None:
                _verify_digest(partial, normalized_checksum, final_digests)
            partial.replace(destination)
            recorded_checksum = normalized_checksum or f"sha256:{final_digests['sha256']}"
            _write_download_state(
                destination,
                url=persisted_url,
                checksum=recorded_checksum,
                sha256=final_digests["sha256"],
            )
            return destination
        except ChecksumError as exc:
            last_error = exc
            partial.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(_retry_delay(attempt, exc))
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 416:
                partial.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(_retry_delay(attempt, exc))
    assert last_error is not None
    message = f"Failed to download {persisted_url} after {attempts} attempts"
    if state_url is not None:
        raise RuntimeError(message) from None
    raise RuntimeError(message) from last_error


def _retry_delay(attempt: int, error: BaseException) -> float:
    fallback = float(min(2 ** (attempt - 1), 8))
    if not isinstance(error, urllib.error.HTTPError) or error.code != 429:
        return fallback
    retry_after = error.headers.get("Retry-After")
    if not isinstance(retry_after, str) or not retry_after.isascii() or not retry_after.isdecimal():
        return fallback
    return float(min(int(retry_after), 300))


def _normalize_checksum(checksum: str) -> str:
    algorithm, expected = _checksum_parts(checksum)
    return f"{algorithm}:{expected}"


def _new_digests(configured_algorithm: str) -> dict[str, Any]:
    digests: dict[str, Any] = {"sha256": hashlib.sha256()}
    if configured_algorithm != "sha256":
        digests[configured_algorithm] = hashlib.new(configured_algorithm)
    return digests


def _file_digests(path: Path, configured_algorithm: str) -> dict[str, Any]:
    digests = _new_digests(configured_algorithm)
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            for digest in digests.values():
                digest.update(chunk)
    return digests


def _verify_digest(
    path: Path,
    checksum: str,
    digests: dict[str, Any],
) -> None:
    algorithm, expected = _checksum_parts(checksum)
    value = digests[algorithm]
    actual = value if isinstance(value, str) else value.hexdigest()
    if actual != expected:
        raise ChecksumError(
            f"Checksum mismatch for {path}: expected {algorithm}:{expected}, "
            f"got {algorithm}:{actual}"
        )


def _response_range(response: Any) -> tuple[int, int, int | None] | None:
    headers = getattr(response, "headers", None)
    content_range = headers.get("Content-Range") if headers is not None else None
    if not isinstance(content_range, str):
        return None
    match = _CONTENT_RANGE.fullmatch(content_range.strip())
    if match is None:
        return None
    total = None if match.group(3) == "*" else int(match.group(3))
    return int(match.group(1)), int(match.group(2)), total


def _response_total(
    response: Any,
    response_range: tuple[int, int, int | None] | None,
) -> int | None:
    if response_range is not None:
        return response_range[2]
    headers = getattr(response, "headers", None)
    content_length = headers.get("Content-Length") if headers is not None else None
    try:
        return int(content_length) if content_length is not None else None
    except (TypeError, ValueError):
        return None


def _validate_response_length(
    response: Any,
    *,
    response_range: tuple[int, int, int | None] | None,
    received: int,
    resulting_size: int,
    expected_total: int | None,
    url: str,
) -> None:
    headers = getattr(response, "headers", None)
    content_length = headers.get("Content-Length") if headers is not None else None
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except (TypeError, ValueError):
            declared_length = None
        if declared_length is not None and received != declared_length:
            raise OSError(f"Server returned a truncated response while downloading {url}")
    if response_range is not None:
        start, end, _total = response_range
        if end < start or received != end - start + 1:
            raise OSError(f"Server returned an invalid Content-Range body for {url}")
    if expected_total is not None and resulting_size != expected_total:
        raise OSError(f"Server returned an incomplete ranged response while downloading {url}")


def _download_state_path(destination: Path) -> Path:
    return destination.with_name(f"{destination.name}.download.json")


def _read_download_state(path: Path) -> dict[str, Any] | None:
    try:
        state = json.loads(_download_state_path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return state if isinstance(state, dict) else None


def _download_state_matches_file(path: Path, state: dict[str, Any]) -> bool:
    try:
        current = path.stat()
    except OSError:
        return False
    identity = FileIdentity.from_dict(state.get("identity"))
    return bool(identity is not None and identity.matches_stat(current))


def _state_sha256(state: dict[str, Any]) -> str | None:
    digest = state.get("sha256")
    if isinstance(digest, str) and _is_hex_digest(digest, 64):
        return digest
    return None


def _write_download_state(
    destination: Path,
    *,
    url: str,
    checksum: str,
    sha256: str,
) -> None:
    current = destination.stat()
    state = {
        "url": url,
        "checksum": checksum,
        "identity": FileIdentity.from_stat(current).as_dict(),
        "sha256": sha256,
    }
    atomic_write_text(_download_state_path(destination), f"{canonical_json(state)}\n")


def _is_hex_digest(value: str, length: int) -> bool:
    return len(value) == length and all(character in "0123456789abcdef" for character in value)


def download_many(
    requests: Iterable[DownloadRequest],
    *,
    max_workers: int = 4,
) -> list[Path]:
    """Download independent artifacts concurrently in input order."""

    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")

    ordered = tuple(requests)
    destinations = [Path(request.destination) for request in ordered]
    if len(destinations) != len(set(destinations)):
        raise ValueError("download destinations must be unique")
    if not ordered:
        return []

    def download(request: DownloadRequest) -> Path:
        return download_file(
            request.url,
            request.destination,
            checksum=request.checksum,
        )

    with ThreadPoolExecutor(
        max_workers=min(max_workers, len(ordered)),
        thread_name_prefix="lrac-download",
    ) as executor:
        return list(executor.map(download, ordered))


def _safe_member_path(
    root: Path,
    member_name: str,
    *,
    resolved_root: Path | None = None,
) -> Path:
    normalized = PurePosixPath(member_name.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise UnsafeArchiveError(f"Unsafe archive member: {member_name!r}")
    target = root.joinpath(*normalized.parts)
    try:
        target.resolve().relative_to(resolved_root or root.resolve())
    except ValueError as exc:
        raise UnsafeArchiveError(f"Unsafe archive member: {member_name!r}") from exc
    return target


def _copy_member(source: IO[bytes], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.part")
    if partial.is_symlink() or partial.exists():
        if partial.is_dir() and not partial.is_symlink():
            raise UnsafeArchiveError(f"Archive partial path is a directory: {partial}")
        partial.unlink()
    try:
        descriptor = os.open(
            partial,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        partial.replace(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def safe_extract_tar(
    archive: Path,
    destination: Path,
    *,
    strip_prefix: str | None = None,
) -> Path:
    archives = (Path(archive),)
    marker, state = _extraction_marker(destination, archives, strip_prefix)
    if destination.is_dir() and _marker_matches(marker, state):
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as bundle:
        _extract_tar_members(bundle, destination, strip_prefix=strip_prefix)
    _mark_complete(marker, state)
    return destination


def safe_extract_multipart_tar(
    parts: Iterable[Path],
    destination: Path,
) -> Path:
    """Safely stream an ordered multipart ``tar.gz`` into ``destination``."""

    ordered_parts = tuple(Path(part) for part in parts)
    if not ordered_parts:
        raise ValueError("multipart extraction requires at least one part")
    marker, state = _extraction_marker(destination, ordered_parts, None)
    if destination.is_dir() and _marker_matches(marker, state):
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    with (
        _MultipartReader(ordered_parts) as stream,
        tarfile.open(fileobj=stream, mode="r|gz") as bundle,
    ):
        _extract_tar_members(bundle, destination, strip_prefix=None)
    _mark_complete(marker, state)
    return destination


def _extract_tar_members(
    bundle: tarfile.TarFile,
    destination: Path,
    *,
    strip_prefix: str | None,
) -> None:
    prefix = PurePosixPath(strip_prefix) if strip_prefix else None
    resolved_destination = destination.resolve()
    for member in bundle:
        if member.issym() or member.islnk() or member.isdev():
            raise UnsafeArchiveError(f"Links and device members are not allowed: {member.name!r}")
        name = PurePosixPath(member.name)
        if prefix is not None:
            try:
                name = name.relative_to(prefix)
            except ValueError:
                continue
        if not name.parts:
            continue
        target = _safe_member_path(
            destination,
            name.as_posix(),
            resolved_root=resolved_destination,
        )
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
        elif member.isfile():
            source = bundle.extractfile(member)
            if source is None:
                raise tarfile.ExtractError(f"Could not read {member.name!r}")
            with source:
                _copy_member(source, target)


class _MultipartReader(io.RawIOBase):
    """A forward-only binary stream over a fixed sequence of files."""

    def __init__(self, parts: tuple[Path, ...]) -> None:
        super().__init__()
        self._parts = iter(parts)
        self._current: IO[bytes] | None = None

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        view = memoryview(buffer).cast("B")
        total = 0
        while view:
            if self._current is None:
                try:
                    self._current = next(self._parts).open("rb")
                except StopIteration:
                    break
            chunk = self._current.read(len(view))
            if chunk:
                view[: len(chunk)] = chunk
                total += len(chunk)
                view = view[len(chunk) :]
                continue
            self._current.close()
            self._current = None
        return total

    def close(self) -> None:
        if self._current is not None:
            self._current.close()
            self._current = None
        super().close()


def safe_extract_zip(archive: Path, destination: Path) -> Path:
    archives = (Path(archive),)
    marker, state = _extraction_marker(destination, archives, None)
    if destination.is_dir() and _marker_matches(marker, state):
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            unix_mode = member.external_attr >> 16
            if stat.S_ISLNK(unix_mode):
                raise UnsafeArchiveError(f"Symbolic links are not allowed: {member.filename!r}")
            target = _safe_member_path(
                destination,
                member.filename,
                resolved_root=resolved_destination,
            )
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                with bundle.open(member) as source:
                    _copy_member(source, target)
    _mark_complete(marker, state)
    return destination


def _extraction_marker(
    destination: Path,
    archives: tuple[Path, ...],
    strip_prefix: str | None,
) -> tuple[Path, str]:
    state = canonical_json(
        {
            "archives": [
                {
                    "path": str(archive.resolve()),
                    "identity": FileIdentity.from_stat(archive.stat()).as_dict(),
                }
                for archive in archives
            ],
            "strip_prefix": strip_prefix,
        }
    )
    directory = destination.parent / f".{destination.name}.lrac-extract"
    return directory / f"{archives[0].name}.json", state


def _marker_matches(marker: Path, state: str) -> bool:
    try:
        return marker.read_text(encoding="utf-8").strip() == state
    except (OSError, UnicodeError):
        return False


def _mark_complete(marker: Path, state: str) -> None:
    if marker.parent.is_symlink():
        raise UnsafeArchiveError(f"Extraction state directory is a symlink: {marker.parent}")
    atomic_write_text(marker, f"{state}\n")


def unsplit_zip(parts_dir: Path, archive_name: str, destination: Path) -> Path:
    """Convert a PKZIP split archive into a regular zip with the system zip tool."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    prefix = Path(archive_name).with_suffix("").name
    parts = [*sorted(parts_dir.glob(f"{prefix}.z[0-9][0-9]")), parts_dir / archive_name]
    if destination.is_file():
        destination_mtime = destination.stat().st_mtime_ns
        if zipfile.is_zipfile(destination) and all(
            part.stat().st_mtime_ns <= destination_mtime for part in parts
        ):
            return destination
        destination.unlink()
    partial = destination.with_name(f".{destination.stem}.part.zip")
    partial.unlink(missing_ok=True)
    try:
        subprocess.run(
            ["zip", "-s", "0", archive_name, "--out", str(partial.resolve())],
            cwd=parts_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("The 'zip' executable is required for FSD50K") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Could not join split zip {archive_name}: {exc.stderr}") from exc
    partial.replace(destination)
    return destination
