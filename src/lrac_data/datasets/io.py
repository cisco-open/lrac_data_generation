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
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import IO, Any, Protocol, cast

from lrac_data.state import FileIdentity, atomic_write_text, canonical_json, fingerprint


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


class Downloader(Protocol):
    """Callable accepted by :func:`download_many` for adapter testability."""

    def __call__(
        self,
        url: str,
        destination: Path,
        *,
        checksum: str | None = None,
    ) -> Path: ...


_DOWNLOAD_STATE_VERSION = 2
_PARTIAL_STATE_VERSION = 1
EXTRACTION_FORMAT_VERSION = 1
_CONTENT_RANGE = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", re.IGNORECASE)


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


def file_checksum(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    derived_state = _read_derived_state(path)
    if derived_state is not None:
        digest = _derived_state_sha256(path, derived_state)
        if digest is not None:
            return digest
    return file_checksum(path)


def verify_checksum(path: Path, checksum: str | None) -> None:
    if not checksum:
        return
    algorithm, expected = _checksum_parts(checksum)
    actual = file_checksum(path, algorithm)
    if actual != expected:
        raise ChecksumError(
            f"Checksum mismatch for {path}: expected {algorithm}:{expected}, "
            f"got {algorithm}:{actual}"
        )


def download_file(
    url: str,
    destination: Path,
    *,
    checksum: str | None = None,
    attempts: int = 4,
    timeout: float = 60.0,
) -> Path:
    """Stream a URL atomically and retain its already-computed SHA-256.

    A partial response is resumed only when the server confirms the requested
    byte range.  Servers that ignore ``Range`` and return 200 are handled as a
    clean restart of the partial file.
    """

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if attempts < 1:
        raise ValueError("attempts must be at least 1")

    normalized_checksum = _normalized_checksum(checksum)
    configured_algorithm = (
        _checksum_parts(normalized_checksum)[0] if normalized_checksum is not None else None
    )
    state_path = _download_state_path(destination)

    if destination.is_symlink():
        destination.unlink()
        state_path.unlink(missing_ok=True)
    elif destination.is_file() and destination.stat().st_size > 0:
        state = _read_download_state(destination)
        if (
            state is not None
            and _download_state_matches_file(destination, state)
            and state.get("url") == url
            and state.get("checksum") == normalized_checksum
            and _state_sha256(state) is not None
        ):
            return destination
        # Migrate an existing cache with one read.  A changed URL without a
        # checksum cannot establish that it still names the same bytes.
        if not (normalized_checksum is None and state is not None and state.get("url") != url):
            digests = _file_digests(destination, configured_algorithm)
            try:
                _verify_digest(destination, normalized_checksum, digests)
            except ChecksumError:
                destination.unlink()
                state_path.unlink(missing_ok=True)
            else:
                _write_download_state(
                    destination,
                    url=url,
                    checksum=normalized_checksum,
                    sha256=digests["sha256"].hexdigest(),
                )
                return destination
        else:
            destination.unlink()
            state_path.unlink(missing_ok=True)
    elif destination.exists():
        destination.unlink()
        state_path.unlink(missing_ok=True)

    partial = destination.with_name(f"{destination.name}.part")
    partial_state_path = _partial_state_path(partial)
    if partial.is_symlink() or (partial.exists() and not partial.is_file()):
        partial.unlink()
        partial_state_path.unlink(missing_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        if partial.exists() and not _partial_can_resume(
            partial,
            url=url,
            checksum=normalized_checksum,
        ):
            partial.unlink()
            partial_state_path.unlink(missing_ok=True)
        offset = partial.stat().st_size if partial.exists() else 0
        digests = (
            _file_digests(partial, configured_algorithm)
            if offset
            else _new_digests(configured_algorithm)
        )
        headers = {"User-Agent": "lrac-data/2026"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
            validator = _partial_validator(partial, url=url)
            if validator is not None:
                headers["If-Range"] = validator
        request = urllib.request.Request(url, headers=headers)
        try:
            with closing(urllib.request.urlopen(request, timeout=timeout)) as response:
                status = getattr(response, "status", response.getcode())
                append = bool(offset and status == 206)
                response_range = _response_range(response)
                if status == 206 and (response_range is None or response_range[0] != offset):
                    partial.unlink(missing_ok=True)
                    partial_state_path.unlink(missing_ok=True)
                    raise OSError(f"Server returned an invalid Content-Range while resuming {url}")
                previous_validator = _partial_validator(partial, url=url) if append else None
                response_validator = _response_validator(response)
                if (
                    append
                    and previous_validator is not None
                    and response_validator is not None
                    and response_validator != previous_validator
                ):
                    partial.unlink(missing_ok=True)
                    partial_state_path.unlink(missing_ok=True)
                    raise OSError(f"Server changed entity validators while resuming {url}")
                if not append:
                    digests = _new_digests(configured_algorithm)
                    offset = 0
                expected_total = _response_total(response, response_range)
                _write_partial_state(
                    partial,
                    url=url,
                    validator=response_validator,
                    total_size=expected_total,
                )
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
            _verify_digest(partial, normalized_checksum, final_digests)
            partial.replace(destination)
            partial_state_path.unlink(missing_ok=True)
            _write_download_state(
                destination,
                url=url,
                checksum=normalized_checksum,
                sha256=final_digests["sha256"],
            )
            return destination
        except ChecksumError as exc:
            last_error = exc
            partial.unlink(missing_ok=True)
            partial_state_path.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 8))
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 416:
                partial.unlink(missing_ok=True)
                partial_state_path.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 8))
    assert last_error is not None
    raise RuntimeError(f"Failed to download {url} after {attempts} attempts") from last_error


def _normalized_checksum(checksum: str | None) -> str | None:
    if checksum is None:
        return None
    algorithm, expected = _checksum_parts(checksum)
    return f"{algorithm}:{expected}"


def _new_digests(configured_algorithm: str | None) -> dict[str, Any]:
    digests: dict[str, Any] = {"sha256": hashlib.sha256()}
    if configured_algorithm is not None and configured_algorithm != "sha256":
        digests[configured_algorithm] = hashlib.new(configured_algorithm)
    return digests


def _file_digests(path: Path, configured_algorithm: str | None) -> dict[str, Any]:
    digests = _new_digests(configured_algorithm)
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            for digest in digests.values():
                digest.update(chunk)
    return digests


def _verify_digest(
    path: Path,
    checksum: str | None,
    digests: dict[str, Any],
) -> None:
    if checksum is None:
        return
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


def _response_validator(response: Any) -> str | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    etag = headers.get("ETag")
    if isinstance(etag, str) and etag and not etag.startswith("W/"):
        return etag
    last_modified = headers.get("Last-Modified")
    return last_modified if isinstance(last_modified, str) and last_modified else None


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


def _partial_state_path(partial: Path) -> Path:
    return partial.with_name(f"{partial.name}.download.json")


def _read_partial_state(partial: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(_partial_state_path(partial).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_partial_state(
    partial: Path,
    *,
    url: str,
    validator: str | None,
    total_size: int | None,
) -> None:
    state = {
        "schema_version": _PARTIAL_STATE_VERSION,
        "url": url,
        "validator": validator,
        "total_size": total_size,
    }
    atomic_write_text(_partial_state_path(partial), f"{canonical_json(state)}\n")


def _partial_validator(partial: Path, *, url: str) -> str | None:
    state = _read_partial_state(partial)
    if (
        state is None
        or state.get("schema_version") != _PARTIAL_STATE_VERSION
        or state.get("url") != url
    ):
        return None
    validator = state.get("validator")
    return validator if isinstance(validator, str) and validator else None


def _partial_can_resume(partial: Path, *, url: str, checksum: str | None) -> bool:
    if not partial.is_file() or partial.stat().st_size == 0:
        return False
    return checksum is not None or _partial_validator(partial, url=url) is not None


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
    if identity is None:
        identity = FileIdentity.from_dict(state)
    return bool(
        state.get("schema_version") == _DOWNLOAD_STATE_VERSION
        and identity is not None
        and identity.matches_stat(current)
    )


def _state_sha256(state: dict[str, Any]) -> str | None:
    digest = state.get("sha256")
    if isinstance(digest, str) and _is_hex_digest(digest, 64):
        return digest
    return None


def _write_download_state(
    destination: Path,
    *,
    url: str,
    checksum: str | None,
    sha256: str,
) -> None:
    current = destination.stat()
    state = {
        "schema_version": _DOWNLOAD_STATE_VERSION,
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
    downloader: Downloader = download_file,
) -> list[Path]:
    """Download independent artifacts with a small, bounded worker pool.

    At most ``max_workers`` requests are submitted at once.  Results retain input
    order, and a failed request prevents any further queued work from starting.
    Already-running downloads finish normally so their resumable partial files are
    left in a consistent state.
    """

    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")

    ordered = tuple(requests)
    destinations = [Path(request.destination) for request in ordered]
    if len(destinations) != len(set(destinations)):
        raise ValueError("download destinations must be unique")
    if not ordered:
        return []

    worker_count = min(max_workers, len(ordered))
    results: list[Path | None] = [None] * len(ordered)
    executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="lrac-download")
    futures: dict[Future[Path], int] = {}
    next_index = 0

    def submit_one(index: int) -> Future[Path]:
        request = ordered[index]
        return executor.submit(
            downloader,
            request.url,
            request.destination,
            checksum=request.checksum,
        )

    try:
        while next_index < worker_count:
            futures[submit_one(next_index)] = next_index
            next_index += 1

        while futures:
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in sorted(completed, key=futures.__getitem__):
                index = futures.pop(future)
                results[index] = future.result()

            while next_index < len(ordered) and len(futures) < worker_count:
                futures[submit_one(next_index)] = next_index
                next_index += 1
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    return [cast(Path, result) for result in results]


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
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as bundle:
        _extract_tar_members(bundle, destination, strip_prefix=strip_prefix)
    return destination


def safe_extract_multipart_tar(
    parts: Iterable[Path],
    destination: Path,
    *,
    strip_prefix: str | None = None,
) -> Path:
    """Safely stream an ordered multipart ``tar.gz`` into ``destination``."""

    ordered_parts = tuple(Path(part) for part in parts)
    if not ordered_parts:
        raise ValueError("multipart extraction requires at least one part")
    destination.mkdir(parents=True, exist_ok=True)
    with (
        _MultipartReader(ordered_parts) as stream,
        tarfile.open(fileobj=stream, mode="r|gz") as bundle,
    ):
        _extract_tar_members(bundle, destination, strip_prefix=strip_prefix)
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
    return destination


def unsplit_zip(parts_dir: Path, archive_name: str, destination: Path) -> Path:
    """Convert a PKZIP split archive into a regular zip with the system zip tool."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    archive_path = parts_dir / archive_name
    prefix = Path(archive_name).with_suffix("").name
    parts = [*sorted(parts_dir.glob(f"{prefix}.z[0-9][0-9]")), archive_path]
    input_fingerprint = fingerprint([(part.name, trusted_file_sha256(part)) for part in parts])
    state_path = _derived_state_path(destination)
    if destination.is_file():
        if zipfile.is_zipfile(destination) and _derived_output_matches(
            destination, state_path, input_fingerprint
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
    _write_derived_state(destination, state_path, input_fingerprint)
    return destination


def remove_derived_archive(path: Path) -> None:
    """Remove a reproducible archive and its input-fingerprint sidecar."""

    path.unlink(missing_ok=True)
    _derived_state_path(path).unlink(missing_ok=True)


def _derived_state_path(destination: Path) -> Path:
    return destination.with_suffix(f"{destination.suffix}.inputs.json")


def _derived_output_matches(
    destination: Path,
    state_path: Path,
    expected_input_fingerprint: str,
) -> bool:
    state = _read_derived_state(destination, state_path=state_path)
    if state is None:
        return False
    return bool(
        state.get("schema_version") == 2
        and state.get("input_fingerprint") == expected_input_fingerprint
        and _derived_state_sha256(destination, state) is not None
    )


def _write_derived_state(
    destination: Path,
    state_path: Path,
    input_fingerprint: str,
) -> None:
    output_sha256 = file_checksum(destination)
    current = destination.stat()
    state = {
        "schema_version": 2,
        "input_fingerprint": input_fingerprint,
        "output_sha256": output_sha256,
        "output_identity": FileIdentity.from_stat(current).as_dict(),
    }
    atomic_write_text(
        state_path,
        f"{canonical_json(state)}\n",
    )


def _read_derived_state(
    destination: Path,
    *,
    state_path: Path | None = None,
) -> dict[str, Any] | None:
    try:
        state = json.loads(
            (state_path or _derived_state_path(destination)).read_text(encoding="utf-8")
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return state if isinstance(state, dict) else None


def _derived_state_sha256(destination: Path, state: dict[str, Any]) -> str | None:
    digest = state.get("output_sha256")
    if not isinstance(digest, str) or not _is_hex_digest(digest, 64):
        return None
    try:
        current = destination.stat()
    except OSError:
        return None
    identity = FileIdentity.from_dict(state.get("output_identity"))
    if identity is None:
        identity = FileIdentity.from_dict(
            {
                "size": state.get("output_size"),
                "mtime_ns": state.get("output_mtime_ns"),
                "ctime_ns": state.get("output_ctime_ns"),
                "device": state.get("output_device"),
                "inode": state.get("output_inode"),
            }
        )
    if state.get("schema_version") != 2 or identity is None or not identity.matches_stat(current):
        return None
    return digest
