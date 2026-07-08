from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import lrac_data.datasets.io as dataset_io
from lrac_data.datasets.io import (
    ChecksumError,
    DownloadRequest,
    UnsafeArchiveError,
    download_file,
    download_many,
    remove_derived_archive,
    safe_extract_multipart_tar,
    safe_extract_tar,
    safe_extract_zip,
    trusted_file_sha256,
    verify_checksum,
)


def _tar_gz_bytes(files: dict[str, bytes]) -> bytes:
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as bundle:
        for name, payload in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            bundle.addfile(member, io.BytesIO(payload))
    return archive.getvalue()


class _FakeResponse(io.BytesIO):
    def __init__(
        self,
        payload: bytes,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(payload)
        self.status = status
        self.headers = headers or {}

    def getcode(self) -> int:
        return self.status


def test_importing_dataset_registry_creates_no_files(tmp_path: Path) -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    subprocess.run(
        [sys.executable, "-c", "import lrac_data.datasets"],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert list(tmp_path.iterdir()) == []


def test_zip_extraction_rejects_parent_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.txt", "not allowed")

    with pytest.raises(UnsafeArchiveError, match="Unsafe archive member"):
        safe_extract_zip(archive, tmp_path / "extracted")

    assert not (tmp_path / "escape.txt").exists()


def test_tar_extraction_rejects_parent_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar"
    payload = b"not allowed"
    with tarfile.open(archive, "w") as bundle:
        member = tarfile.TarInfo("../escape.txt")
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))

    with pytest.raises(UnsafeArchiveError, match="Unsafe archive member"):
        safe_extract_tar(archive, tmp_path / "extracted")

    assert not (tmp_path / "escape.txt").exists()


def test_checksum_verification_accepts_known_digests_and_rejects_mismatch(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "fixture.bin"
    fixture.write_bytes(b"abc")

    verify_checksum(fixture, "md5:900150983cd24fb0d6963f7d28e17f72")
    verify_checksum(
        fixture,
        "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    )
    with pytest.raises(ChecksumError, match="Checksum mismatch"):
        verify_checksum(fixture, "sha256:" + "0" * 64)


def test_download_streams_to_partial_file_then_publishes_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"fixture archive bytes"
    checksum = "sha256:" + hashlib.sha256(payload).hexdigest()

    class FakeResponse(io.BytesIO):
        status = 200

        def getcode(self) -> int:
            return self.status

    calls: list[str] = []

    def fake_urlopen(request: object, *, timeout: float) -> FakeResponse:
        calls.append(request.full_url)  # type: ignore[attr-defined]
        assert timeout == 1.0
        return FakeResponse(payload)

    monkeypatch.setattr("lrac_data.datasets.io.urllib.request.urlopen", fake_urlopen)
    destination = tmp_path / "downloads" / "fixture.zip"

    result = download_file(
        "https://example.invalid/fixture.zip",
        destination,
        checksum=checksum,
        attempts=1,
        timeout=1.0,
    )

    assert result == destination
    assert destination.read_bytes() == payload
    assert not destination.with_name("fixture.zip.part").exists()
    state = json.loads(destination.with_name("fixture.zip.download.json").read_text())
    assert state["sha256"] == hashlib.sha256(payload).hexdigest()
    assert state["checksum"] == checksum
    assert state["identity"]["size"] == len(payload)
    assert set(state["identity"]) == {"size", "mtime_ns", "ctime_ns", "device", "inode"}
    assert calls == ["https://example.invalid/fixture.zip"]


def test_download_reuses_matching_stat_url_and_checksum_without_reading_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"cached archive"
    checksum = "md5:" + hashlib.md5(payload).hexdigest()
    destination = tmp_path / "cached.tar"
    monkeypatch.setattr(
        "lrac_data.datasets.io.urllib.request.urlopen",
        lambda *_args, **_kwargs: _FakeResponse(payload),
    )

    download_file("https://example.invalid/cached.tar", destination, checksum=checksum)
    state_path = destination.with_name("cached.tar.download.json")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update(state.pop("identity"))
    state_path.write_text(json.dumps(state), encoding="utf-8")

    def fail(*_args: object, **_kwargs: object) -> None:
        pytest.fail("matching download cache must not be read or requested again")

    monkeypatch.setattr("lrac_data.datasets.io.urllib.request.urlopen", fail)
    monkeypatch.setattr("lrac_data.datasets.io.file_checksum", fail)
    assert trusted_file_sha256(destination) == hashlib.sha256(payload).hexdigest()
    assert (
        download_file(
            "https://example.invalid/cached.tar",
            destination,
            checksum=checksum,
        )
        == destination
    )


def test_trusted_sha256_falls_back_after_download_stat_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = b"original"
    changed = b"modified"
    destination = tmp_path / "fixture.bin"
    monkeypatch.setattr(
        "lrac_data.datasets.io.urllib.request.urlopen",
        lambda *_args, **_kwargs: _FakeResponse(original),
    )
    download_file("https://example.invalid/fixture.bin", destination)
    previous = destination.stat()

    replacement = destination.with_suffix(".replacement")
    replacement.write_bytes(changed)
    os.utime(replacement, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    replacement.replace(destination)

    assert trusted_file_sha256(destination) == hashlib.sha256(changed).hexdigest()


def test_download_resumes_only_from_confirmed_content_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"complete response body"
    offset = 9
    destination = tmp_path / "archive.tar"
    partial = destination.with_name("archive.tar.part")
    partial.write_bytes(payload[:offset])

    def fake_urlopen(request: object, *, timeout: float) -> _FakeResponse:
        del timeout
        assert request.get_header("Range") == f"bytes={offset}-"  # type: ignore[attr-defined]
        return _FakeResponse(
            payload[offset:],
            status=206,
            headers={"Content-Range": f"bytes {offset}-{len(payload) - 1}/{len(payload)}"},
        )

    monkeypatch.setattr("lrac_data.datasets.io.urllib.request.urlopen", fake_urlopen)

    download_file(
        "https://example.invalid/archive.tar",
        destination,
        checksum="sha256:" + hashlib.sha256(payload).hexdigest(),
        attempts=1,
    )

    assert destination.read_bytes() == payload


def test_checksum_free_download_resumes_only_with_entity_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"complete response body"
    offset = 9
    destination = tmp_path / "archive.tar"

    class InterruptedResponse(_FakeResponse):
        def __init__(self) -> None:
            super().__init__(
                payload,
                headers={"ETag": '"fixture-v1"', "Content-Length": str(len(payload))},
            )
            self.returned = False

        def read(self, size: int = -1) -> bytes:
            del size
            if not self.returned:
                self.returned = True
                return payload[:offset]
            raise OSError("connection interrupted")

    monkeypatch.setattr(
        "lrac_data.datasets.io.urllib.request.urlopen",
        lambda *_args, **_kwargs: InterruptedResponse(),
    )
    with pytest.raises(RuntimeError, match="Failed to download"):
        download_file("https://example.invalid/archive.tar", destination, attempts=1)

    def resume(request: object, *, timeout: float) -> _FakeResponse:
        del timeout
        assert request.get_header("Range") == f"bytes={offset}-"  # type: ignore[attr-defined]
        assert request.get_header("If-range") == '"fixture-v1"'  # type: ignore[attr-defined]
        return _FakeResponse(
            payload[offset:],
            status=206,
            headers={
                "ETag": '"fixture-v1"',
                "Content-Length": str(len(payload) - offset),
                "Content-Range": f"bytes {offset}-{len(payload) - 1}/{len(payload)}",
            },
        )

    monkeypatch.setattr("lrac_data.datasets.io.urllib.request.urlopen", resume)

    download_file("https://example.invalid/archive.tar", destination, attempts=1)

    assert destination.read_bytes() == payload
    assert not destination.with_name("archive.tar.part.download.json").exists()


def test_download_does_not_publish_incomplete_ranged_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"complete response body"
    offset = 5
    destination = tmp_path / "archive.tar"
    destination.with_name("archive.tar.part").write_bytes(payload[:offset])

    monkeypatch.setattr(
        "lrac_data.datasets.io.urllib.request.urlopen",
        lambda *_args, **_kwargs: _FakeResponse(
            payload[offset : offset + 4],
            status=206,
            headers={"Content-Range": f"bytes {offset}-{offset + 3}/{len(payload)}"},
        ),
    )

    with pytest.raises(RuntimeError, match="Failed to download"):
        download_file(
            "https://example.invalid/archive.tar",
            destination,
            checksum="sha256:" + hashlib.sha256(payload).hexdigest(),
            attempts=1,
        )

    assert not destination.exists()


def test_download_restarts_partial_when_server_ignores_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"full response"
    destination = tmp_path / "archive.tar"
    destination.with_name("archive.tar.part").write_bytes(b"stale prefix")

    monkeypatch.setattr(
        "lrac_data.datasets.io.urllib.request.urlopen",
        lambda *_args, **_kwargs: _FakeResponse(payload, status=200),
    )

    download_file(
        "https://example.invalid/archive.tar",
        destination,
        checksum="sha256:" + hashlib.sha256(payload).hexdigest(),
        attempts=1,
    )

    assert destination.read_bytes() == payload


def test_download_discards_unconfirmed_range_before_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"complete response body"
    destination = tmp_path / "archive.tar"
    destination.with_name("archive.tar.part").write_bytes(payload[:5])
    calls = 0

    def fake_urlopen(request: object, *, timeout: float) -> _FakeResponse:
        del request, timeout
        nonlocal calls
        calls += 1
        if calls == 1:
            return _FakeResponse(payload[5:], status=206, headers={"Content-Range": "bytes 0-9/22"})
        return _FakeResponse(payload)

    monkeypatch.setattr("lrac_data.datasets.io.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("lrac_data.datasets.io.time.sleep", lambda _seconds: None)

    download_file(
        "https://example.invalid/archive.tar",
        destination,
        checksum="sha256:" + hashlib.sha256(payload).hexdigest(),
        attempts=2,
    )

    assert calls == 2
    assert destination.read_bytes() == payload


def test_download_without_checksum_refetches_when_url_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = iter((b"first", b"second"))
    monkeypatch.setattr(
        "lrac_data.datasets.io.urllib.request.urlopen",
        lambda *_args, **_kwargs: _FakeResponse(next(responses)),
    )
    destination = tmp_path / "mutable.bin"

    download_file("https://example.invalid/first", destination)
    download_file("https://example.invalid/second", destination)

    assert destination.read_bytes() == b"second"


def test_download_checksum_failure_publishes_neither_file_nor_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "corrupt.tar"
    monkeypatch.setattr(
        "lrac_data.datasets.io.urllib.request.urlopen",
        lambda *_args, **_kwargs: _FakeResponse(b"unexpected bytes"),
    )

    with pytest.raises(RuntimeError, match="Failed to download"):
        download_file(
            "https://example.invalid/corrupt.tar",
            destination,
            checksum="sha256:" + "0" * 64,
            attempts=1,
        )

    assert not destination.exists()
    assert not destination.with_name("corrupt.tar.part").exists()
    assert not destination.with_name("corrupt.tar.download.json").exists()


def test_download_many_is_bounded_and_returns_destination_order(tmp_path: Path) -> None:
    release = threading.Event()
    first_batch_started = threading.Event()
    lock = threading.Lock()
    started: list[str] = []
    active = 0
    peak_active = 0

    requests = [
        DownloadRequest(
            url=f"https://example.invalid/{index}",
            destination=tmp_path / f"{index}.zip",
        )
        for index in range(8)
    ]

    def controlled_download(
        url: str,
        destination: Path,
        *,
        checksum: str | None = None,
    ) -> Path:
        del url, checksum
        nonlocal active, peak_active
        with lock:
            active += 1
            peak_active = max(peak_active, active)
            started.append(destination.name)
            if len(started) == 4:
                first_batch_started.set()
        if not release.wait(timeout=2):
            raise TimeoutError("test did not release downloads")
        with lock:
            active -= 1
        return destination.with_suffix(".complete")

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(download_many, requests, downloader=controlled_download)
        assert first_batch_started.wait(timeout=2)
        with lock:
            assert len(started) == 4
            assert set(started) == {"0.zip", "1.zip", "2.zip", "3.zip"}
        release.set()
        results = future.result(timeout=2)

    assert peak_active == 4
    assert results == [request.destination.with_suffix(".complete") for request in requests]


def test_download_many_stops_before_starting_later_requests_after_failure(
    tmp_path: Path,
) -> None:
    requests = [
        DownloadRequest(
            url=f"https://example.invalid/{index}",
            destination=tmp_path / f"{index}.zip",
        )
        for index in range(3)
    ]
    started: list[Path] = []

    def failing_download(
        url: str,
        destination: Path,
        *,
        checksum: str | None = None,
    ) -> Path:
        del url, checksum
        started.append(destination)
        raise RuntimeError("download failed")

    with pytest.raises(RuntimeError, match="download failed"):
        download_many(requests, max_workers=1, downloader=failing_download)

    assert started == [requests[0].destination]


def test_multipart_tar_streams_across_part_boundaries_and_strips_prefix(
    tmp_path: Path,
) -> None:
    payload = _tar_gz_bytes(
        {
            "dataset/audio/first.wav": b"first audio",
            "dataset/audio/nested/second.wav": b"second audio",
            "ignored/readme.txt": b"outside prefix",
        }
    )
    cut_one = 7
    cut_two = len(payload) // 2
    parts = [tmp_path / f"archive.part{suffix}" for suffix in ("aa", "ab", "ac")]
    parts[0].write_bytes(payload[:cut_one])
    parts[1].write_bytes(payload[cut_one:cut_two])
    parts[2].write_bytes(payload[cut_two:])
    destination = tmp_path / "extracted"

    safe_extract_multipart_tar(parts, destination, strip_prefix="dataset/audio")

    assert (destination / "first.wav").read_bytes() == b"first audio"
    assert (destination / "nested" / "second.wav").read_bytes() == b"second audio"
    assert not (destination / "ignored").exists()


def test_multipart_tar_rejects_links(tmp_path: Path) -> None:
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as bundle:
        link = tarfile.TarInfo("outside-link")
        link.type = tarfile.SYMTYPE
        link.linkname = "../outside"
        bundle.addfile(link)
    payload = archive.getvalue()
    parts = [tmp_path / "archive.partaa", tmp_path / "archive.partab"]
    parts[0].write_bytes(payload[:11])
    parts[1].write_bytes(payload[11:])
    destination = tmp_path / "extracted"

    with pytest.raises(UnsafeArchiveError, match="Links and device members"):
        safe_extract_multipart_tar(parts, destination)


def test_changed_archive_reextracts(tmp_path: Path) -> None:
    archive = tmp_path / "archive.tar"
    destination = tmp_path / "extracted"

    def write_archive(payload: bytes) -> None:
        with tarfile.open(archive, "w") as bundle:
            member = tarfile.TarInfo("payload.txt")
            member.size = len(payload)
            bundle.addfile(member, io.BytesIO(payload))

    write_archive(b"first")
    safe_extract_tar(archive, destination)
    write_archive(b"second")
    safe_extract_tar(archive, destination)

    assert (destination / "payload.txt").read_bytes() == b"second"


def test_atomic_member_copy_preserves_target_and_removes_partial_on_failure(
    tmp_path: Path,
) -> None:
    target = tmp_path / "payload.bin"
    target.write_bytes(b"previous complete file")

    class FailingSource(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            del size
            raise OSError("interrupted source")

    with pytest.raises(OSError, match="interrupted source"):
        dataset_io._copy_member(FailingSource(), target)

    assert target.read_bytes() == b"previous complete file"
    assert not target.with_name("payload.bin.part").exists()


def test_atomic_member_copy_does_not_follow_partial_symlink(tmp_path: Path) -> None:
    target = tmp_path / "payload.bin"
    sentinel = tmp_path / "sentinel.bin"
    sentinel.write_bytes(b"keep")
    target.with_name("payload.bin.part").symlink_to(sentinel)

    dataset_io._copy_member(io.BytesIO(b"archive payload"), target)

    assert sentinel.read_bytes() == b"keep"
    assert target.read_bytes() == b"archive payload"


def test_remove_derived_archive_preserves_upstream_parts(tmp_path: Path) -> None:
    upstream = tmp_path / "archive.zip"
    derived = tmp_path / "archive.unsplit.zip"
    state = derived.with_suffix(f"{derived.suffix}.inputs.json")
    upstream.write_bytes(b"immutable upstream")
    derived.write_bytes(b"reproducible derived archive")
    state.write_text("{}\n", encoding="utf-8")

    remove_derived_archive(derived)

    assert upstream.read_bytes() == b"immutable upstream"
    assert not derived.exists()
    assert not state.exists()
