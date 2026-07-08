"""Durable, fingerprinted state for resumable preparation runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    """Return a stable JSON representation suitable for hashing."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    """Write a file in-place only after its complete contents are durable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as destination:
            destination.write(text)
            destination.flush()
            os.fsync(destination.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@dataclass(frozen=True, slots=True)
class FileIdentity:
    """Cheap identity for deciding whether a cached digest still describes a file."""

    size: int
    mtime_ns: int
    ctime_ns: int
    device: int
    inode: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> FileIdentity:
        return cls(
            size=value.st_size,
            mtime_ns=value.st_mtime_ns,
            ctime_ns=value.st_ctime_ns,
            device=value.st_dev,
            inode=value.st_ino,
        )

    @classmethod
    def from_dict(cls, value: object) -> FileIdentity | None:
        if not isinstance(value, Mapping):
            return None
        size = value.get("size")
        mtime_ns = value.get("mtime_ns")
        ctime_ns = value.get("ctime_ns")
        device = value.get("device")
        inode = value.get("inode")
        if not all(type(field) is int for field in (size, mtime_ns, ctime_ns, device, inode)):
            return None
        assert isinstance(size, int)
        assert isinstance(mtime_ns, int)
        assert isinstance(ctime_ns, int)
        assert isinstance(device, int)
        assert isinstance(inode, int)
        return cls(size, mtime_ns, ctime_ns, device, inode)

    def as_dict(self) -> dict[str, int]:
        return asdict(self)

    def matches_stat(self, value: os.stat_result) -> bool:
        return self == self.from_stat(value)


@dataclass(frozen=True)
class StageState:
    key: str
    status: str
    fingerprint: str
    started_at: str
    finished_at: str | None = None
    outputs: Mapping[str, str] | None = None
    error: str | None = None


class StateStore:
    """Persist stage state under ``workspace/runs/<run>/state``."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _path(self, key: str) -> Path:
        safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "-", key).strip("-")
        if not safe_key:
            raise ValueError("stage key must contain at least one safe character")
        return self.directory / f"{safe_key}.json"

    def read(self, key: str) -> StageState | None:
        path = self._path(key)
        if not path.exists():
            return None
        return StageState(**json.loads(path.read_text(encoding="utf-8")))

    def is_complete(
        self,
        key: str,
        expected_fingerprint: str,
        *,
        verify_outputs: bool = True,
        verify_paths: Iterable[Path] | None = None,
    ) -> bool:
        state = self.read(key)
        if state is None or state.status != "complete":
            return False
        if state.fingerprint != expected_fingerprint:
            return False
        if verify_outputs:
            outputs = state.outputs or {}
            candidates: Iterable[tuple[str, str | None]]
            if verify_paths is None:
                candidates = outputs.items()
            else:
                requested = (str(path.resolve()) for path in verify_paths)
                candidates = ((filename, outputs.get(filename)) for filename in requested)
            for filename, digest in candidates:
                if digest is None:
                    return False
                path = Path(filename)
                if not path.is_file() or sha256_file(path) != digest:
                    return False
        return True

    def mark_running(self, key: str, stage_fingerprint: str) -> StageState:
        state = StageState(
            key=key,
            status="running",
            fingerprint=stage_fingerprint,
            started_at=_now(),
        )
        self._write(state)
        return state

    def mark_complete(
        self,
        key: str,
        stage_fingerprint: str,
        outputs: list[Path],
        *,
        started_at: str | None = None,
        known_digests: Mapping[Path, str] | None = None,
    ) -> StageState:
        known = {str(path.resolve()): digest for path, digest in (known_digests or {}).items()}
        output_digests: dict[str, str] = {}
        for path in outputs:
            filename = str(path.resolve())
            output_digests[filename] = known.get(filename) or sha256_file(path)
        state = StageState(
            key=key,
            status="complete",
            fingerprint=stage_fingerprint,
            started_at=started_at or _now(),
            finished_at=_now(),
            outputs=output_digests,
        )
        self._write(state)
        return state

    def mark_failed(
        self,
        key: str,
        stage_fingerprint: str,
        error: BaseException | str,
        *,
        started_at: str | None = None,
    ) -> StageState:
        state = StageState(
            key=key,
            status="failed",
            fingerprint=stage_fingerprint,
            started_at=started_at or _now(),
            finished_at=_now(),
            error=str(error),
        )
        self._write(state)
        return state

    def all(self) -> list[StageState]:
        if not self.directory.exists():
            return []
        return [
            StageState(**json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(self.directory.glob("*.json"))
        ]

    def _write(self, state: StageState) -> None:
        atomic_write_text(self._path(state.key), f"{canonical_json(asdict(state))}\n")


def environment_provenance(repo_root: Path) -> dict[str, Any]:
    """Collect inexpensive provenance without importing optional audio packages."""

    return {
        "git_sha": _command_output(["git", "rev-parse", "HEAD"], cwd=repo_root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "ffmpeg": _command_output(["ffmpeg", "-version"], first_line=True),
        "git": _command_output(["git", "--version"], first_line=True),
        "zip": _command_output(["zip", "-v"], first_line=True),
    }


def _command_output(
    command: list[str], *, cwd: Path | None = None, first_line: bool = False
) -> str | None:
    try:
        output = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return output.splitlines()[0] if first_line and output else output


def _now() -> str:
    return datetime.now(UTC).isoformat()
