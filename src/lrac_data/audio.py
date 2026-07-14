"""Audio inspection and deterministic PCM16 WAV materialization."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat as stat_module
import subprocess
import threading
from collections import OrderedDict
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from pathlib import Path

from .state import FileIdentity, canonical_json, fingerprint, sha256_file

_FFMPEG_BATCH_SIZE = 32
_JOURNAL_NAME = ".lrac-materialization.jsonl"
_JOURNAL_SNAPSHOT_CACHE_SIZE = 32


@dataclass(frozen=True)
class AudioMetadata:
    path: Path
    sample_rate_hz: int
    channels: int
    num_frames: int
    format: str
    subtype: str
    sha256: str
    identity: FileIdentity | None = None
    checksum_fresh: bool = field(default=False, compare=False, repr=False)

    def matches_stat(self, path: Path) -> bool:
        """Return whether ``path`` still has the file identity captured here."""

        if self.path != path or self.identity is None:
            return False
        try:
            current = path.stat()
        except OSError:
            return False
        return self.identity.matches_stat(current)


@dataclass(frozen=True)
class MaterializationTask:
    source: Path
    destination: Path
    sample_rate_hz: int = 24_000
    channels: int = 1
    source_release: str = "unspecified"
    implementation_fingerprint: str = "unspecified"
    source_sha256: str | None = None


@dataclass(frozen=True)
class _PendingMaterialization:
    task: MaterializationTask
    temporary: Path
    expected_fingerprint: str


@dataclass(frozen=True)
class _JournalRecord:
    destination: str
    fingerprint: str
    output_sha256: str
    sample_rate_hz: int
    channels: int
    num_frames: int
    format: str
    subtype: str
    identity: FileIdentity

    @classmethod
    def from_json(cls, value: object) -> _JournalRecord | None:
        if not isinstance(value, dict) or value.get("schema_version") != 3:
            return None
        destination = value.get("destination")
        fingerprint_value = value.get("fingerprint")
        output_sha256 = value.get("output_sha256")
        sample_rate_hz = value.get("sample_rate_hz")
        channels = value.get("channels")
        num_frames = value.get("num_frames")
        format_value = value.get("format", "WAV")
        subtype = value.get("subtype")
        identity = FileIdentity.from_dict(value.get("identity"))
        if identity is None:
            identity = FileIdentity.from_dict(
                {
                    "size": value.get("size_bytes"),
                    "mtime_ns": value.get("mtime_ns"),
                    "ctime_ns": value.get("ctime_ns"),
                    "device": value.get("device"),
                    "inode": value.get("inode"),
                }
            )
        if not (
            isinstance(destination, str)
            and isinstance(fingerprint_value, str)
            and isinstance(output_sha256, str)
            and isinstance(sample_rate_hz, int)
            and isinstance(channels, int)
            and isinstance(num_frames, int)
            and isinstance(format_value, str)
            and isinstance(subtype, str)
            and identity is not None
        ):
            return None
        if not destination or Path(destination).name != destination:
            return None
        return cls(
            destination=destination,
            fingerprint=fingerprint_value,
            output_sha256=output_sha256,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            num_frames=num_frames,
            format=format_value,
            subtype=subtype,
            identity=identity,
        )

    def as_json(self) -> dict[str, object]:
        return {
            "schema_version": 3,
            "destination": self.destination,
            "fingerprint": self.fingerprint,
            "output_sha256": self.output_sha256,
            "sample_rate_hz": self.sample_rate_hz,
            "channels": self.channels,
            "num_frames": self.num_frames,
            "format": self.format,
            "subtype": self.subtype,
            "identity": self.identity.as_dict(),
        }


@dataclass(frozen=True)
class _JournalSnapshot:
    records: dict[str, _JournalRecord]
    valid_bytes: int
    file_identity: FileIdentity | None


_journal_guard = threading.Lock()
_journal_locks: dict[Path, threading.Lock] = {}
_journal_snapshots: OrderedDict[Path, _JournalSnapshot] = OrderedDict()


class _FFmpegBatchError(RuntimeError):
    """Raised when a multi-input FFmpeg command cannot complete."""


def output_path(
    audio_root: Path,
    dataset: str,
    source_id: str,
    *,
    materialization_key: str,
) -> Path:
    """Create a stable, bounded path for a dataset-qualified source item."""

    digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest()
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_id).strip("._")
    safe_id = safe_id[:120] or digest
    return (
        audio_root
        / dataset
        / digest[:2]
        / f"{safe_id}-{digest[:12]}-{materialization_key[:12]}.wav"
    )


def probe(path: Path, *, include_checksum: bool = True) -> AudioMetadata:
    import soundfile as sf

    before = path.stat()
    info = sf.info(str(path))
    checksum = sha256_file(path) if include_checksum else ""
    after = path.stat()
    if FileIdentity.from_stat(before) != FileIdentity.from_stat(after):
        raise RuntimeError(f"audio changed while it was being inspected: {path}")
    return AudioMetadata(
        path=path,
        sample_rate_hz=info.samplerate,
        channels=info.channels,
        num_frames=info.frames,
        format=info.format,
        subtype=info.subtype,
        sha256=checksum,
        identity=FileIdentity.from_stat(after),
        checksum_fresh=include_checksum,
    )


def materialize(task: MaterializationTask) -> AudioMetadata:
    """Convert one source to mono PCM16 WAV and publish it atomically."""

    prepared = _prepare_materialization(task)
    if isinstance(prepared, AudioMetadata):
        return prepared
    return _materialize_pending(prepared)


def _prepare_materialization(
    task: MaterializationTask,
) -> AudioMetadata | _PendingMaterialization:
    if task.channels != 1:
        raise ValueError("LRAC materialization currently supports mono output only")
    if task.sample_rate_hz <= 0:
        raise ValueError("target sample rate must be positive")

    if not task.source.is_file():
        raise FileNotFoundError(task.source)

    destination = task.destination
    expected_fingerprint = _task_fingerprint(task)
    if destination.is_file():
        reusable = _reusable_metadata(
            destination,
            expected_fingerprint=expected_fingerprint,
            sample_rate_hz=task.sample_rate_hz,
            channels=task.channels,
        )
        if reusable is not None:
            return reusable

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp.wav")
    temporary.unlink(missing_ok=True)
    return _PendingMaterialization(
        task=task,
        temporary=temporary,
        expected_fingerprint=expected_fingerprint,
    )


def _materialize_pending(pending: _PendingMaterialization) -> AudioMetadata:
    try:
        if shutil.which("ffmpeg") is not None:
            _materialize_with_ffmpeg(pending.task, pending.temporary)
        else:
            _materialize_with_soundfile(pending.task, pending.temporary)
        written = _verify_temporary(pending)
        return _publish_verified(((pending, written),))[0]
    except BaseException:
        pending.temporary.unlink(missing_ok=True)
        raise


def _verify_temporary(pending: _PendingMaterialization) -> AudioMetadata:
    written = probe(pending.temporary)
    task = pending.task
    if (
        written.sample_rate_hz != task.sample_rate_hz
        or written.channels != task.channels
        or written.format != "WAV"
        or written.subtype != "PCM_16"
    ):
        raise ValueError(f"materialized audio failed validation: {task.source}")
    return written


def _publish_verified(
    verified: Iterable[tuple[_PendingMaterialization, AudioMetadata]],
) -> list[AudioMetadata]:
    """Publish verified files, then durably checkpoint the completed batch."""

    published: list[tuple[_PendingMaterialization, AudioMetadata]] = []
    for pending, written in verified:
        destination = pending.task.destination
        pending.temporary.replace(destination)
        stat = destination.stat()
        metadata = replace(
            written,
            path=destination,
            identity=FileIdentity.from_stat(stat),
        )
        published.append((pending, metadata))

    _append_journal_records(
        (
            pending.task.destination,
            _JournalRecord(
                destination=pending.task.destination.name,
                fingerprint=pending.expected_fingerprint,
                output_sha256=metadata.sha256,
                sample_rate_hz=metadata.sample_rate_hz,
                channels=metadata.channels,
                num_frames=metadata.num_frames,
                format=metadata.format,
                subtype=metadata.subtype,
                identity=metadata.identity
                or FileIdentity.from_stat(pending.task.destination.stat()),
            ),
        )
        for pending, metadata in published
    )
    return [metadata for _pending, metadata in published]


def _state_path(destination: Path) -> Path:
    return destination.parent / _JOURNAL_NAME


def _journal_path(destination: Path) -> Path:
    # Keep the absolute spelling without resolving a possibly planted symlink.
    return Path(os.path.abspath(_state_path(destination)))


def _journal_lock(path: Path) -> threading.Lock:
    with _journal_guard:
        return _journal_locks.setdefault(path, threading.Lock())


def _journal_record(destination: Path) -> _JournalRecord | None:
    journal = _journal_path(destination)
    with _journal_lock(journal):
        return _journal_snapshot(journal).records.get(destination.name)


def _journal_snapshot(path: Path) -> _JournalSnapshot:
    identity = _journal_file_identity(path)
    cached = _cached_journal_snapshot(path, identity)
    if cached is not None:
        return cached

    records: dict[str, _JournalRecord] = {}
    valid_bytes = 0
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as source:
            while line := source.readline():
                if not line.endswith(b"\n"):
                    break
                try:
                    record = _JournalRecord.from_json(json.loads(line))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    break
                if record is None:
                    break
                records[record.destination] = record
                valid_bytes += len(line)
    except FileNotFoundError:
        pass

    snapshot = _JournalSnapshot(
        records=records,
        valid_bytes=valid_bytes,
        file_identity=_journal_file_identity(path),
    )
    _cache_journal_snapshot(path, snapshot)
    return snapshot


def _cached_journal_snapshot(path: Path, identity: FileIdentity | None) -> _JournalSnapshot | None:
    with _journal_guard:
        snapshot = _journal_snapshots.get(path)
        if snapshot is None:
            return None
        if snapshot.file_identity != identity:
            _journal_snapshots.pop(path, None)
            return None
        _journal_snapshots.move_to_end(path)
        return snapshot


def _cache_journal_snapshot(path: Path, snapshot: _JournalSnapshot) -> None:
    with _journal_guard:
        _journal_snapshots[path] = snapshot
        _journal_snapshots.move_to_end(path)
        while len(_journal_snapshots) > _JOURNAL_SNAPSHOT_CACHE_SIZE:
            _journal_snapshots.popitem(last=False)


def _journal_file_identity(path: Path) -> FileIdentity | None:
    try:
        current = path.lstat()
    except FileNotFoundError:
        return None
    if stat_module.S_ISLNK(current.st_mode):
        raise RuntimeError(f"materialization journal must not be a symlink: {path}")
    return FileIdentity.from_stat(current)


def _append_journal_records(
    entries: Iterable[tuple[Path, _JournalRecord]],
) -> None:
    grouped: dict[Path, dict[str, _JournalRecord]] = {}
    for destination, record in entries:
        journal = _journal_path(destination)
        grouped.setdefault(journal, {})[record.destination] = record

    for journal in sorted(grouped, key=str):
        with _journal_lock(journal):
            snapshot = _journal_snapshot(journal)
            records = [grouped[journal][name] for name in sorted(grouped[journal])]
            payload = "".join(f"{canonical_json(record.as_json())}\n" for record in records)
            encoded = payload.encode("utf-8")
            journal.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                journal,
                os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
                0o600,
            )
            with os.fdopen(descriptor, "r+b") as stream:
                stream.truncate(snapshot.valid_bytes)
                stream.seek(snapshot.valid_bytes)
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
                current = os.fstat(stream.fileno())

            updated = dict(snapshot.records)
            updated.update((record.destination, record) for record in records)
            _cache_journal_snapshot(
                journal,
                _JournalSnapshot(
                    records=updated,
                    valid_bytes=snapshot.valid_bytes + len(encoded),
                    file_identity=FileIdentity.from_stat(current),
                ),
            )


def _task_fingerprint(task: MaterializationTask) -> str:
    source_sha256 = task.source_sha256 or sha256_file(task.source)
    return materialization_fingerprint(
        source_sha256=source_sha256,
        source_release=task.source_release,
        sample_rate_hz=task.sample_rate_hz,
        channels=task.channels,
        implementation_fingerprint=task.implementation_fingerprint,
    )


def materialization_fingerprint(
    *,
    source_sha256: str,
    source_release: str,
    sample_rate_hz: int,
    channels: int,
    implementation_fingerprint: str,
) -> str:
    """Identify source bytes and conversion behavior independently of selection mode."""

    return fingerprint(
        {
            "source_sha256": source_sha256,
            "source_release": source_release,
            "sample_rate_hz": sample_rate_hz,
            "channels": channels,
            "sample_format": "pcm_s16le",
            "container": "wav",
            "implementation": implementation_fingerprint,
        }
    )


def _reusable_metadata(
    destination: Path,
    *,
    expected_fingerprint: str,
    sample_rate_hz: int,
    channels: int,
) -> AudioMetadata | None:
    state = _journal_record(destination)
    if state is None:
        return None
    if (
        state.fingerprint != expected_fingerprint
        or state.sample_rate_hz != sample_rate_hz
        or state.channels != channels
        or state.format != "WAV"
        or state.subtype != "PCM_16"
    ):
        return None
    try:
        stat = destination.stat()
    except OSError:
        return None
    if not state.identity.matches_stat(stat):
        return None
    return AudioMetadata(
        path=destination,
        sample_rate_hz=state.sample_rate_hz,
        channels=state.channels,
        num_frames=state.num_frames,
        format=state.format,
        subtype=state.subtype,
        sha256=state.output_sha256,
        identity=state.identity,
    )


def _materialize_with_ffmpeg(task: MaterializationTask, temporary: Path) -> None:
    try:
        subprocess.run(
            _ffmpeg_command(((task, temporary),)),
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or "").strip()
        if not _needs_explicit_channel_average(detail):
            raise RuntimeError(f"ffmpeg could not materialize {task.source}: {detail}") from error

        source_channels = probe(task.source, include_checksum=False).channels
        if source_channels <= 1:
            raise RuntimeError(f"ffmpeg could not materialize {task.source}: {detail}") from error

        temporary.unlink(missing_ok=True)
        try:
            subprocess.run(
                _ffmpeg_command(
                    ((task, temporary),),
                    channel_averages={0: source_channels},
                ),
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as retry_error:
            retry_detail = (retry_error.stderr or "").strip()
            raise RuntimeError(
                f"ffmpeg could not materialize {task.source}: {retry_detail}"
            ) from retry_error


def _materialize_with_ffmpeg_batch(
    pending: list[_PendingMaterialization],
) -> None:
    try:
        subprocess.run(
            _ffmpeg_command(tuple((item.task, item.temporary) for item in pending)),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = ""
        if isinstance(error, subprocess.CalledProcessError):
            detail = (error.stderr or "").strip()
        suffix = f": {detail}" if detail else ""
        raise _FFmpegBatchError(
            f"ffmpeg could not materialize a batch of {len(pending)} inputs{suffix}"
        ) from error


def _ffmpeg_command(
    conversions: tuple[tuple[MaterializationTask, Path], ...],
    *,
    channel_averages: dict[int, int] | None = None,
) -> list[str]:
    channel_averages = channel_averages or {}
    invalid_indexes = sorted(set(channel_averages) - set(range(len(conversions))))
    if invalid_indexes:
        raise ValueError(f"channel averages reference unknown inputs: {invalid_indexes}")

    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-filter_threads",
        "1",
    ]
    for task, _temporary in conversions:
        command.extend(("-threads:a", "1", "-i", str(task.source)))
    for input_index, (task, temporary) in enumerate(conversions):
        source_channels = channel_averages.get(input_index)
        channel_filter: tuple[str, ...] = ()
        if source_channels is not None:
            if source_channels <= 1:
                raise ValueError("channel averaging requires a multichannel source")
            if task.channels != 1:
                raise ValueError("channel averaging supports mono output only")
            inputs = "+".join(f"c{index}" for index in range(source_channels))
            channel_filter = ("-filter:a", f"pan=mono|c0<{inputs}")
        command.extend(
            (
                "-map",
                f"{input_index}:a:0",
                "-map_metadata:g",
                f"{input_index}:g",
                "-map_chapters",
                str(input_index),
                *channel_filter,
                "-ac",
                str(task.channels),
                "-ar",
                str(task.sample_rate_hz),
                "-c:a",
                "pcm_s16le",
                "-threads:a",
                "1",
                "-f",
                "wav",
                str(temporary),
            )
        )
    return command


def _needs_explicit_channel_average(stderr: str) -> bool:
    return (
        re.search(
            r"rematrix\s+is\s+needed.*?not\s+enough\s+information",
            stderr,
            flags=re.IGNORECASE | re.DOTALL,
        )
        is not None
    )


def _materialize_with_soundfile(task: MaterializationTask, temporary: Path) -> None:
    """Test/development fallback; complete CLI builds require ffmpeg."""

    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    samples, source_rate = sf.read(str(task.source), dtype="float32", always_2d=True)
    if samples.shape[1] > 1:
        samples = samples.mean(axis=1, keepdims=True, dtype=np.float32)
    if source_rate != task.sample_rate_hz:
        divisor = math.gcd(int(source_rate), int(task.sample_rate_hz))
        samples = resample_poly(
            samples,
            task.sample_rate_hz // divisor,
            source_rate // divisor,
            axis=0,
        ).astype(np.float32, copy=False)
    sf.write(
        str(temporary),
        np.clip(samples, -1.0, 1.0),
        task.sample_rate_hz,
        format="WAV",
        subtype="PCM_16",
    )


def materialize_all(
    tasks: Iterable[MaterializationTask], *, workers: int = 1
) -> list[AudioMetadata]:
    ordered = sorted(tasks, key=lambda task: str(task.destination))
    batch_size = _FFMPEG_BATCH_SIZE if shutil.which("ffmpeg") is not None else 1
    batches = [ordered[start : start + batch_size] for start in range(0, len(ordered), batch_size)]
    if workers <= 1:
        return [metadata for batch in batches for metadata in _materialize_batch(batch)]
    return _materialize_threaded(batches, workers=workers)


def _materialize_threaded(
    batches: list[list[MaterializationTask]], *, workers: int
) -> list[AudioMetadata]:
    """Materialize bounded concurrent batches with destination-ordered results."""

    if not batches:
        return []

    pending_limit = min(len(batches), workers * 2)
    indexed_batches = iter(enumerate(batches))
    results: list[list[AudioMetadata] | None] = [None] * len(batches)
    pending: dict[Future[list[AudioMetadata]], int] = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        try:
            while len(pending) < pending_limit:
                index, batch = next(indexed_batches)
                pending[executor.submit(_materialize_batch, batch)] = index
        except StopIteration:
            pass

        try:
            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    results[pending.pop(future)] = future.result()

                try:
                    while len(pending) < pending_limit:
                        index, batch = next(indexed_batches)
                        pending[executor.submit(_materialize_batch, batch)] = index
                except StopIteration:
                    pass
        except BaseException:
            for future in pending:
                future.cancel()
            raise

    if any(metadata is None for metadata in results):
        raise RuntimeError("audio materialization completed without metadata for every batch")
    return [
        metadata
        for batch_result in results
        if batch_result is not None
        for metadata in batch_result
    ]


def _materialize_batch(tasks: list[MaterializationTask]) -> list[AudioMetadata]:
    results: list[AudioMetadata | None] = [None] * len(tasks)
    pending: list[tuple[int, _PendingMaterialization]] = []
    for index, task in enumerate(tasks):
        prepared = _prepare_materialization(task)
        if isinstance(prepared, AudioMetadata):
            results[index] = prepared
        else:
            pending.append((index, prepared))

    if pending:
        materialized = _materialize_pending_batch([item for _index, item in pending])
        for (index, _item), metadata in zip(pending, materialized, strict=True):
            results[index] = metadata

    if any(metadata is None for metadata in results):
        raise RuntimeError("audio batch completed without metadata for every task")
    return [metadata for metadata in results if metadata is not None]


def _materialize_pending_batch(
    pending: list[_PendingMaterialization],
) -> list[AudioMetadata]:
    if len(pending) == 1 or shutil.which("ffmpeg") is None:
        return [_materialize_pending(item) for item in pending]

    try:
        _materialize_with_ffmpeg_batch(pending)
    except _FFmpegBatchError:
        _cleanup_temporaries(pending)
        return _materialize_pending_individually_with_ffmpeg(pending)

    return _verify_and_publish_pending(pending)


def _materialize_pending_individually_with_ffmpeg(
    pending: list[_PendingMaterialization],
) -> list[AudioMetadata]:
    try:
        for item in pending:
            _materialize_with_ffmpeg(item.task, item.temporary)
        return _verify_and_publish_pending(pending)
    finally:
        _cleanup_temporaries(pending)


def _verify_and_publish_pending(
    pending: list[_PendingMaterialization],
) -> list[AudioMetadata]:
    try:
        written = [_verify_temporary(item) for item in pending]
        return _publish_verified(zip(pending, written, strict=True))
    finally:
        _cleanup_temporaries(pending)


def _cleanup_temporaries(pending: Iterable[_PendingMaterialization]) -> None:
    for item in pending:
        item.temporary.unlink(missing_ok=True)
