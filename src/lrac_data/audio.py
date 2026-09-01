# Copyright 2026 Cisco Systems, Inc. and its affiliates
# Apache-2.0

"""Audio inspection and deterministic float32 WAV materialization."""

from __future__ import annotations

import hashlib
import math
import os
import re
import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

from .models import ChannelMode, SourceSegment
from .state import FileIdentity, fingerprint, sha256_file

_CACHE_NAME = ".lrac-audio.sqlite3"
_MP3_SEEK_PREROLL_SECONDS = 1
_REQUIRED_SOURCE_FORMATS = frozenset({"FLAC", "MP3", "WAV"})
_WAV_FORMATS = frozenset({"WAV", "WAVEX"})


@dataclass(frozen=True, slots=True)
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

        if self.identity is None:
            return False
        try:
            current = path.stat()
        except OSError:
            return False
        return self.identity.matches_stat(current)


@dataclass(frozen=True, slots=True)
class MaterializationTask:
    source: Path
    destination: Path
    sample_rate_hz: int = 24_000
    channel_mode: ChannelMode = ChannelMode.PRESERVE
    source_release: str = "unspecified"
    implementation_fingerprint: str = "unspecified"
    source_sha256: str | None = None
    source_identity: FileIdentity | None = None
    source_segment: SourceSegment | None = None


@dataclass(frozen=True, slots=True)
class _PendingMaterialization:
    task: MaterializationTask
    temporary: Path
    source_metadata: AudioMetadata
    start_frame: int
    input_frames: int


@dataclass(frozen=True, slots=True)
class _CacheRecord:
    fingerprint: str
    sha256: str
    sample_rate_hz: int
    channels: int
    num_frames: int
    format: str
    subtype: str
    identity: FileIdentity

    def metadata(self, path: Path) -> AudioMetadata:
        return AudioMetadata(
            path=path,
            sample_rate_hz=self.sample_rate_hz,
            channels=self.channels,
            num_frames=self.num_frames,
            format=self.format,
            subtype=self.subtype,
            sha256=self.sha256,
            identity=self.identity,
            checksum_fresh=True,
        )


class _MaterializationCache:
    """Coordinator-owned cache for authenticated materialized audio."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise RuntimeError(f"materialization cache must not be a symlink: {path}")
        self.connection = _open_cache(path)

    def close(self) -> None:
        self.connection.close()

    def read(self, destinations: Iterable[Path]) -> dict[Path, _CacheRecord]:
        paths = list(destinations)
        records: dict[Path, _CacheRecord] = {}
        for start in range(0, len(paths), 400):
            chunk = paths[start : start + 400]
            keys = [_cache_key(path) for path in chunk]
            placeholders = ",".join("?" for _key in keys)
            query = f"SELECT * FROM materialized WHERE destination IN ({placeholders})"
            for row in self.connection.execute(query, keys):
                try:
                    destination = Path(row[0])
                    records[destination] = _CacheRecord(
                        fingerprint=row[1],
                        sha256=row[2],
                        sample_rate_hz=row[3],
                        channels=row[4],
                        num_frames=row[5],
                        format=row[6],
                        subtype=row[7],
                        identity=FileIdentity(*row[8:13]),
                    )
                except (TypeError, ValueError):
                    continue
        return records

    def write(
        self,
        tasks: Iterable[MaterializationTask],
        metadata: Mapping[Path, AudioMetadata],
    ) -> None:
        rows = []
        for task in tasks:
            result = metadata[task.destination]
            identity = result.identity or FileIdentity.from_stat(task.destination.stat())
            rows.append(
                (
                    _cache_key(task.destination),
                    _task_fingerprint(task),
                    result.sha256,
                    result.sample_rate_hz,
                    result.channels,
                    result.num_frames,
                    result.format,
                    result.subtype,
                    identity.size,
                    identity.mtime_ns,
                    identity.ctime_ns,
                    identity.device,
                    identity.inode,
                )
            )
        with self.connection:
            self.connection.executemany(
                """
                INSERT INTO materialized VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(destination) DO UPDATE SET
                  fingerprint=excluded.fingerprint,
                  sha256=excluded.sha256,
                  sample_rate_hz=excluded.sample_rate_hz,
                  channels=excluded.channels,
                  num_frames=excluded.num_frames,
                  format=excluded.format,
                  subtype=excluded.subtype,
                  size=excluded.size,
                  mtime_ns=excluded.mtime_ns,
                  ctime_ns=excluded.ctime_ns,
                  device=excluded.device,
                  inode=excluded.inode
                """,
                rows,
            )

    def refresh_identities(self, metadata: Mapping[Path, AudioMetadata]) -> None:
        def rows() -> Iterator[tuple[int, int, int, int, int, str, str]]:
            for path, result in metadata.items():
                identity = result.identity
                if identity is not None and result.matches_stat(path):
                    yield (
                        identity.size,
                        identity.mtime_ns,
                        identity.ctime_ns,
                        identity.device,
                        identity.inode,
                        _cache_key(path),
                        result.sha256,
                    )

        with self.connection:
            self.connection.executemany(
                """
                UPDATE materialized
                SET size=?, mtime_ns=?, ctime_ns=?, device=?, inode=?
                WHERE destination=? AND sha256=?
                """,
                rows(),
            )


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

    info = sf.info(str(path))
    checksum = sha256_file(path) if include_checksum else ""
    return AudioMetadata(
        path=path,
        sample_rate_hz=info.samplerate,
        channels=info.channels,
        num_frames=info.frames,
        format=info.format,
        subtype=info.subtype,
        sha256=checksum,
        identity=FileIdentity.from_stat(path.stat()),
        checksum_fresh=include_checksum,
    )


def is_float32_wav(format_name: str, subtype: str) -> bool:
    """Return whether libsndfile metadata describes IEEE float32 WAV."""

    return format_name in _WAV_FORMATS and subtype == "FLOAT"


def missing_soundfile_formats() -> tuple[str, ...]:
    """Return configured source formats unsupported by the libsndfile backend."""

    try:
        import soundfile as sf

        available = sf.available_formats().keys()
    except (ImportError, OSError, RuntimeError):
        return tuple(sorted(_REQUIRED_SOURCE_FORMATS))
    return tuple(sorted(_REQUIRED_SOURCE_FORMATS - available))


def _materialize_task(task: MaterializationTask) -> AudioMetadata:
    if task.sample_rate_hz <= 0:
        raise ValueError("target sample rate must be positive")
    if not task.source.is_file():
        raise FileNotFoundError(task.source)
    _require_authenticated_source(task)
    source_metadata = probe(task.source, include_checksum=False)
    start_frame, input_frames = _frame_window(task, source_metadata)
    destination = task.destination
    temporary = destination.with_name(f".{destination.stem}.tmp.wav")
    temporary.unlink(missing_ok=True)
    pending = _PendingMaterialization(
        task=task,
        temporary=temporary,
        source_metadata=source_metadata,
        start_frame=start_frame,
        input_frames=input_frames,
    )
    try:
        output_frames = _write_soundfile_audio(pending)
        written = probe(temporary)
        expected_channels = (
            1 if task.channel_mode is ChannelMode.DOWNMIX else source_metadata.channels
        )
        if (
            written.sample_rate_hz != task.sample_rate_hz
            or written.channels != expected_channels
            or written.num_frames != output_frames
            or not is_float32_wav(written.format, written.subtype)
        ):
            raise ValueError(f"materialized audio failed validation: {task.source}")
        pending.temporary.replace(task.destination)
        return replace(
            written,
            path=task.destination,
            identity=FileIdentity.from_stat(task.destination.stat()),
        )
    finally:
        pending.temporary.unlink(missing_ok=True)


def _task_fingerprint(task: MaterializationTask) -> str:
    source_identity = task.source_identity
    if task.source_sha256 is None and source_identity is None:
        source_identity = FileIdentity.from_stat(task.source.stat())
    source: dict[str, object] = {"sha256": task.source_sha256}
    if task.source_sha256 is None:
        assert source_identity is not None
        source = {"identity": source_identity.as_dict()}
    return fingerprint(
        {
            "source": source,
            "source_release": task.source_release,
            "sample_rate_hz": task.sample_rate_hz,
            "channels": task.channel_mode.value,
            "sample_format": "float32",
            "container": "wav",
            "source_segment": (
                task.source_segment.model_dump(mode="json")
                if task.source_segment is not None
                else None
            ),
            "implementation": task.implementation_fingerprint,
        }
    )


def _cache_key(path: Path) -> str:
    return str(Path(os.path.abspath(path)))


def _open_cache(path: Path) -> sqlite3.Connection:
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(path)
        _create_cache_table(connection)
        return connection
    except sqlite3.DatabaseError:
        if connection is not None:
            connection.close()
        for suffix in ("", "-journal", "-wal", "-shm"):
            Path(f"{path}{suffix}").unlink(missing_ok=True)
        connection = sqlite3.connect(path)
        _create_cache_table(connection)
        return connection


def _create_cache_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS materialized (
          destination TEXT PRIMARY KEY,
          fingerprint TEXT NOT NULL,
          sha256 TEXT NOT NULL,
          sample_rate_hz INTEGER NOT NULL,
          channels INTEGER NOT NULL,
          num_frames INTEGER NOT NULL,
          format TEXT NOT NULL,
          subtype TEXT NOT NULL,
          size INTEGER NOT NULL,
          mtime_ns INTEGER NOT NULL,
          ctime_ns INTEGER NOT NULL,
          device INTEGER NOT NULL,
          inode INTEGER NOT NULL
        )
        """
    )


def _reuse_cached(task: MaterializationTask, record: _CacheRecord | None) -> AudioMetadata | None:
    if record is None or record.fingerprint != _task_fingerprint(task):
        return None
    _require_authenticated_source(task)
    destination = task.destination
    if not destination.is_file() or destination.is_symlink():
        return None
    try:
        current = FileIdentity.from_stat(destination.stat())
        if current == record.identity:
            return record.metadata(destination)
        metadata = probe(destination)
        if metadata.sha256 != record.sha256:
            return None
    except (OSError, RuntimeError, ValueError):
        return None
    return metadata


def _microseconds_to_frame(microseconds: int, sample_rate_hz: int) -> int:
    return (microseconds * sample_rate_hz + 500_000) // 1_000_000


def _frame_window(
    task: MaterializationTask,
    source_metadata: AudioMetadata,
) -> tuple[int, int]:
    if source_metadata.sample_rate_hz <= 0 or source_metadata.channels <= 0:
        raise ValueError(f"source audio has invalid metadata: {task.source}")

    if task.source_segment is None:
        start_frame = 0
        end_frame = source_metadata.num_frames
    else:
        start_frame = _microseconds_to_frame(
            task.source_segment.start_us,
            source_metadata.sample_rate_hz,
        )
        end_frame = _microseconds_to_frame(
            task.source_segment.end_us,
            source_metadata.sample_rate_hz,
        )
    if start_frame < 0 or start_frame >= end_frame or end_frame > source_metadata.num_frames:
        segment = task.source_segment
        description = (
            f"{segment.start_us}:{segment.end_us} us" if segment is not None else "full file"
        )
        raise ValueError(f"source segment {description} is outside {task.source}")

    return start_frame, end_frame - start_frame


def _write_soundfile_audio(pending: _PendingMaterialization) -> int:
    """Read one bounded source window, transform it, and write float32 WAV."""

    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    task = pending.task
    source = pending.source_metadata
    read_start = pending.start_frame
    if source.format == "MP3":
        # Discard decoder warm-up because a direct libsndfile MP3 seek can begin with silence.
        read_start = max(
            0,
            pending.start_frame - _MP3_SEEK_PREROLL_SECONDS * source.sample_rate_hz,
        )
    discarded_frames = pending.start_frame - read_start
    read_frames = (
        -1
        if task.source_segment is None and source.format == "MP3"
        else discarded_frames + pending.input_frames
    )

    with sf.SoundFile(str(task.source)) as stream:
        position = stream.seek(read_start)
        if position != read_start:
            raise RuntimeError(f"could not seek to source segment in {task.source}")
        samples = stream.read(
            frames=read_frames,
            dtype="float32",
            always_2d=True,
        )

    if (
        samples.ndim != 2
        or samples.shape[1] != source.channels
        or (read_frames >= 0 and samples.shape[0] != read_frames)
        or samples.shape[0] <= discarded_frames
    ):
        raise RuntimeError(f"could not read the complete source segment from {task.source}")
    samples = samples[discarded_frames:]
    output_frames = int(
        samples.shape[0] * task.sample_rate_hz + source.sample_rate_hz - 1
    ) // source.sample_rate_hz
    _require_authenticated_source(task)
    if task.channel_mode is ChannelMode.DOWNMIX and source.channels > 1:
        samples = samples.mean(axis=1, keepdims=True, dtype=np.float32)
    if source.sample_rate_hz != task.sample_rate_hz:
        divisor = math.gcd(source.sample_rate_hz, task.sample_rate_hz)
        samples = resample_poly(
            samples,
            task.sample_rate_hz // divisor,
            source.sample_rate_hz // divisor,
            axis=0,
        ).astype(np.float32, copy=False)
    if samples.shape[0] != output_frames:
        raise RuntimeError(f"resampling produced an unexpected frame count for {task.source}")
    if not np.isfinite(samples).all():
        raise ValueError(f"materialized audio contains non-finite samples: {task.source}")

    sf.write(
        str(pending.temporary),
        samples,
        task.sample_rate_hz,
        format="WAV",
        subtype="FLOAT",
    )
    return output_frames


def _require_authenticated_source(task: MaterializationTask) -> None:
    """Reject source replacement after its checksum was authenticated."""

    if task.source_identity is None:
        return
    try:
        unchanged = task.source_identity.matches_stat(task.source.stat())
    except OSError:
        unchanged = False
    if not unchanged:
        raise RuntimeError(f"audio changed after its checksum was computed: {task.source}")


def materialize_all(
    tasks: Iterable[MaterializationTask],
    *,
    workers: int = 1,
    checkpoint: Path | None = None,
) -> list[AudioMetadata]:
    ordered = sorted(tasks, key=lambda task: str(task.destination))
    if workers < 1:
        raise ValueError("materialization workers must be positive")
    if not ordered:
        return []
    destinations = [task.destination for task in ordered]
    if len(destinations) != len(set(destinations)):
        raise ValueError("materialization destinations must be unique")
    for parent in sorted({destination.parent for destination in destinations}):
        parent.mkdir(parents=True, exist_ok=True)

    if checkpoint is None:
        common_parent = Path(
            os.path.commonpath([str(path.parent.absolute()) for path in destinations])
        )
        checkpoint = common_parent / _CACHE_NAME
    cache = _MaterializationCache(checkpoint)
    try:
        cached = cache.read(destinations)
        metadata_by_path: dict[Path, AudioMetadata] = {}
        pending: list[MaterializationTask] = []
        for task in ordered:
            record = cached.get(Path(_cache_key(task.destination)))
            metadata = _reuse_cached(task, record)
            if metadata is None:
                pending.append(task)
            else:
                metadata_by_path[task.destination] = metadata

        if pending:
            if workers == 1:
                converted = [_materialize_task(task) for task in pending]
            else:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    converted = list(executor.map(_materialize_task, pending))
            metadata_by_path.update((metadata.path, metadata) for metadata in converted)
        cache.write(ordered, metadata_by_path)
        return [metadata_by_path[task.destination] for task in ordered]
    finally:
        cache.close()


def refresh_materialization_cache(
    checkpoint: Path,
    metadata: Mapping[Path, AudioMetadata],
) -> None:
    """Record exact identities after publication adds hard links."""

    if not metadata:
        return
    cache = _MaterializationCache(checkpoint)
    try:
        cache.refresh_identities(metadata)
    finally:
        cache.close()
