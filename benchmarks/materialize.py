"""Benchmark deterministic audio materialization without network access."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import wave
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar

from lrac_data.audio import AudioMetadata, MaterializationTask, materialize_all
from lrac_data.manifests import write_manifest
from lrac_data.models import ManifestItem, MediaKind, Split
from lrac_data.state import FileIdentity, sha256_file
from lrac_data.validation import validate_manifests

SCHEMA_VERSION = 2
IMPLEMENTATION_FINGERPRINT = "lrac-synthetic-materialization-v1"
_JOURNAL_NAME = ".lrac-materialization.jsonl"

ResultT = TypeVar("ResultT")


class PeakRssSampler:
    """Sample aggregate resident memory for this process and its descendants."""

    def __init__(self, *, interval_seconds: float = 0.05) -> None:
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self.samples = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._sample()
        self._thread = threading.Thread(target=self._run, name="rss-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self._sample()

    def _sample(self) -> None:
        rss = _process_tree_rss_bytes(os.getpid())
        self.peak_bytes = max(self.peak_bytes, rss)
        self.samples += 1


def run_benchmark(
    workspace: Path,
    *,
    items: int = 1_000,
    duration_seconds: float = 0.25,
    source_sample_rate_hz: int = 48_000,
    source_channels: int = 2,
    target_sample_rate_hz: int = 24_000,
    worker_counts: tuple[int, ...] = (1, 2, 4, 8),
    repetitions: int = 1,
    seed: int = 42,
    json_output: Path | None = None,
    require_ffmpeg: bool = False,
) -> dict[str, Any]:
    """Benchmark cold conversion, warm reuse, validation, and journal recovery."""

    _validate_arguments(
        items=items,
        duration_seconds=duration_seconds,
        source_sample_rate_hz=source_sample_rate_hz,
        source_channels=source_channels,
        target_sample_rate_hz=target_sample_rate_hz,
        worker_counts=worker_counts,
        repetitions=repetitions,
    )
    ffmpeg = shutil.which("ffmpeg")
    if require_ffmpeg and ffmpeg is None:
        raise RuntimeError("ffmpeg is required for this benchmark run")

    workspace = workspace.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    run_dir = _create_run_directory(workspace)
    source_dir = run_dir / "sources"

    generation_started = time.perf_counter()
    sources = _generate_sources(
        source_dir,
        items=items,
        duration_seconds=duration_seconds,
        sample_rate_hz=source_sample_rate_hz,
        channels=source_channels,
        seed=seed,
    )
    generation_wall_seconds = time.perf_counter() - generation_started
    source_checksums = {path.name: sha256_file(path) for path in sources}
    source_bytes = sum(path.stat().st_size for path in sources)
    frame_count = max(1, int(duration_seconds * source_sample_rate_hz + 0.5))
    exact_audio_seconds = len(sources) * frame_count / source_sample_rate_hz

    cases: list[dict[str, Any]] = []
    checksum_sets: list[dict[str, str]] = []
    baseline_checksums: dict[str, str] | None = None
    for workers in worker_counts:
        for repetition in range(1, repetitions + 1):
            output_dir = run_dir / "outputs" / f"workers-{workers}" / f"run-{repetition}"
            tasks = [
                MaterializationTask(
                    source=source,
                    destination=(
                        output_dir
                        / hashlib.sha256(source.name.encode("utf-8")).hexdigest()[:2]
                        / source.name
                    ),
                    sample_rate_hz=target_sample_rate_hz,
                    channels=1,
                    source_release="synthetic-v1",
                    implementation_fingerprint=IMPLEMENTATION_FINGERPRINT,
                    source_sha256=source_checksums[source.name],
                )
                for source in sources
            ]
            cold_metadata, cold_phase = _measure_phase(
                partial(materialize_all, tasks, workers=workers),
                items=items,
                audio_seconds=exact_audio_seconds,
            )
            cold_checksums = {item.path.name: item.sha256 for item in cold_metadata}
            checksum_sets.append(cold_checksums)
            if baseline_checksums is None:
                baseline_checksums = cold_checksums

            manifest = _write_benchmark_manifest(
                run_dir,
                output_dir / "manifest.jsonl",
                cold_metadata,
            )
            warm_metadata, warm_phase = _measure_phase(
                partial(materialize_all, tasks, workers=workers),
                items=items,
                audio_seconds=exact_audio_seconds,
            )
            warm_checksums = {item.path.name: item.sha256 for item in warm_metadata}
            checksum_sets.append(warm_checksums)
            warm_validation, warm_validation_phase = _measure_phase(
                partial(
                    validate_manifests,
                    [manifest],
                    workspace=run_dir,
                    known_audio={item.path: item for item in warm_metadata},
                    workers=workers,
                ),
                items=items,
                audio_seconds=exact_audio_seconds,
            )

            journals = sorted(output_dir.rglob(_JOURNAL_NAME))
            if not journals:
                raise RuntimeError(f"materialization created no journals below {output_dir}")
            journal = journals[-1]
            identities_before_recovery = {
                item.path.name: FileIdentity.from_stat(item.path.stat()) for item in warm_metadata
            }
            torn_destination = _tear_last_journal_record(journal)
            recovered_metadata, recovery_phase = _measure_phase(
                partial(materialize_all, tasks, workers=workers),
                items=items,
                audio_seconds=exact_audio_seconds,
            )
            recovered_checksums = {item.path.name: item.sha256 for item in recovered_metadata}
            checksum_sets.append(recovered_checksums)
            recovery_validation, recovery_validation_phase = _measure_phase(
                partial(
                    validate_manifests,
                    [manifest],
                    workspace=run_dir,
                    workers=workers,
                ),
                items=items,
                audio_seconds=exact_audio_seconds,
            )
            identities_after_recovery = {
                item.path.name: FileIdentity.from_stat(item.path.stat())
                for item in recovered_metadata
            }
            reprocessed = sorted(
                name
                for name, identity in identities_before_recovery.items()
                if identities_after_recovery[name] != identity
            )
            journal_health = [_journal_health(path) for path in journals]
            journal_records = sum(records for records, _complete in journal_health)
            journal_complete = all(complete for _records, complete in journal_health)
            torn_checksum_matches = (
                sha256_file(journal.parent / torn_destination)
                == baseline_checksums[torn_destination]
            )

            cold_mismatches = _checksum_mismatches(baseline_checksums, cold_checksums)
            warm_mismatches = _checksum_mismatches(baseline_checksums, warm_checksums)
            recovery_mismatches = _checksum_mismatches(
                baseline_checksums,
                recovered_checksums,
            )
            output_bytes = sum(item.path.stat().st_size for item in recovered_metadata)
            cases.append(
                {
                    "workers": workers,
                    "repetition": repetition,
                    "output_bytes": output_bytes,
                    "checksum_set_sha256": _checksum_set_digest(cold_checksums),
                    "phases": {
                        "cold_materialization": cold_phase,
                        "warm_materialization": warm_phase,
                        "warm_checksum_validation": warm_validation_phase,
                        "torn_journal_recovery": recovery_phase,
                        "recovery_checksum_validation": recovery_validation_phase,
                    },
                    "parity": {
                        "cold": not cold_mismatches,
                        "warm": not warm_mismatches,
                        "recovery": not recovery_mismatches,
                        "warm_validation": warm_validation.ok,
                        "recovery_validation": recovery_validation.ok,
                        "torn_output_checksum": torn_checksum_matches,
                        "mismatched_paths": sorted(
                            set(cold_mismatches + warm_mismatches + recovery_mismatches)
                        )[:20],
                        "validation_errors": list(
                            warm_validation.errors + recovery_validation.errors
                        )[:20],
                    },
                    "recovery": {
                        "torn_destination": torn_destination,
                        "reprocessed_paths": reprocessed,
                        "reprocessed_expected_path": reprocessed == [torn_destination],
                        "journal_records": journal_records,
                        "journal_files": len(journals),
                        "journal_complete": journal_complete,
                    },
                }
            )

    parity_mismatches: list[str] = []
    assert baseline_checksums is not None
    for checksums in checksum_sets[1:]:
        parity_mismatches.extend(_checksum_mismatches(baseline_checksums, checksums))
    parity_mismatches = sorted(set(parity_mismatches))
    case_failures = [
        f"workers={case['workers']}, repetition={case['repetition']}"
        for case in cases
        if not all(
            case["parity"][key]
            for key in (
                "cold",
                "warm",
                "recovery",
                "warm_validation",
                "recovery_validation",
                "torn_output_checksum",
            )
        )
        or not case["recovery"]["reprocessed_expected_path"]
        or not case["recovery"]["journal_complete"]
        or case["recovery"]["journal_records"] != items
    ]

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "synthetic-wav-materialization",
        "created_at": datetime.now(UTC).isoformat(),
        "run_directory": str(run_dir),
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "logical_cpus": os.cpu_count(),
            "backend": "ffmpeg" if ffmpeg is not None else "soundfile-fallback",
            "ffmpeg": _ffmpeg_version(ffmpeg),
            "rss_measurement": "sampled aggregate VmRSS for the process tree",
        },
        "configuration": {
            "items": items,
            "duration_seconds": duration_seconds,
            "source_sample_rate_hz": source_sample_rate_hz,
            "source_channels": source_channels,
            "target_sample_rate_hz": target_sample_rate_hz,
            "target_channels": 1,
            "worker_counts": list(worker_counts),
            "repetitions": repetitions,
            "seed": seed,
        },
        "sources": {
            "generation_wall_seconds": generation_wall_seconds,
            "files": len(sources),
            "bytes": source_bytes,
            "audio_seconds": exact_audio_seconds,
            "checksum_set_sha256": _checksum_set_digest(source_checksums),
        },
        "cases": cases,
        "parity": {
            "ok": not parity_mismatches and not case_failures,
            "reference_workers": worker_counts[0],
            "mismatched_paths": parity_mismatches[:100],
            "failed_cases": case_failures,
        },
    }
    destination = json_output.expanduser().resolve() if json_output else run_dir / "result.json"
    _atomic_write_json(destination, result)
    result["result_path"] = str(destination)
    return result


def _validate_arguments(
    *,
    items: int,
    duration_seconds: float,
    source_sample_rate_hz: int,
    source_channels: int,
    target_sample_rate_hz: int,
    worker_counts: tuple[int, ...],
    repetitions: int,
) -> None:
    if items <= 0:
        raise ValueError("items must be positive")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if source_sample_rate_hz <= 0 or target_sample_rate_hz <= 0:
        raise ValueError("sample rates must be positive")
    if source_channels <= 0:
        raise ValueError("source_channels must be positive")
    if not worker_counts or any(workers <= 0 for workers in worker_counts):
        raise ValueError("worker_counts must contain positive values")
    if len(worker_counts) != len(set(worker_counts)):
        raise ValueError("worker_counts must be unique")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")


def _measure_phase(
    operation: Callable[[], ResultT],
    *,
    items: int,
    audio_seconds: float,
) -> tuple[ResultT, dict[str, float | int]]:
    before_times = os.times()
    sampler = PeakRssSampler()
    sampler.start()
    started = time.perf_counter()
    try:
        result = operation()
    finally:
        wall_seconds = time.perf_counter() - started
        sampler.stop()
    after_times = os.times()
    user_seconds = (after_times.user - before_times.user) + (
        after_times.children_user - before_times.children_user
    )
    system_seconds = (after_times.system - before_times.system) + (
        after_times.children_system - before_times.children_system
    )
    return result, {
        "wall_seconds": wall_seconds,
        "user_cpu_seconds": user_seconds,
        "system_cpu_seconds": system_seconds,
        "peak_rss_bytes": sampler.peak_bytes,
        "rss_samples": sampler.samples,
        "items_per_second": items / wall_seconds,
        "audio_seconds_per_second": audio_seconds / wall_seconds,
    }


def _write_benchmark_manifest(
    workspace: Path,
    destination: Path,
    metadata: list[AudioMetadata],
) -> Path:
    records = [
        ManifestItem(
            id=f"synthetic:{item.path.name}",
            dataset="synthetic",
            media_kind=MediaKind.NOISE,
            audio_path=PurePosixPath(item.path.relative_to(workspace).as_posix()),
            source_release="synthetic-v1",
            source_id=item.path.name,
            split=Split.TRAIN,
            sample_rate_hz=item.sample_rate_hz,
            channels=item.channels,
            frame_count=item.num_frames,
            checksum=item.sha256,
        )
        for item in metadata
    ]
    return write_manifest(destination, records)


def _tear_last_journal_record(path: Path) -> str:
    payload = path.read_bytes()
    lines = payload.splitlines(keepends=True)
    if not lines or not lines[-1].endswith(b"\n"):
        raise RuntimeError(f"cannot tear incomplete materialization journal: {path}")
    try:
        last_record = json.loads(lines[-1])
        destination = last_record["destination"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError(f"cannot decode materialization journal: {path}") from error
    if not isinstance(destination, str):
        raise RuntimeError(f"journal destination is not a string: {path}")

    last_line = lines[-1][:-1]
    torn_tail = last_line[: max(1, len(last_line) // 2)]
    with path.open("r+b") as stream:
        stream.seek(0)
        stream.write(b"".join(lines[:-1]))
        stream.write(torn_tail)
        stream.truncate()
        stream.flush()
        os.fsync(stream.fileno())
    return destination


def _journal_health(path: Path) -> tuple[int, bool]:
    payload = path.read_bytes()
    if not payload.endswith(b"\n"):
        return 0, False
    destinations: set[str] = set()
    try:
        for line in payload.splitlines():
            value = json.loads(line)
            destination = value["destination"]
            if not isinstance(destination, str) or destination in destinations:
                return len(destinations), False
            destinations.add(destination)
    except (json.JSONDecodeError, KeyError, TypeError):
        return len(destinations), False
    return len(destinations), True


def _generate_sources(
    directory: Path,
    *,
    items: int,
    duration_seconds: float,
    sample_rate_hz: int,
    channels: int,
    seed: int,
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=False)
    frame_count = max(1, int(duration_seconds * sample_rate_hz + 0.5))
    sample_count = frame_count * channels
    values = array.array(
        "h",
        (((sample * 1103 + seed * 7919) % 60_001) - 30_000 for sample in range(sample_count)),
    )
    if sys.byteorder != "little":
        values.byteswap()
    base_payload = values.tobytes()

    width = max(6, len(str(items - 1)))
    paths: list[Path] = []
    for index in range(items):
        path = directory / f"item-{index:0{width}d}.wav"
        payload = bytearray(base_payload)
        marker = hashlib.sha256(f"{seed}:{index}".encode()).digest()
        payload[: min(len(payload), len(marker))] = marker[: min(len(payload), len(marker))]
        temporary = path.with_name(f".{path.name}.tmp")
        with wave.open(str(temporary), "wb") as output:
            output.setnchannels(channels)
            output.setsampwidth(2)
            output.setframerate(sample_rate_hz)
            output.writeframes(payload)
        temporary.replace(path)
        paths.append(path)
    return paths


def _create_run_directory(workspace: Path) -> Path:
    stem = datetime.now(UTC).strftime("run-%Y%m%dT%H%M%S.%fZ")
    candidate = workspace / stem
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = workspace / f"{stem}-{suffix}"
    candidate.mkdir()
    return candidate


def _checksum_mismatches(reference: dict[str, str], candidate: dict[str, str]) -> list[str]:
    return sorted(
        name
        for name in set(reference) | set(candidate)
        if reference.get(name) != candidate.get(name)
    )


def _checksum_set_digest(checksums: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name, checksum in sorted(checksums.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(checksum.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _process_tree_rss_bytes(root_pid: int) -> int:
    proc = Path("/proc")
    if not proc.is_dir():
        return 0
    children: dict[int, list[int]] = {}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            fields = stat[stat.rfind(")") + 2 :].split()
            parent_pid = int(fields[1])
            pid = int(entry.name)
        except (FileNotFoundError, IndexError, OSError, ValueError):
            continue
        children.setdefault(parent_pid, []).append(pid)

    total_kib = 0
    queue = deque([root_pid])
    seen: set[int] = set()
    while queue:
        pid = queue.popleft()
        if pid in seen:
            continue
        seen.add(pid)
        queue.extend(children.get(pid, ()))
        try:
            status = (proc / str(pid) / "status").read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            continue
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                total_kib += int(line.split()[1])
                break
    return total_kib * 1024


def _ffmpeg_version(executable: str | None) -> str | None:
    if executable is None:
        return None
    try:
        output = subprocess.run(
            [executable, "-version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return output.splitlines()[0] if output else None


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--items", type=int, default=1_000)
    parser.add_argument("--duration-seconds", type=float, default=0.25)
    parser.add_argument("--source-sample-rate-hz", type=int, default=48_000)
    parser.add_argument("--source-channels", type=int, default=2)
    parser.add_argument("--target-sample-rate-hz", type=int, default=24_000)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--require-ffmpeg", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_benchmark(
        args.workspace,
        items=args.items,
        duration_seconds=args.duration_seconds,
        source_sample_rate_hz=args.source_sample_rate_hz,
        source_channels=args.source_channels,
        target_sample_rate_hz=args.target_sample_rate_hz,
        worker_counts=tuple(args.workers),
        repetitions=args.repetitions,
        seed=args.seed,
        json_output=args.json_output,
        require_ffmpeg=args.require_ffmpeg,
    )
    print(result["result_path"])
    if not result["parity"]["ok"]:
        print("benchmark parity or recovery checks failed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
