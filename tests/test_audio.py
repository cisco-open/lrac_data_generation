from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import lrac_data.audio as audio_module
from lrac_data.audio import AudioMetadata, MaterializationTask, materialize, materialize_all, probe
from lrac_data.state import sha256_file


def test_materialize_writes_mono_24khz_pcm16_wav(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    destination = tmp_path / "prepared" / "item.wav"
    frames = np.arange(4_800, dtype=np.float32)
    stereo = np.column_stack(
        (
            np.sin(2 * np.pi * 440 * frames / 48_000),
            np.sin(2 * np.pi * 660 * frames / 48_000),
        )
    )
    sf.write(source, stereo, 48_000, subtype="FLOAT")

    metadata = materialize(MaterializationTask(source=source, destination=destination))
    info = sf.info(destination)

    assert metadata == probe(destination)
    assert metadata.sample_rate_hz == 24_000
    assert metadata.channels == 1
    assert metadata.num_frames == 2_400
    assert metadata.format == "WAV"
    assert len(metadata.sha256) == 64
    assert info.format == "WAV"
    assert info.subtype == "PCM_16"
    assert not (destination.parent / ".item.tmp.wav").exists()


def test_materialize_rejects_flac_bytes_written_to_wav_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.wav"
    destination = tmp_path / "prepared.wav"
    sf.write(source, np.zeros(240, dtype=np.float32), 24_000, subtype="PCM_16")

    def write_flac(_task: MaterializationTask, temporary: Path) -> None:
        sf.write(
            temporary,
            np.zeros(240, dtype=np.float32),
            24_000,
            format="FLAC",
            subtype="PCM_16",
        )

    monkeypatch.setattr(audio_module.shutil, "which", lambda _name: None)
    monkeypatch.setattr(audio_module, "_materialize_with_soundfile", write_flac)

    with pytest.raises(ValueError, match="materialized audio failed validation"):
        materialize(MaterializationTask(source, destination))

    assert not destination.exists()
    assert not destination.with_name(".prepared.tmp.wav").exists()


def test_materialize_reuses_only_when_source_and_fingerprint_match(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    destination = tmp_path / "prepared.wav"
    sf.write(source, np.zeros(240, dtype=np.float32), 24_000, subtype="PCM_16")

    first = materialize(MaterializationTask(source=source, destination=destination))
    second = materialize(MaterializationTask(source=source, destination=destination))

    assert second == first

    sf.write(source, np.ones(240, dtype=np.float32) * 0.5, 24_000, subtype="PCM_16")
    changed = materialize(MaterializationTask(source=source, destination=destination))

    assert changed.sha256 != first.sha256


def test_materialize_hashes_new_output_once_and_reuses_by_stat(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.wav"
    destination = tmp_path / "prepared.wav"
    sf.write(source, np.zeros(240, dtype=np.float32), 24_000, subtype="PCM_16")
    task = MaterializationTask(
        source=source,
        destination=destination,
        source_sha256=sha256_file(source),
    )
    checksum_paths: list[Path] = []
    real_sha256_file = audio_module.sha256_file

    def recording_sha256_file(path: Path) -> str:
        checksum_paths.append(path)
        return real_sha256_file(path)

    monkeypatch.setattr(audio_module, "sha256_file", recording_sha256_file)

    first = materialize(task)
    assert checksum_paths == [destination.with_name(".prepared.tmp.wav")]

    journal = audio_module._state_path(destination)
    journal_record = json.loads(journal.read_text(encoding="utf-8"))
    identity = journal_record.pop("identity")
    journal_record.pop("format")
    journal_record.update(
        {
            "size_bytes": identity["size"],
            "mtime_ns": identity["mtime_ns"],
            "ctime_ns": identity["ctime_ns"],
            "device": identity["device"],
            "inode": identity["inode"],
        }
    )
    journal.write_text(json.dumps(journal_record) + "\n", encoding="utf-8")
    audio_module._journal_snapshots.pop(audio_module._journal_path(destination), None)
    checksum_paths.clear()
    second = materialize(task)
    assert checksum_paths == []
    assert second == first

    stat = destination.stat()
    os.utime(destination, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    checksum_paths.clear()
    materialize(task)
    assert checksum_paths == [destination.with_name(".prepared.tmp.wav")]

    checksum_paths.clear()
    changed_task = MaterializationTask(
        source=source,
        destination=destination,
        source_sha256="0" * 64,
    )
    materialize(changed_task)
    assert checksum_paths == [destination.with_name(".prepared.tmp.wav")]


def test_materialize_does_not_reuse_journal_record_with_non_wav_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.wav"
    destination = tmp_path / "prepared.wav"
    sf.write(source, np.zeros(240, dtype=np.float32), 24_000, subtype="PCM_16")
    task = MaterializationTask(source, destination)
    monkeypatch.setattr(audio_module.shutil, "which", lambda _name: None)
    materialize(task)

    journal = audio_module._state_path(destination)
    record = json.loads(journal.read_text(encoding="utf-8"))
    record["format"] = "FLAC"
    journal.write_text(json.dumps(record) + "\n", encoding="utf-8")
    audio_module._journal_snapshots.pop(audio_module._journal_path(destination), None)
    conversions: list[Path] = []
    real_materialize = audio_module._materialize_with_soundfile

    def recording_materialize(task: MaterializationTask, temporary: Path) -> None:
        conversions.append(task.destination)
        real_materialize(task, temporary)

    monkeypatch.setattr(audio_module, "_materialize_with_soundfile", recording_materialize)

    materialize(task)

    assert conversions == [destination]


def test_materialize_rejects_same_size_output_replacement_with_restored_mtime(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.wav"
    destination = tmp_path / "prepared.wav"
    sf.write(source, np.zeros(240, dtype=np.float32), 24_000, subtype="PCM_16")
    task = MaterializationTask(
        source=source,
        destination=destination,
        source_sha256=sha256_file(source),
    )
    monkeypatch.setattr(audio_module.shutil, "which", lambda _name: None)
    first = materialize(task)
    original_stat = destination.stat()
    replacement = destination.with_suffix(".replacement.wav")
    replacement.write_bytes(destination.read_bytes())
    payload = bytearray(replacement.read_bytes())
    payload[-1] ^= 1
    replacement.write_bytes(payload)
    os.utime(replacement, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    replacement.replace(destination)
    conversions: list[Path] = []
    real_materialize = audio_module._materialize_with_soundfile

    def recording_materialize(task: MaterializationTask, temporary: Path) -> None:
        conversions.append(task.destination)
        real_materialize(task, temporary)

    monkeypatch.setattr(audio_module, "_materialize_with_soundfile", recording_materialize)

    second = materialize(task)

    assert conversions == [destination]
    assert second.sha256 == first.sha256


def test_materialization_journal_does_not_follow_symlinks(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.wav"
    destination = tmp_path / "prepared" / "audio.wav"
    destination.parent.mkdir()
    sf.write(source, np.zeros(240, dtype=np.float32), 24_000, subtype="PCM_16")
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    audio_module._state_path(destination).symlink_to(sentinel)
    monkeypatch.setattr(audio_module.shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError, match="journal must not be a symlink"):
        materialize(MaterializationTask(source, destination))

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_materialize_all_parallel_work_is_bounded_and_destination_sorted(monkeypatch) -> None:
    active = 0
    maximum_active = 0
    lock = threading.Lock()
    pending_sizes: list[int] = []
    real_wait = audio_module.wait

    def fake_materialize_batch(tasks: list[MaterializationTask]) -> list[AudioMetadata]:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            time.sleep(0.02)
            return [
                AudioMetadata(
                    path=task.destination,
                    sample_rate_hz=task.sample_rate_hz,
                    channels=task.channels,
                    num_frames=1,
                    format="WAV",
                    subtype="PCM_16",
                    sha256=task.destination.stem,
                )
                for task in tasks
            ]
        finally:
            with lock:
                active -= 1

    def recording_wait(futures, *, return_when):
        pending_sizes.append(len(futures))
        return real_wait(futures, return_when=return_when)

    monkeypatch.setattr(audio_module, "_FFMPEG_BATCH_SIZE", 2)
    monkeypatch.setattr(audio_module.shutil, "which", lambda _name: "/ffmpeg")
    monkeypatch.setattr(audio_module, "_materialize_batch", fake_materialize_batch)
    monkeypatch.setattr(audio_module, "wait", recording_wait)
    tasks = [
        MaterializationTask(Path(f"source-{name}"), Path(f"output-{name}"))
        for name in reversed("abcdefghijkl")
    ]

    results = materialize_all(tasks, workers=3)

    assert [metadata.path for metadata in results] == sorted(task.destination for task in tasks)
    assert maximum_active == 3
    assert max(pending_sizes) == 6


def test_ffmpeg_command_scopes_inputs_outputs_metadata_and_threads(tmp_path: Path) -> None:
    conversions = (
        (
            MaterializationTask(tmp_path / "first.flac", tmp_path / "first.wav"),
            tmp_path / ".first.tmp.wav",
        ),
        (
            MaterializationTask(
                tmp_path / "second.mp3",
                tmp_path / "second.wav",
                sample_rate_hz=16_000,
            ),
            tmp_path / ".second.tmp.wav",
        ),
    )

    command = audio_module._ffmpeg_command(conversions)

    assert command.count("-filter_threads") == 1
    assert command.count("-threads:a") == 4
    assert command.count("-i") == 2
    assert max(index for index, value in enumerate(command) if value == "-i") < command.index(
        "-map"
    )
    assert [command[index + 1] for index, value in enumerate(command) if value == "-map"] == [
        "0:a:0",
        "1:a:0",
    ]
    assert [
        command[index + 1] for index, value in enumerate(command) if value == "-map_metadata:g"
    ] == ["0:g", "1:g"]
    assert [
        command[index + 1] for index, value in enumerate(command) if value == "-map_chapters"
    ] == ["0", "1"]
    assert [command[index + 1] for index, value in enumerate(command) if value == "-ar"] == [
        "24000",
        "16000",
    ]


def test_ffmpeg_command_scopes_explicit_channel_averages(tmp_path: Path) -> None:
    conversions = (
        (
            MaterializationTask(tmp_path / "first.wav", tmp_path / "first-output.wav"),
            tmp_path / ".first.tmp.wav",
        ),
        (
            MaterializationTask(tmp_path / "second.wav", tmp_path / "second-output.wav"),
            tmp_path / ".second.tmp.wav",
        ),
    )

    command = audio_module._ffmpeg_command(conversions, channel_averages={1: 30})

    assert command.count("-filter:a") == 1
    filter_index = command.index("-filter:a")
    assert command[filter_index + 1] == (
        "pan=mono|c0<" + "+".join(f"c{index}" for index in range(30))
    )
    map_indexes = [index for index, value in enumerate(command) if value == "-map"]
    assert map_indexes[0] < map_indexes[1] < filter_index
    assert filter_index < command.index("-ac", filter_index)


def test_materialize_all_uses_bounded_ffmpeg_batches_sharded_journal_and_reuse(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(240, dtype=np.float32), 24_000, subtype="PCM_16")
    tasks = [
        MaterializationTask(source, tmp_path / "prepared" / f"{index:02d}.wav")
        for index in range(audio_module._FFMPEG_BATCH_SIZE + 1)
    ]
    commands: list[list[str]] = []
    fsync_calls: list[int] = []

    def fake_run(command: list[str], **_kwargs):
        commands.append(command)
        for index, value in enumerate(command):
            if value == "-f" and command[index + 1] == "wav":
                sf.write(
                    command[index + 2],
                    np.zeros(240, dtype=np.float32),
                    24_000,
                    subtype="PCM_16",
                )
        return audio_module.subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(audio_module.shutil, "which", lambda _name: "/ffmpeg")
    monkeypatch.setattr(audio_module.subprocess, "run", fake_run)
    monkeypatch.setattr(audio_module.os, "fsync", fsync_calls.append)

    first = materialize_all(tasks, workers=2)

    assert sorted(command.count("-i") for command in commands) == [1, 32]
    assert len({metadata.sha256 for metadata in first}) == 1
    journal_paths = {audio_module._state_path(task.destination) for task in tasks}
    assert len(journal_paths) == 1
    journal = journal_paths.pop()
    assert len(journal.read_text(encoding="utf-8").splitlines()) == len(tasks)
    assert len(fsync_calls) == 2

    commands.clear()
    fsync_calls.clear()
    second = materialize_all(reversed(tasks), workers=2)
    assert commands == []
    assert fsync_calls == []
    assert second == first


def test_materialize_all_bounds_journal_snapshots_and_still_reuses(
    tmp_path: Path, monkeypatch
) -> None:
    audio_module._journal_snapshots.clear()
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(240, dtype=np.float32), 24_000, subtype="PCM_16")
    source_sha256 = sha256_file(source)
    task_count = audio_module._JOURNAL_SNAPSHOT_CACHE_SIZE + 8
    tasks = [
        MaterializationTask(
            source,
            tmp_path / "prepared" / f"shard-{index:02d}" / "audio.wav",
            source_sha256=source_sha256,
        )
        for index in range(task_count)
    ]
    conversions: list[Path] = []
    peak_snapshots = 0
    snapshot_lock = threading.Lock()
    real_materialize = audio_module._materialize_with_soundfile
    real_snapshot = audio_module._journal_snapshot

    def recording_materialize(task: MaterializationTask, temporary: Path) -> None:
        conversions.append(task.destination)
        real_materialize(task, temporary)

    def recording_snapshot(path: Path):
        nonlocal peak_snapshots
        snapshot = real_snapshot(path)
        with snapshot_lock:
            peak_snapshots = max(peak_snapshots, len(audio_module._journal_snapshots))
        return snapshot

    monkeypatch.setattr(audio_module.shutil, "which", lambda _name: None)
    monkeypatch.setattr(audio_module, "_materialize_with_soundfile", recording_materialize)
    monkeypatch.setattr(audio_module, "_journal_snapshot", recording_snapshot)

    first = materialize_all(tasks, workers=4)

    assert peak_snapshots <= audio_module._JOURNAL_SNAPSHOT_CACHE_SIZE
    assert len(audio_module._journal_snapshots) == audio_module._JOURNAL_SNAPSHOT_CACHE_SIZE
    assert set(conversions) == {task.destination for task in tasks}
    assert len(conversions) == len(tasks)

    conversions.clear()
    second = materialize_all(reversed(tasks), workers=4)

    assert second == first
    assert conversions == []
    assert len(audio_module._journal_snapshots) == audio_module._JOURNAL_SNAPSHOT_CACHE_SIZE


def test_sparse_journals_preserve_full_ffmpeg_batches(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(240, dtype=np.float32), 24_000, subtype="PCM_16")
    source_sha256 = sha256_file(source)
    tasks = [
        MaterializationTask(
            source,
            tmp_path / "prepared" / f"shard-{index:02d}" / "audio.wav",
            source_sha256=source_sha256,
        )
        for index in range(audio_module._FFMPEG_BATCH_SIZE * 2)
    ]
    batch_sizes: list[int] = []
    real_materialize = audio_module._materialize_with_soundfile

    def recording_batch(pending) -> None:
        batch_sizes.append(len(pending))
        for item in pending:
            real_materialize(item.task, item.temporary)

    monkeypatch.setattr(audio_module.shutil, "which", lambda _name: "/ffmpeg")
    monkeypatch.setattr(audio_module, "_materialize_with_ffmpeg_batch", recording_batch)

    materialize_all(tasks, workers=1)

    assert batch_sizes == [audio_module._FFMPEG_BATCH_SIZE] * 2


def test_truncated_journal_tail_redoes_only_the_unrecorded_output(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(240, dtype=np.float32), 24_000, subtype="PCM_16")
    tasks = [
        MaterializationTask(
            source,
            tmp_path / "prepared" / f"{index}.wav",
            source_sha256=sha256_file(source),
        )
        for index in range(2)
    ]
    converted: list[Path] = []
    real_materialize = audio_module._materialize_with_soundfile

    def recording_materialize(task: MaterializationTask, temporary: Path) -> None:
        converted.append(task.destination)
        real_materialize(task, temporary)

    monkeypatch.setattr(audio_module.shutil, "which", lambda _name: None)
    monkeypatch.setattr(audio_module, "_materialize_with_soundfile", recording_materialize)

    materialize_all(tasks, workers=1)
    assert converted == [task.destination for task in tasks]

    journal = audio_module._state_path(tasks[0].destination)
    contents = journal.read_bytes()
    journal.write_bytes(contents[:-10])
    converted.clear()

    materialize(tasks[0])
    materialize(tasks[1])

    assert converted == [tasks[1].destination]
    assert len(journal.read_text(encoding="utf-8").splitlines()) == 2


@pytest.mark.skipif(
    audio_module.shutil.which("ffmpeg") is None,
    reason="ffmpeg is required for byte-parity testing",
)
def test_ffmpeg_batch_output_is_byte_identical_to_individual_output(tmp_path: Path) -> None:
    frames = np.arange(4_800, dtype=np.float32)
    wav_source = tmp_path / "source.wav"
    flac_source = tmp_path / "source.flac"
    sf.write(
        wav_source,
        np.column_stack(
            (
                np.sin(2 * np.pi * 440 * frames / 48_000),
                np.sin(2 * np.pi * 660 * frames / 48_000),
            )
        ),
        48_000,
        subtype="FLOAT",
    )
    sf.write(flac_source, np.sin(2 * np.pi * 220 * frames / 16_000), 16_000)
    sources = (wav_source, flac_source)
    individual_tasks = [
        MaterializationTask(source, tmp_path / "individual" / f"{index}.wav")
        for index, source in enumerate(sources)
    ]
    batch_tasks = [
        MaterializationTask(source, tmp_path / "batch" / f"{index}.wav")
        for index, source in enumerate(sources)
    ]

    individual = [materialize(task) for task in individual_tasks]
    batched = materialize_all(batch_tasks, workers=1)

    assert [item.sha256 for item in batched] == [item.sha256 for item in individual]
    for individual_task, batch_task in zip(individual_tasks, batch_tasks, strict=True):
        assert batch_task.destination.read_bytes() == individual_task.destination.read_bytes()


@pytest.mark.skipif(
    audio_module.shutil.which("ffmpeg") is None,
    reason="ffmpeg is required for multichannel downmix testing",
)
def test_ffmpeg_materializes_unlabeled_30_channel_wav(tmp_path: Path) -> None:
    frame_indexes = np.arange(480, dtype=np.float32)
    samples = np.column_stack(
        [
            0.1 * np.sin(2 * np.pi * frequency * frame_indexes / 24_000)
            for frequency in range(100, 130)
        ]
    )
    source = tmp_path / "source.wav"
    destination = tmp_path / "prepared.wav"
    sf.write(source, samples, 24_000, subtype="FLOAT")

    metadata = materialize(MaterializationTask(source, destination))
    written, sample_rate = sf.read(destination, dtype="float32")

    assert metadata.channels == 1
    assert sample_rate == 24_000
    np.testing.assert_allclose(written, samples.mean(axis=1), atol=2 / 32_768)


@pytest.mark.skipif(
    audio_module.shutil.which("ffmpeg") is None,
    reason="ffmpeg is required for multichannel downmix testing",
)
def test_failed_rematrix_batch_preserves_supported_multichannel_output(
    tmp_path: Path,
) -> None:
    frames = np.arange(480, dtype=np.float32)
    supported_samples = np.column_stack(
        [0.1 * np.sin(2 * np.pi * frequency * frames / 24_000) for frequency in range(100, 106)]
    )
    unsupported_samples = np.column_stack(
        [0.1 * np.sin(2 * np.pi * frequency * frames / 24_000) for frequency in range(200, 230)]
    )
    supported = tmp_path / "supported.wav"
    unsupported = tmp_path / "unsupported.wav"
    sf.write(supported, supported_samples, 24_000, subtype="FLOAT")
    sf.write(unsupported, unsupported_samples, 24_000, subtype="FLOAT")
    reference = MaterializationTask(supported, tmp_path / "reference.wav")
    batched_supported = MaterializationTask(supported, tmp_path / "batch" / "supported.wav")
    batched_unsupported = MaterializationTask(
        unsupported,
        tmp_path / "batch" / "unsupported.wav",
    )

    reference_metadata = materialize(reference)
    batch_metadata = materialize_all(
        [batched_supported, batched_unsupported],
        workers=1,
    )

    assert batch_metadata[0].sha256 == reference_metadata.sha256
    assert batched_supported.destination.read_bytes() == reference.destination.read_bytes()
    assert batch_metadata[1].channels == 1


def test_failed_rematrix_batch_retries_only_the_failing_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    multichannel = tmp_path / "multichannel.wav"
    mono = tmp_path / "mono.wav"
    sf.write(multichannel, np.zeros((240, 30), dtype=np.float32), 24_000)
    sf.write(mono, np.zeros(240, dtype=np.float32), 24_000)
    tasks = [
        MaterializationTask(mono, tmp_path / "prepared" / "mono.wav"),
        MaterializationTask(multichannel, tmp_path / "prepared" / "multi.wav"),
    ]
    commands: list[list[str]] = []

    def fail_batch_then_multichannel(command: list[str], **_kwargs):
        commands.append(command)
        if command.count("-i") == 2:
            raise audio_module.subprocess.CalledProcessError(
                1,
                command,
                stderr="batch failed while processing decoded data for stream #1:0",
            )
        source = command[command.index("-i") + 1]
        if source == str(multichannel) and "-filter:a" not in command:
            raise audio_module.subprocess.CalledProcessError(
                1,
                command,
                stderr=(
                    "REMATRIX is needed between 30 channels and mono, but there is\n"
                    "NOT ENOUGH INFORMATION to do it"
                ),
            )
        for index, value in enumerate(command):
            if value == "-f" and command[index + 1] == "wav":
                sf.write(
                    command[index + 2],
                    np.zeros(240, dtype=np.float32),
                    24_000,
                    subtype="PCM_16",
                )
        return audio_module.subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(audio_module.shutil, "which", lambda _name: "/ffmpeg")
    monkeypatch.setattr(audio_module.subprocess, "run", fail_batch_then_multichannel)

    results = materialize_all(tasks, workers=1)

    assert [command.count("-i") for command in commands] == [2, 1, 1, 1]
    assert ["-filter:a" in command for command in commands] == [
        False,
        False,
        False,
        True,
    ]
    assert [metadata.path for metadata in results] == sorted(task.destination for task in tasks)


def test_failed_ffmpeg_batch_falls_back_to_individual_conversion(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(240, dtype=np.float32), 24_000, subtype="PCM_16")
    tasks = [
        MaterializationTask(source, tmp_path / "prepared" / f"{index}.wav") for index in range(3)
    ]
    individual_calls: list[Path] = []

    def fail_batch(pending) -> None:
        for item in pending:
            item.temporary.write_bytes(b"partial")
        raise audio_module._FFmpegBatchError("expected failure")

    def fake_individual(task: MaterializationTask, temporary: Path) -> None:
        assert not temporary.exists()
        individual_calls.append(task.destination)
        temporary.write_bytes(task.source.read_bytes())

    monkeypatch.setattr(audio_module.shutil, "which", lambda _name: "/ffmpeg")
    monkeypatch.setattr(audio_module, "_materialize_with_ffmpeg_batch", fail_batch)
    monkeypatch.setattr(audio_module, "_materialize_with_ffmpeg", fake_individual)

    results = materialize_all(tasks, workers=1)

    assert individual_calls == [task.destination for task in tasks]
    assert [item.path for item in results] == [task.destination for task in tasks]
    assert all(audio_module._state_path(task.destination).is_file() for task in tasks)
    assert not list(tmp_path.rglob("*.tmp.wav"))


def test_batch_validation_failure_preserves_old_destinations_and_cleans_temps(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(240, dtype=np.float32), 24_000, subtype="PCM_16")
    tasks = [
        MaterializationTask(source, tmp_path / "prepared" / f"{index}.wav") for index in range(3)
    ]
    original = {}
    original_states = {}
    for index, task in enumerate(tasks):
        task.destination.parent.mkdir(parents=True, exist_ok=True)
        original[task.destination] = f"old-{index}".encode()
        task.destination.write_bytes(original[task.destination])
        state_path = audio_module._state_path(task.destination)
        original_states[state_path] = f'{{"old":{index}}}\n'.encode()
        state_path.write_bytes(original_states[state_path])

    def write_invalid_batch(pending) -> None:
        for index, item in enumerate(pending):
            item.temporary.write_bytes(b"invalid" if index == 1 else item.task.source.read_bytes())

    monkeypatch.setattr(audio_module.shutil, "which", lambda _name: "/ffmpeg")
    monkeypatch.setattr(audio_module, "_materialize_with_ffmpeg_batch", write_invalid_batch)

    with pytest.raises(RuntimeError):
        materialize_all(tasks, workers=1)

    assert {task.destination: task.destination.read_bytes() for task in tasks} == original
    assert {path: path.read_bytes() for path in original_states} == original_states
    assert not list(tmp_path.rglob("*.tmp.wav"))


def test_failed_individual_fallback_preserves_the_entire_old_batch(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(240, dtype=np.float32), 24_000, subtype="PCM_16")
    tasks = [
        MaterializationTask(source, tmp_path / "prepared" / f"{index}.wav") for index in range(3)
    ]
    original = {}
    for index, task in enumerate(tasks):
        task.destination.parent.mkdir(parents=True, exist_ok=True)
        original[task.destination] = f"old-{index}".encode()
        task.destination.write_bytes(original[task.destination])

    def fail_batch(_pending) -> None:
        raise audio_module._FFmpegBatchError("expected batch failure")

    def fail_second_individual(task: MaterializationTask, temporary: Path) -> None:
        if task.destination == tasks[1].destination:
            raise RuntimeError("expected individual failure")
        temporary.write_bytes(task.source.read_bytes())

    monkeypatch.setattr(audio_module.shutil, "which", lambda _name: "/ffmpeg")
    monkeypatch.setattr(audio_module, "_materialize_with_ffmpeg_batch", fail_batch)
    monkeypatch.setattr(audio_module, "_materialize_with_ffmpeg", fail_second_individual)

    with pytest.raises(RuntimeError, match="expected individual failure"):
        materialize_all(tasks, workers=1)

    assert {task.destination: task.destination.read_bytes() for task in tasks} == original
    assert not any(audio_module._state_path(task.destination).exists() for task in tasks)
    assert not list(tmp_path.rglob("*.tmp.wav"))
