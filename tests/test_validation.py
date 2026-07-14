from __future__ import annotations

import os
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf

import lrac_data.validation as validation_module
from lrac_data.audio import MaterializationTask, materialize
from lrac_data.manifests import write_manifest
from lrac_data.models import AudioFormat, ManifestItem, MediaKind, Split, qualify_id
from lrac_data.validation import validate_manifests


def _manifest_item(
    audio_path: Path,
    *,
    workspace: Path,
    checksum: str,
    source_id: str | None = None,
    split: Split = Split.TRAIN,
    speaker_id: str | None = None,
    sample_rate_hz: int = 24_000,
    channels: int = 1,
    frame_count: int = 240,
) -> ManifestItem:
    source_id = source_id or audio_path.stem
    return ManifestItem(
        id=qualify_id("fixture", source_id),
        dataset="fixture",
        media_kind=MediaKind.SPEECH if speaker_id is not None else MediaKind.NOISE,
        audio_path=audio_path.relative_to(workspace).as_posix(),
        source_release="fixture-v1",
        source_id=source_id,
        split=split,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        frame_count=frame_count,
        checksum=checksum,
        speaker_id=speaker_id,
    )


def test_validation_rejects_blank_jsonl_lines(tmp_path: Path) -> None:
    manifest = tmp_path / "train.jsonl"
    manifest.write_text("\n", encoding="utf-8")

    report = validate_manifests([manifest], workspace=tmp_path)

    assert not report.ok
    assert report.errors == (f"{manifest}:1: blank JSONL line",)


def test_validation_reuses_known_audio_only_while_stat_is_unchanged(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.wav"
    audio_path = tmp_path / "prepared" / "audio.wav"
    manifest = tmp_path / "train.jsonl"
    sf.write(source, np.zeros(240, dtype=np.float32), 24_000, subtype="PCM_16")
    metadata = materialize(MaterializationTask(source, audio_path))
    write_manifest(
        manifest, [_manifest_item(audio_path, workspace=tmp_path, checksum=metadata.sha256)]
    )
    real_probe = validation_module.probe
    probed: list[Path] = []

    def recording_probe(path: Path, *, include_checksum: bool = True):
        probed.append(path)
        return real_probe(path, include_checksum=include_checksum)

    monkeypatch.setattr(validation_module, "probe", recording_probe)

    current = validate_manifests(
        [manifest],
        workspace=tmp_path,
        known_audio={audio_path: metadata},
    )
    assert current.ok
    assert probed == []

    reused_metadata = materialize(MaterializationTask(source, audio_path))
    deep_checked = validate_manifests(
        [manifest],
        workspace=tmp_path,
        known_audio={audio_path: reused_metadata},
    )
    assert deep_checked.ok
    assert probed == [audio_path]
    probed.clear()

    stat = audio_path.stat()
    payload = bytearray(audio_path.read_bytes())
    payload[-1] ^= 1
    audio_path.write_bytes(payload)
    os.utime(audio_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    stale = validate_manifests(
        [manifest],
        workspace=tmp_path,
        known_audio={audio_path: metadata},
    )
    assert not stale.ok
    assert stale.errors == ("fixture:audio: checksum mismatch",)
    assert probed == [audio_path]


def test_standalone_validation_deep_checks_audio(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.wav"
    audio_path = tmp_path / "prepared" / "audio.wav"
    manifest = tmp_path / "train.jsonl"
    sf.write(source, np.zeros(240, dtype=np.float32), 24_000, subtype="PCM_16")
    metadata = materialize(MaterializationTask(source, audio_path))
    write_manifest(
        manifest, [_manifest_item(audio_path, workspace=tmp_path, checksum=metadata.sha256)]
    )
    real_probe = validation_module.probe
    checksum_flags: list[bool] = []

    def recording_probe(path: Path, *, include_checksum: bool = True):
        checksum_flags.append(include_checksum)
        return real_probe(path, include_checksum=include_checksum)

    monkeypatch.setattr(validation_module, "probe", recording_probe)

    report = validate_manifests([manifest], workspace=tmp_path)

    assert report.ok
    assert checksum_flags == [True]


def test_parallel_validation_keeps_manifest_error_order(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "train.jsonl"
    records = [
        _manifest_item(tmp_path / f"{index}.wav", workspace=tmp_path, checksum="0" * 64)
        for index in range(6)
    ]
    write_manifest(manifest, records)
    active = 0
    maximum_active = 0
    lock = threading.Lock()

    def fake_inspect(check, **_kwargs):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            time.sleep(0.002 * (6 - int(check.item.source_id)))
            return (f"{check.item.id}: synthetic error",)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(validation_module, "_inspect_audio", fake_inspect)

    report = validate_manifests([manifest], workspace=tmp_path, workers=3)

    assert maximum_active == 3
    assert report.errors == tuple(f"fixture:{index}: synthetic error" for index in range(6))


def test_parallel_validation_rehashes_reused_audio_in_manifest_order(
    tmp_path: Path, monkeypatch
) -> None:
    manifest = tmp_path / "train.jsonl"
    records: list[ManifestItem] = []
    known_audio = {}
    for index in range(6):
        audio_path = tmp_path / f"{index}.wav"
        sf.write(audio_path, np.zeros(240, dtype=np.float32), 24_000, subtype="PCM_16")
        metadata = validation_module.probe(audio_path)
        known_audio[audio_path] = replace(metadata, checksum_fresh=False)
        records.append(_manifest_item(audio_path, workspace=tmp_path, checksum="0" * 64))
    write_manifest(manifest, records)

    real_probe = validation_module.probe
    barrier = threading.Barrier(3)
    active = 0
    maximum_active = 0
    checksum_flags: list[bool] = []
    lock = threading.Lock()

    def recording_probe(path: Path, *, include_checksum: bool = True):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
            checksum_flags.append(include_checksum)
        try:
            barrier.wait(timeout=2)
            return real_probe(path, include_checksum=include_checksum)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(validation_module, "probe", recording_probe)

    report = validate_manifests([manifest], workspace=tmp_path, known_audio=known_audio, workers=3)

    assert maximum_active == 3
    assert checksum_flags == [True] * 6
    assert report.errors == tuple(f"fixture:{index}: checksum mismatch" for index in range(6))


def test_validation_rejects_empty_training_partition(tmp_path: Path) -> None:
    manifests = [tmp_path / f"{name}.jsonl" for name in ("train", "validation", "evaluation")]
    for manifest in manifests:
        write_manifest(manifest, [])

    report = validate_manifests(
        manifests,
        workspace=tmp_path,
        expected_counts={"train": 0, "validation": 0, "evaluation": 0},
    )

    assert not report.ok
    assert "train.jsonl: training partition must not be empty" in report.errors


def test_validation_reconciles_recorded_partition_counts(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    sf.write(audio_path, np.zeros(240, dtype=np.float32), 24_000, subtype="PCM_16")
    metadata = validation_module.probe(audio_path)
    manifests = [tmp_path / f"{name}.jsonl" for name in ("train", "validation", "evaluation")]
    write_manifest(
        manifests[0],
        [_manifest_item(audio_path, workspace=tmp_path, checksum=metadata.sha256)],
    )
    write_manifest(manifests[1], [])
    write_manifest(manifests[2], [])

    report = validate_manifests(
        manifests,
        workspace=tmp_path,
        expected_counts={"train": 2, "validation": 0, "evaluation": 0},
    )

    assert "train.jsonl: run.json records 2 items, found 1" in report.errors


def test_validation_rejects_record_in_wrong_manifest_split(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    sf.write(audio_path, np.zeros(240, dtype=np.float32), 24_000, subtype="PCM_16")
    metadata = validation_module.probe(audio_path)
    manifest = tmp_path / "train.jsonl"
    write_manifest(
        manifest,
        [
            _manifest_item(
                audio_path,
                workspace=tmp_path,
                checksum=metadata.sha256,
                split=Split.VALIDATION,
            )
        ],
    )

    report = validate_manifests([manifest], workspace=tmp_path)

    assert any("does not match manifest partition 'train'" in error for error in report.errors)


def test_validation_rejects_train_validation_speaker_leakage(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    sf.write(audio_path, np.zeros(240, dtype=np.float32), 24_000, subtype="PCM_16")
    metadata = validation_module.probe(audio_path)
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    write_manifest(
        train,
        [
            _manifest_item(
                audio_path,
                workspace=tmp_path,
                checksum=metadata.sha256,
                source_id="train",
                speaker_id="speaker",
            )
        ],
    )
    write_manifest(
        validation,
        [
            _manifest_item(
                audio_path,
                workspace=tmp_path,
                checksum=metadata.sha256,
                source_id="validation",
                split=Split.VALIDATION,
                speaker_id="speaker",
            )
        ],
    )

    report = validate_manifests([train, validation], workspace=tmp_path)

    assert (
        "fixture:speaker: speaker appears in both train and validation manifests" in report.errors
    )


def test_validation_enforces_edition_audio_target_not_manifest_claims(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    sf.write(audio_path, np.zeros((240, 2), dtype=np.float32), 48_000, subtype="FLOAT")
    metadata = validation_module.probe(audio_path)
    manifest = tmp_path / "train.jsonl"
    write_manifest(
        manifest,
        [
            _manifest_item(
                audio_path,
                workspace=tmp_path,
                checksum=metadata.sha256,
                sample_rate_hz=48_000,
                channels=2,
            )
        ],
    )

    report = validate_manifests(
        [manifest],
        workspace=tmp_path,
        target_audio=AudioFormat(),
    )

    assert any("manifest declares 48000 Hz" in error for error in report.errors)
    assert any("manifest declares 2 channels" in error for error in report.errors)
    assert "fixture:audio: sample-rate mismatch" in report.errors
    assert "fixture:audio: channel-count mismatch" in report.errors
    assert "fixture:audio: sample-format mismatch" in report.errors


def test_validation_rejects_flac_bytes_renamed_as_wav(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    sf.write(
        audio_path,
        np.zeros(240, dtype=np.float32),
        24_000,
        format="FLAC",
        subtype="PCM_16",
    )
    metadata = validation_module.probe(audio_path)
    manifest = tmp_path / "train.jsonl"
    write_manifest(
        manifest,
        [_manifest_item(audio_path, workspace=tmp_path, checksum=metadata.sha256)],
    )

    report = validate_manifests(
        [manifest],
        workspace=tmp_path,
        target_audio=AudioFormat(),
    )

    assert metadata.format == "FLAC"
    assert "fixture:audio: container mismatch" in report.errors
    assert "fixture:audio: expected PCM_16 WAV audio" in report.errors
