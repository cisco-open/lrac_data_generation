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
from lrac_data.models import ManifestItem, MediaKind, Split, qualify_id
from lrac_data.validation import validate_manifests


def _manifest_item(audio_path: Path, *, workspace: Path, checksum: str) -> ManifestItem:
    return ManifestItem(
        id=qualify_id("fixture", audio_path.stem),
        dataset="fixture",
        media_kind=MediaKind.NOISE,
        audio_path=audio_path.relative_to(workspace).as_posix(),
        source_release="fixture-v1",
        source_id=audio_path.stem,
        split=Split.TRAIN,
        sample_rate_hz=24_000,
        channels=1,
        frame_count=240,
        checksum=checksum,
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
