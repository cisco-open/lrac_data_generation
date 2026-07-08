from __future__ import annotations

import json
from pathlib import Path

import pytest

from lrac_data.exporters.kaldi import export_kaldi


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(record, sort_keys=True)}\n" for record in records),
        encoding="utf-8",
    )


def _noise_record(source_id: str = "noise-a") -> dict[str, object]:
    return {
        "id": f"fixture:{source_id}",
        "schema_version": 1,
        "dataset": "fixture",
        "source_release": "fixture-v1",
        "source_id": source_id,
        "media_kind": "noise",
        "audio_path": f"prepared/{source_id}.wav",
        "split": "train",
        "sample_rate_hz": 24_000,
        "channels": 1,
        "frame_count": 100,
        "checksum": "c" * 64,
    }


def test_export_kaldi_matches_golden_files(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    manifest = workspace / "manifests" / "curated.jsonl"
    records: list[dict[str, object]] = [
        {
            "id": "fixture:speech-b",
            "schema_version": 1,
            "dataset": "fixture",
            "source_release": "fixture-v1",
            "source_id": "speech-b",
            "media_kind": "speech",
            "audio_path": "prepared/speech-b.wav",
            "split": "train",
            "sample_rate_hz": 24_000,
            "channels": 1,
            "frame_count": 100,
            "checksum": "b" * 64,
            "speaker_id": "speaker-1",
            "gender": "female",
            "text": "second utterance",
        },
        {
            "id": "fixture:noise-a",
            "schema_version": 1,
            "dataset": "fixture",
            "source_release": "fixture-v1",
            "source_id": "noise-a",
            "media_kind": "noise",
            "audio_path": "prepared/noise-a.wav",
            "split": "train",
            "sample_rate_hz": 24_000,
            "channels": 1,
            "frame_count": 100,
            "checksum": "c" * 64,
        },
        {
            "id": "fixture:speech-a",
            "schema_version": 1,
            "dataset": "fixture",
            "source_release": "fixture-v1",
            "source_id": "speech-a",
            "media_kind": "speech",
            "audio_path": "prepared/speech-a.wav",
            "split": "train",
            "sample_rate_hz": 24_000,
            "channels": 1,
            "frame_count": 100,
            "checksum": "a" * 64,
            "speaker_id": "speaker-1",
            "gender": "female",
            "text": "first utterance",
        },
    ]
    _write_jsonl(manifest, records)

    output = tmp_path / "kaldi"
    counts = export_kaldi(manifest, output, workspace=workspace)
    prepared = workspace / "prepared"

    assert counts == {
        "wav.scp": 3,
        "utt2fs": 3,
        "utt2category": 3,
        "spk1.scp": 2,
        "utt2spk": 2,
        "spk2utt": 1,
        "text": 2,
        "spk2gender": 1,
    }
    assert (output / "wav.scp").read_text(encoding="utf-8") == (
        f"fixture:noise-a {prepared / 'noise-a.wav'}\n"
        f"fixture:speech-a {prepared / 'speech-a.wav'}\n"
        f"fixture:speech-b {prepared / 'speech-b.wav'}\n"
    )
    assert (output / "spk1.scp").read_text(encoding="utf-8") == (
        f"fixture:speech-a {prepared / 'speech-a.wav'}\n"
        f"fixture:speech-b {prepared / 'speech-b.wav'}\n"
    )
    assert (output / "utt2spk").read_text(encoding="utf-8") == (
        "fixture:speech-a speaker-1\nfixture:speech-b speaker-1\n"
    )
    assert (output / "spk2utt").read_text(encoding="utf-8") == (
        "speaker-1 fixture:speech-a fixture:speech-b\n"
    )
    assert (output / "text").read_text(encoding="utf-8") == (
        "fixture:speech-a first utterance\nfixture:speech-b second utterance\n"
    )
    assert (output / "spk2gender").read_text(encoding="utf-8") == "speaker-1 f\n"
    assert (output / "utt2fs").read_text(encoding="utf-8") == (
        "fixture:noise-a 24000\nfixture:speech-a 24000\nfixture:speech-b 24000\n"
    )
    assert (output / "utt2category").read_text(encoding="utf-8") == (
        "fixture:noise-a 1ch_24000Hz\nfixture:speech-a 1ch_24000Hz\nfixture:speech-b 1ch_24000Hz\n"
    )


def test_export_kaldi_rejects_duplicate_manifest_ids(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    record: dict[str, object] = {
        "id": "fixture:duplicate",
        "schema_version": 1,
        "dataset": "fixture",
        "source_release": "fixture-v1",
        "source_id": "duplicate",
        "media_kind": "noise",
        "audio_path": "prepared.wav",
        "split": "train",
        "sample_rate_hz": 24_000,
        "channels": 1,
        "frame_count": 100,
        "checksum": "d" * 64,
    }
    _write_jsonl(manifest, [record, record])

    with pytest.raises(ValueError, match="duplicate manifest ID"):
        export_kaldi(manifest, tmp_path / "kaldi")


def test_export_kaldi_infers_workspace_from_canonical_manifest_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    manifest = workspace / "manifests" / "2026" / "curated" / "train.jsonl"
    _write_jsonl(manifest, [_noise_record()])

    output = tmp_path / "kaldi"
    export_kaldi(manifest, output)

    assert (output / "wav.scp").read_text(encoding="utf-8") == (
        f"fixture:noise-a {workspace / 'prepared' / 'noise-a.wav'}\n"
    )


def test_export_kaldi_replaces_the_complete_previous_generation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    speech_manifest = workspace / "manifests" / "speech.jsonl"
    noise_manifest = workspace / "manifests" / "noise.jsonl"
    speech = {
        **_noise_record("speech-a"),
        "id": "fixture:speech-a",
        "source_id": "speech-a",
        "media_kind": "speech",
        "speaker_id": "speaker-a",
        "text": "fixture speech",
    }
    _write_jsonl(speech_manifest, [speech])
    _write_jsonl(noise_manifest, [_noise_record()])
    output = tmp_path / "kaldi"

    export_kaldi(speech_manifest, output, workspace=workspace)
    assert (output / "utt2spk").is_file()

    export_kaldi(noise_manifest, output, workspace=workspace)

    assert {path.name for path in output.iterdir()} == {
        "utt2category",
        "utt2fs",
        "wav.scp",
    }


def test_export_kaldi_failure_preserves_the_previous_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    manifest = workspace / "manifests" / "train.jsonl"
    _write_jsonl(manifest, [_noise_record()])
    output = tmp_path / "kaldi"
    export_kaldi(manifest, output, workspace=workspace)
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    calls = 0

    def fail_during_staging(path: Path, text: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated write failure")
        path.write_text(text, encoding="utf-8")

    monkeypatch.setattr("lrac_data.exporters.kaldi.atomic_write_text", fail_during_staging)

    with pytest.raises(OSError, match="simulated write failure"):
        export_kaldi(manifest, output, workspace=workspace)

    assert {path.name: path.read_bytes() for path in output.iterdir()} == before
    assert not list(tmp_path.glob(".kaldi.staging-*"))


def test_export_kaldi_refuses_to_replace_an_unowned_directory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    manifest = workspace / "manifests" / "train.jsonl"
    _write_jsonl(manifest, [_noise_record()])
    output = tmp_path / "kaldi"
    output.mkdir()
    unrelated = output / "keep.txt"
    unrelated.write_text("not an LRAC export", encoding="utf-8")

    with pytest.raises(ValueError, match="not owned by lrac-data"):
        export_kaldi(manifest, output, workspace=workspace)

    assert unrelated.read_text(encoding="utf-8") == "not an LRAC export"
