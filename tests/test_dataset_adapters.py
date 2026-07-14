from __future__ import annotations

import gzip
import inspect
import json
from collections.abc import Iterable
from pathlib import Path

import pytest

from lrac_data.datasets import ADAPTERS, create_adapter
from lrac_data.datasets.io import DownloadRequest
from lrac_data.models import DatasetConfig, InventoryItem, MediaKind, SourceSpec


def _config(
    adapter: str,
    *media_kinds: MediaKind,
    options: dict[str, object] | None = None,
    sources: tuple[SourceSpec, ...] = (),
    expected_files: tuple[str, ...] = (),
) -> DatasetConfig:
    return DatasetConfig(
        id=adapter,
        adapter=adapter,
        release="fixture-v1",
        license="fixture-only",
        media_kinds=media_kinds,
        sources=sources,
        expected_files=expected_files,
        options=options or {},
    )


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")
    return path


def _inventory(config: DatasetConfig, repo_root: Path, workspace: Path) -> list[InventoryItem]:
    return create_adapter(config, repo_root, workspace).inventory()


def test_all_registered_adapters_are_concrete() -> None:
    assert ADAPTERS
    assert not [name for name, adapter in ADAPTERS.items() if inspect.isabstract(adapter)]


def test_adapter_workers_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="workers must be at least 1"):
        create_adapter(
            _config("fma", MediaKind.NOISE),
            tmp_path,
            tmp_path / "workspace",
            workers=0,
        )


def test_named_source_downloads_use_adapter_worker_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = SourceSpec(
        name="archive",
        url="https://example.invalid/archive",
        filename="archive.zip",
    )
    adapter = create_adapter(
        _config("fma", MediaKind.NOISE, sources=(source,)),
        tmp_path,
        tmp_path / "workspace",
        workers=7,
    )
    worker_counts: list[int] = []

    def fake_download_many(
        requests: Iterable[DownloadRequest],
        *,
        max_workers: int,
        downloader: object,
    ) -> list[Path]:
        del downloader
        worker_counts.append(max_workers)
        return [request.destination for request in requests]

    monkeypatch.setattr(
        "lrac_data.datasets.base.dataset_io.download_many",
        fake_download_many,
    )

    assert adapter.download_remote_sources("archive") == [adapter.download_dir / "archive.zip"]
    assert worker_counts == [7]


def test_artifact_checksums_verify_without_changing_cache_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_source = SourceSpec(
        name="archive",
        url="https://example.invalid/archive",
        filename="archive.zip",
        checksum="md5:" + "1" * 32,
        options={"legacy": True},
    )
    pinned_source = legacy_source.model_copy(
        update={"artifact_checksums": {"archive.zip": "2" * 64}}
    )
    legacy = create_adapter(
        _config("fma", MediaKind.NOISE, sources=(legacy_source,)),
        tmp_path,
        tmp_path / "workspace",
    )
    pinned = create_adapter(
        _config("fma", MediaKind.NOISE, sources=(pinned_source,)),
        tmp_path,
        tmp_path / "workspace",
    )
    checksums: list[str | None] = []

    def fake_download(url: str, destination: Path, *, checksum: str | None = None) -> Path:
        del url
        checksums.append(checksum)
        return destination

    monkeypatch.setattr("lrac_data.datasets.base.dataset_io.download_file", fake_download)

    assert legacy.cache_namespace == pinned.cache_namespace
    pinned.download_remote_sources("archive")
    assert checksums == ["sha256:" + "2" * 64]


def test_named_source_downloads_preserve_order_and_ignore_unrequested_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = tuple(
        SourceSpec(
            name=name,
            url=f"https://example.invalid/{name}",
            filename=f"{name}.zip",
            checksum="sha256:" + str(index) * 64,
        )
        for index, name in enumerate(("metadata", "medium", "unused"), start=1)
    )
    adapter = create_adapter(
        _config("fma", MediaKind.NOISE, sources=sources),
        tmp_path,
        tmp_path / "workspace",
    )
    calls: list[tuple[str, Path, str | None]] = []

    def fake_download(
        url: str,
        destination: Path,
        *,
        checksum: str | None = None,
    ) -> Path:
        calls.append((url, destination, checksum))
        return destination

    monkeypatch.setattr("lrac_data.datasets.base.dataset_io.download_file", fake_download)

    paths = adapter.download_remote_sources("medium", "metadata")

    assert paths == [
        adapter.download_dir / "medium.zip",
        adapter.download_dir / "metadata.zip",
    ]
    assert {url.rsplit("/", 1)[-1] for url, _, _ in calls} == {"medium", "metadata"}
    assert all("unused" not in str(value) for call in calls for value in call if value is not None)


def test_named_source_download_rejects_runtime_destination_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unsafe = SourceSpec.model_construct(
        name="archive",
        url="https://example.invalid/archive",
        filename="../outside.zip",
        checksum=None,
        path=None,
        options={},
    )
    adapter = create_adapter(
        _config("fma", MediaKind.NOISE, sources=(unsafe,)),
        tmp_path,
        tmp_path / "workspace",
    )
    called = False

    def fake_download(*_args: object, **_kwargs: object) -> Path:
        nonlocal called
        called = True
        raise AssertionError("unsafe downloads must be rejected before I/O")

    monkeypatch.setattr("lrac_data.datasets.base.dataset_io.download_file", fake_download)

    with pytest.raises(ValueError, match="escapes its download directory"):
        adapter.download_remote_sources("archive")

    assert not called


def test_named_source_download_rejects_symlink_destination_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = SourceSpec(
        name="archive",
        url="https://example.invalid/archive",
        filename="archive.zip",
    )
    adapter = create_adapter(
        _config("fma", MediaKind.NOISE, sources=(source,)),
        tmp_path,
        tmp_path / "workspace",
    )
    adapter.download_dir.mkdir(parents=True)
    outside = tmp_path / "outside.zip"
    outside.write_bytes(b"keep")
    (adapter.download_dir / "archive.zip").symlink_to(outside)

    def fake_download(*_args: object, **_kwargs: object) -> Path:
        raise AssertionError("unsafe downloads must be rejected before I/O")

    monkeypatch.setattr("lrac_data.datasets.base.dataset_io.download_file", fake_download)

    with pytest.raises(ValueError, match="escapes its download directory"):
        adapter.download_remote_sources("archive")

    assert outside.read_bytes() == b"keep"


def test_dataset_cache_directory_rejects_intermediate_symlink_escape(tmp_path: Path) -> None:
    source = SourceSpec(
        name="archive",
        url="https://example.invalid/archive",
        filename="archive.zip",
    )
    workspace = tmp_path / "workspace"
    adapter = create_adapter(
        _config("fma", MediaKind.NOISE, sources=(source,)),
        tmp_path,
        workspace,
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    cache_parent = workspace / "downloads" / "fma"
    cache_parent.mkdir(parents=True)
    assert adapter.cache_namespace is not None
    (cache_parent / adapter.cache_namespace).symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="downloads directory"):
        adapter.download_remote_sources("archive")


def test_dns5_inventory_parses_speech_noise_and_rir(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    root = workspace / "extracted" / "dns5"
    _touch(root / "Track1_Headset" / "mnt" / "book_00001_chp_0001_reader_00002_1_seg_1.wav")
    _touch(root / "datasets_fullband" / "noise_fullband" / "audioset_000" / "noise.wav")
    _touch(root / "datasets_fullband" / "impulse_responses" / "rir.wav")

    records = _inventory(
        _config("dns5", MediaKind.SPEECH, MediaKind.NOISE, MediaKind.RIR),
        tmp_path,
        workspace,
    )

    assert [(item.id, item.media_kind) for item in records] == [
        ("dns5:book_00001_chp_0001_reader_00002_1_seg_1", MediaKind.SPEECH),
        ("dns5:noise", MediaKind.NOISE),
        ("dns5:rir", MediaKind.RIR),
    ]
    assert records[0].speaker_id == "dns5_reader_00002"


def test_dns5_inventory_uses_relative_rir_paths_for_duplicate_stems(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    rir_root = workspace / "extracted" / "dns5" / "datasets_fullband" / "impulse_responses"
    _touch(rir_root / "SLR26" / "largeroom" / "Room001" / "Room001-00001.wav")
    _touch(rir_root / "SLR26" / "mediumroom" / "Room001" / "Room001-00001.wav")
    _touch(rir_root / "SLR26" / "smallroom" / "Room001" / "Room001-00001.wav")
    _touch(rir_root / "AIR" / "air_type1_unique.wav")

    records = _inventory(_config("dns5", MediaKind.RIR), tmp_path, workspace)

    assert {item.source_id for item in records} == {
        "AIR/air_type1_unique",
        "SLR26/largeroom/Room001/Room001-00001",
        "SLR26/mediumroom/Room001/Room001-00001",
        "SLR26/smallroom/Room001/Room001-00001",
    }


def test_libritts_inventory_parses_transcript_and_gender(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    corpus = workspace / "extracted" / "libritts" / "LibriTTS"
    audio = _touch(corpus / "train-clean-100" / "1" / "2" / "1_2_000001_000000.wav")
    audio.with_suffix(".normalized.txt").write_text("Fixture sentence.\n", encoding="utf-8")
    (corpus / "speakers.tsv").write_text("READER\tGENDER\n1\tF\n", encoding="utf-8")

    records = _inventory(
        _config(
            "libritts",
            MediaKind.SPEECH,
            options={"splits": ["train-clean-100"]},
        ),
        tmp_path,
        workspace,
    )

    assert len(records) == 1
    assert records[0].id == "libritts:1_2_000001_000000"
    assert records[0].speaker_id == "libritts_1"
    assert records[0].text == "Fixture sentence."
    assert records[0].gender == "f"


def test_vctk_inventory_maps_mic_audio_to_shared_transcript(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    corpus = workspace / "extracted" / "vctk" / "VCTK-Corpus"
    _touch(corpus / "wav48_silence_trimmed" / "p225" / "p225_001_mic2.flac")
    _touch(corpus / "wav48_silence_trimmed" / "p232" / "p232_001_mic2.flac")
    _touch(corpus / "wav48_silence_trimmed" / "p257" / "p257_001_mic2.flac")
    transcript = corpus / "txt" / "p225" / "p225_001.txt"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text("VCTK fixture.\n", encoding="utf-8")
    (corpus / "speaker-info.txt").write_text(
        "ID AGE GENDER\n225 23 F\n232 30 M\n257 28 F\n",
        encoding="utf-8",
    )

    records = _inventory(_config("vctk", MediaKind.SPEECH), tmp_path, workspace)
    by_id = {record.id: record for record in records}

    assert set(by_id) == {
        "vctk:p225_001_mic2",
        "vctk:p232_001_mic2",
        "vctk:p257_001_mic2",
    }
    assert by_id["vctk:p225_001_mic2"].speaker_id == "vctk_p225"
    assert by_id["vctk:p225_001_mic2"].text == "VCTK fixture."
    assert by_id["vctk:p225_001_mic2"].gender == "f"
    assert by_id["vctk:p232_001_mic2"].speaker_id == "vctk_p232"
    assert by_id["vctk:p232_001_mic2"].gender == "m"
    assert by_id["vctk:p257_001_mic2"].speaker_id == "vctk_p257"
    assert by_id["vctk:p257_001_mic2"].gender == "f"


def test_ears_inventory_joins_json_metadata(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    root = workspace / "extracted" / "ears"
    _touch(root / "p001" / "foo.wav")
    metadata = root / "metadata" / "repo"
    metadata.mkdir(parents=True)
    (metadata / "transcripts.json").write_text(
        json.dumps({"foo": "EARS fixture."}), encoding="utf-8"
    )
    (metadata / "speaker_statistics.json").write_text(
        json.dumps({"p001": {"gender": "female"}}), encoding="utf-8"
    )

    records = _inventory(_config("ears", MediaKind.SPEECH), tmp_path, workspace)

    assert len(records) == 1
    assert records[0].id == "ears:p001_foo"
    assert records[0].speaker_id == "ears_p001"
    assert records[0].text == "EARS fixture."
    assert records[0].gender == "f"


def test_mls_inventory_reads_edition_transcript_archive(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    transcript = repo_root / "datafiles" / "mls" / "german_train_transcripts.gz"
    transcript.parent.mkdir(parents=True)
    with gzip.open(transcript, "wt", encoding="utf-8") as stream:
        stream.write("1_2_000001 Deutsche Testzeile.\n")
    workspace = tmp_path / "workspace"
    _touch(
        workspace
        / "extracted"
        / "mls"
        / "german"
        / "train"
        / "audio"
        / "1"
        / "2"
        / "1_2_000001.flac"
    )

    records = _inventory(
        _config("mls", MediaKind.SPEECH, options={"languages": ["german"]}),
        repo_root,
        workspace,
    )

    assert len(records) == 1
    assert records[0].id == "mls:mls_german_1_2_000001"
    assert records[0].speaker_id == "1"
    assert records[0].text == "Deutsche Testzeile."
    assert records[0].language == "german"


def test_globe_inventory_validates_index_and_referenced_audio(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    root = workspace / "extracted" / "globe"
    _touch(root / "train" / "audio" / "speaker-1" / "utterance.flac")
    index = root / "inventory.jsonl"
    index.write_text(
        json.dumps(
            {
                "source_id": "speaker-1_utterance",
                "source_path": "train/audio/speaker-1/utterance.flac",
                "speaker_id": "globe_speaker-1",
                "text": "GLOBE fixture.",
                "language": "en",
                "gender": "f",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = _inventory(_config("globe", MediaKind.SPEECH), tmp_path, workspace)

    assert len(records) == 1
    assert records[0].id == "globe:speaker-1_utterance"
    assert (
        records[0].source_path
        == (root / "train" / "audio" / "speaker-1" / "utterance.flac").resolve()
    )
    assert records[0].text == "GLOBE fixture."


def test_globe_fetch_uses_record_batches_and_configured_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as parquet

    config = _config(
        "globe",
        MediaKind.SPEECH,
        sources=(
            SourceSpec(
                name="parquet_shards",
                url="https://example.invalid/{index:04d}.parquet",
                filename="{index:04d}.parquet",
                artifact_checksums={
                    "0000.parquet": "0" * 64,
                    "0001.parquet": "1" * 64,
                },
                options={"first": 0, "last": 1},
            ),
        ),
        expected_files=("inventory.jsonl", "train/audio/**/*"),
    )
    adapter = create_adapter(
        config,
        tmp_path,
        tmp_path / "workspace",
        workers=6,
    )
    parquet_paths = [adapter.download_dir / f"{index:04d}.parquet" for index in range(2)]
    rows = [
        {
            "audio": {"bytes": f"audio-{index}".encode(), "path": f"u{index}.flac"},
            "speaker_id": f"s{index}",
            "transcript": f"Transcript {index}",
            "language": "en",
            "gender": "Female" if index == 0 else "Male",
        }
        for index in range(2)
    ]
    for path, row in zip(parquet_paths, rows, strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        parquet.write_table(pa.Table.from_pylist([row]), path)

    worker_counts: list[int] = []

    def fake_download_many(
        requests: Iterable[DownloadRequest],
        *,
        max_workers: int,
        downloader: object,
    ) -> list[Path]:
        del downloader
        assert len(list(requests)) == 2
        worker_counts.append(max_workers)
        return parquet_paths

    monkeypatch.setattr("lrac_data.datasets.globe.download_many", fake_download_many)

    adapter.fetch()

    assert worker_counts == [6]
    assert (adapter.extracted_dir / "train/audio/s0/u0.flac").read_bytes() == b"audio-0"
    assert (adapter.extracted_dir / "train/audio/s1/u1.flac").read_bytes() == b"audio-1"
    assert [record.id for record in adapter.inventory()] == ["globe:s0_u0", "globe:s1_u1"]

    replacement = {
        **rows[0],
        "audio": {"bytes": b"replacement-audio", "path": "u0.flac"},
    }
    parquet.write_table(pa.Table.from_pylist([replacement]), parquet_paths[0])

    adapter.fetch()

    assert worker_counts == [6, 6]
    assert (adapter.extracted_dir / "train/audio/s0/u0.flac").read_bytes() == b"replacement-audio"


def test_fma_inventory_qualifies_track_id(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _touch(workspace / "extracted" / "fma" / "fma_medium" / "000" / "000001.mp3")

    records = _inventory(_config("fma", MediaKind.NOISE), tmp_path, workspace)

    assert [(item.id, item.media_kind) for item in records] == [("fma:fma_000001", MediaKind.NOISE)]


def test_fsd50k_inventory_qualifies_clip_id(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _touch(workspace / "extracted" / "fsd50k" / "FSD50K.dev_audio" / "100001.wav")

    records = _inventory(_config("fsd50k", MediaKind.NOISE), tmp_path, workspace)

    assert [(item.id, item.media_kind) for item in records] == [
        ("fsd50k:fsd50k_100001", MediaKind.NOISE)
    ]


def test_wham_inventory_preserves_clip_id(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _touch(workspace / "extracted" / "wham" / "high_res_wham" / "audio" / "foo.wav")

    records = _inventory(_config("wham", MediaKind.NOISE), tmp_path, workspace)

    assert [(item.id, item.media_kind) for item in records] == [("wham:foo", MediaKind.NOISE)]


def test_motus_inventory_qualifies_rir_id(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _touch(workspace / "extracted" / "motus" / "raw_rirs" / "foo.wav")

    records = _inventory(_config("motus", MediaKind.RIR), tmp_path, workspace)

    assert [(item.id, item.media_kind) for item in records] == [("motus:motus_foo", MediaKind.RIR)]


def test_dns5_fetch_streams_speech_parts_without_joining_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(
        "dns5",
        MediaKind.SPEECH,
        MediaKind.RIR,
        options={"speech_part_suffixes": ["aa"]},
        sources=(
            SourceSpec(
                name="read_speech_parts",
                url="https://example.invalid/read_speech.part{suffix}",
                filename="read_speech.part{suffix}",
                artifact_checksums={"read_speech.tgz.partaa": "a" * 64},
            ),
            SourceSpec(
                name="impulse_responses",
                url="https://example.invalid/rir.tar",
                filename="rir.tar",
                artifact_checksums={"rir.tar": "b" * 64},
            ),
        ),
        expected_files=(
            "Track1_Headset/mnt/**/*.wav",
            "datasets_fullband/impulse_responses/**/*.wav",
        ),
    )
    adapter = create_adapter(config, tmp_path, tmp_path / "workspace")
    downloaded: list[Path] = []
    streamed_parts: list[Path] = []

    def fake_download(url: str, destination: Path, **_kwargs: object) -> Path:
        downloaded.append(_touch(destination))
        return destination

    def fake_stream(parts: Iterable[Path], destination: Path, **_kwargs: object) -> Path:
        streamed_parts.extend(parts)
        _touch(destination / "mnt" / "speaker" / "speech.wav")
        return destination

    def fake_extract(archive: Path, destination: Path, **_kwargs: object) -> Path:
        del archive
        _touch(destination / "datasets_fullband" / "impulse_responses" / "room" / "rir.wav")
        return destination

    monkeypatch.setattr("lrac_data.datasets.dns5.download_file", fake_download)
    monkeypatch.setattr("lrac_data.datasets.dns5.safe_extract_multipart_tar", fake_stream)
    monkeypatch.setattr("lrac_data.datasets.dns5.safe_extract_tar", fake_extract)

    adapter.fetch()

    joined = adapter.download_dir / "speech" / "read_speech.tgz"
    assert downloaded and all(path.is_file() for path in downloaded)
    assert streamed_parts == [adapter.download_dir / "speech" / "read_speech.tgz.partaa"]
    assert not joined.exists()


def test_dns5_fetch_retains_source_parts_when_streaming_extraction_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(
        "dns5",
        MediaKind.SPEECH,
        options={"speech_part_suffixes": ["aa"]},
        sources=(
            SourceSpec(
                name="read_speech_parts",
                url="https://example.invalid/read_speech.part{suffix}",
                filename="read_speech.part{suffix}",
                artifact_checksums={"read_speech.tgz.partaa": "a" * 64},
            ),
            SourceSpec(
                name="impulse_responses",
                url="https://example.invalid/rir.tar",
                filename="rir.tar",
                artifact_checksums={"rir.tar": "b" * 64},
            ),
        ),
    )
    adapter = create_adapter(config, tmp_path, tmp_path / "workspace")

    def fake_download(url: str, destination: Path, **_kwargs: object) -> Path:
        return _touch(destination)

    def fail_extract(parts: object, destination: Path, **_kwargs: object) -> Path:
        del parts, destination
        raise RuntimeError("interrupted extraction")

    monkeypatch.setattr("lrac_data.datasets.dns5.download_file", fake_download)
    monkeypatch.setattr("lrac_data.datasets.dns5.safe_extract_multipart_tar", fail_extract)

    with pytest.raises(RuntimeError, match="interrupted extraction"):
        adapter.fetch()

    assert (adapter.download_dir / "speech" / "read_speech.tgz.partaa").is_file()
    assert not (adapter.download_dir / "speech" / "read_speech.tgz").exists()


def test_fsd50k_fetch_removes_unsplit_archive_but_keeps_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(
        "fsd50k",
        MediaKind.NOISE,
        sources=(
            SourceSpec(
                name="audio_zip",
                url="https://example.invalid/FSD50K.dev_audio.zip",
                filename="FSD50K.dev_audio.zip",
            ),
        ),
        expected_files=("FSD50K.dev_audio/**/*.wav",),
    )
    adapter = create_adapter(config, tmp_path, tmp_path / "workspace")
    upstream: list[Path] = []

    def fake_download(url: str, destination: Path, **_kwargs: object) -> Path:
        upstream.append(_touch(destination))
        return destination

    def fake_unsplit(parts_dir: Path, archive_name: str, destination: Path) -> Path:
        del parts_dir, archive_name
        _touch(destination)
        destination.with_suffix(f"{destination.suffix}.inputs.json").write_text(
            "{}\n", encoding="utf-8"
        )
        return destination

    def fake_extract(archive: Path, destination: Path) -> Path:
        del archive
        _touch(destination / "FSD50K.dev_audio" / "clip.wav")
        return destination

    monkeypatch.setattr("lrac_data.datasets.fsd50k.download_file", fake_download)
    monkeypatch.setattr("lrac_data.datasets.fsd50k.unsplit_zip", fake_unsplit)
    monkeypatch.setattr("lrac_data.datasets.fsd50k.safe_extract_zip", fake_extract)

    adapter.fetch()

    joined = adapter.download_dir / "FSD50K.dev_audio.unsplit.zip"
    assert upstream and all(path.is_file() for path in upstream)
    assert not joined.exists()
    assert not joined.with_suffix(f"{joined.suffix}.inputs.json").exists()


def test_vctk_fetch_removes_reproducible_inner_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(
        "vctk",
        MediaKind.SPEECH,
        sources=(
            SourceSpec(
                name="corpus",
                url="https://example.invalid/vctk.zip",
                filename="vctk.zip",
                artifact_checksums={"vctk.zip": "a" * 64},
            ),
        ),
        expected_files=("VCTK-Corpus/wav48_silence_trimmed/**/*.flac",),
    )
    adapter = create_adapter(config, tmp_path, tmp_path / "workspace")
    outer: Path | None = None
    download_calls = 0
    outer_extractions = 0
    inner_extractions = 0

    def fake_download(url: str, destination: Path, **_kwargs: object) -> Path:
        nonlocal download_calls, outer
        download_calls += 1
        outer = _touch(destination)
        return destination

    def fake_extract(archive: Path, destination: Path) -> Path:
        nonlocal inner_extractions, outer_extractions
        if archive == outer:
            outer_extractions += 1
            _touch(destination / "VCTK-Corpus-0.92.zip")
        else:
            inner_extractions += 1
            _touch(destination / "wav48_silence_trimmed" / "p001" / "clip.flac")
        return destination

    monkeypatch.setattr("lrac_data.datasets.vctk.download_file", fake_download)
    monkeypatch.setattr("lrac_data.datasets.vctk.safe_extract_zip", fake_extract)

    adapter.fetch()

    inner = adapter.extracted_dir / "outer" / "VCTK-Corpus-0.92.zip"
    assert outer is not None and outer.is_file()
    assert not inner.exists()

    adapter.fetch()

    assert (download_calls, outer_extractions, inner_extractions) == (2, 2, 2)
    assert not inner.exists()
