from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

import lrac_data.pipeline as pipeline_module
from lrac_data.datasets.base import DatasetAdapter
from lrac_data.manifests import read_manifest
from lrac_data.models import InventoryItem, MediaKind, qualify_id
from lrac_data.pipeline import (
    WorkspaceLayout,
    _attach_source_checksums,
    _audio_implementation_fingerprint,
    _inventory_implementation_fingerprint,
    _prepared_path,
    _publish_manifest_set,
    _workspace_prepare_lock,
    prepare_edition,
    workspace_status,
)
from lrac_data.state import sha256_file


class FixtureAdapter(DatasetAdapter):
    fetch_calls = 0

    def fetch(self) -> Path:
        type(self).fetch_calls += 1
        return self.extracted_dir

    def inventory(self) -> list[InventoryItem]:
        return [
            self.item(
                source_id,
                MediaKind.SPEECH,
                self.repo_root / "audio" / f"{source_id}.wav",
                speaker_id=f"speaker-{source_id}",
                language="en",
                text=source_id,
            )
            for source_id in ("curated", "extra", "validation", "evaluation")
        ]


def _alternate_ffmpeg_command(_conversions: object) -> list[str]:
    return ["ffmpeg", "alternate"]


def _alternate_channel_average(_stderr: str) -> bool:
    return True


def test_workspace_creation_rejects_managed_directory_symlinks(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    (workspace / "extracted").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="managed workspace directory"):
        WorkspaceLayout.at(workspace).create()

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_workspace_prepare_lock_rejects_a_second_writer(tmp_path: Path) -> None:
    lock = tmp_path / "state" / "prepare.lock"
    lock.parent.mkdir(parents=True)

    with (
        _workspace_prepare_lock(lock),
        pytest.raises(RuntimeError, match="another preparation"),
        _workspace_prepare_lock(lock),
    ):
        pytest.fail("a second writer must not acquire the workspace lock")


def test_prepared_audio_path_cannot_follow_dataset_symlink(tmp_path: Path) -> None:
    layout = WorkspaceLayout.at(tmp_path / "workspace")
    layout.create()
    outside = tmp_path / "outside"
    outside.mkdir()
    (layout.prepared_audio / "fixture").symlink_to(outside, target_is_directory=True)
    item = InventoryItem(
        id="fixture:item",
        dataset="fixture",
        source_id="item",
        source_release="fixture-v1",
        media_kind=MediaKind.NOISE,
        source_path=tmp_path / "source.wav",
        source_checksum="0" * 64,
    )
    loaded = SimpleNamespace(
        config=SimpleNamespace(audio=SimpleNamespace(sample_rate_hz=24_000, channels=1))
    )

    with pytest.raises(ValueError, match="prepared audio path"):
        _prepared_path(
            item,
            loaded=loaded,
            layout=layout,
            implementation_fingerprint="implementation",
        )

    assert list(outside.iterdir()) == []


def test_source_digest_cache_rehashes_only_changed_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"one")
    item = InventoryItem(
        id="fixture:source",
        dataset="fixture",
        source_id="source",
        source_release="fixture-v1",
        media_kind=MediaKind.NOISE,
        source_path=source,
    )
    cache = tmp_path / "state" / "source-checksums.json"
    real_sha256 = pipeline_module.sha256_file
    calls: list[Path] = []

    def recording_sha256(path: Path) -> str:
        calls.append(path)
        return real_sha256(path)

    monkeypatch.setattr(pipeline_module, "sha256_file", recording_sha256)

    first = _attach_source_checksums((item,), workers=2, cache_path=cache)
    cache_data = json.loads(cache.read_text(encoding="utf-8"))
    cached_record = cache_data[str(source)]
    cached_record.update(cached_record.pop("identity"))
    cache.write_text(json.dumps(cache_data), encoding="utf-8")
    calls.clear()
    second = _attach_source_checksums((item,), workers=2, cache_path=cache)

    assert first[0].source_checksum == second[0].source_checksum
    assert calls == []

    previous = source.stat()
    replacement = source.with_suffix(".replacement")
    replacement.write_bytes(b"two")
    os.utime(replacement, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    replacement.replace(source)
    changed = _attach_source_checksums((item,), workers=2, cache_path=cache)

    assert calls == [source]
    assert changed[0].source_checksum != first[0].source_checksum


@pytest.mark.parametrize("cache_damage", ["missing", "invalid-utf8"])
def test_complete_curated_and_uncurated_runs_share_inventory_and_audio(
    tmp_path: Path, monkeypatch, cache_damage: str
) -> None:
    repo = tmp_path / "repo"
    audio = repo / "audio"
    audio.mkdir(parents=True)
    samples = np.linspace(-0.25, 0.25, 800, dtype=np.float32)
    for source_id in ("curated", "extra", "validation", "evaluation"):
        sf.write(audio / f"{source_id}.wav", samples, 16_000, subtype="PCM_16")

    config = repo / "configs" / "editions" / "fixture.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        """\
schema_version: 1
edition: fixture
audio: {sample_rate_hz: 24000, channels: 1, sample_format: pcm_s16le, container: wav}
datasets:
  - id: fixture
    adapter: fixture
    release: test
    license: test-only
    media_kinds: [speech]
exclusions:
  - name: validation
    partition: validation
    dataset: fixture
    source_ids: [validation]
  - name: evaluation
    partition: evaluation
    dataset: fixture
    source_ids: [evaluation]
curations:
  - name: quality
    dataset: fixture
    action: include
    source_ids: [curated, validation]
""",
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    FixtureAdapter.fetch_calls = 0
    monkeypatch.setattr(
        "lrac_data.pipeline.create_adapter",
        lambda *args, **kwargs: FixtureAdapter(*args, **kwargs),
    )
    monkeypatch.setattr(
        "lrac_data.pipeline._require_preparation_requirements", lambda _loaded: None
    )

    curated = prepare_edition("fixture", selection="curated", workspace=workspace, repo_root=repo)
    checksum_cache = workspace / "state" / "source-checksums" / "fixture.json"
    assert checksum_cache.is_file()
    if cache_damage == "missing":
        checksum_cache.unlink()
    else:
        checksum_cache.write_bytes(b"\xff")
    uncurated = prepare_edition(
        "fixture", selection="uncurated", workspace=workspace, repo_root=repo
    )

    assert curated.counts == {
        "training": 1,
        "validation": 1,
        "evaluation": 1,
        "quality_rejected": 1,
    }
    assert uncurated.counts == {
        "training": 2,
        "validation": 1,
        "evaluation": 1,
        "quality_rejected": 0,
    }
    assert FixtureAdapter.fetch_calls == 1
    assert checksum_cache.is_file()
    assert uncurated.resumed_datasets == ("fixture",)
    assert {item.source_id for item in read_manifest(curated.manifests["train"])} == {"curated"}
    assert {item.source_id for item in read_manifest(uncurated.manifests["train"])} == {
        "curated",
        "extra",
    }
    assert len(list((workspace / "prepared" / "audio").rglob("*.wav"))) == 4
    run = json.loads(
        (workspace / "manifests" / "fixture" / "uncurated" / "run.json").read_text(encoding="utf-8")
    )
    assert run["selection_policy"] == "all-eligible"
    assert run["counts"]["training"] == 2
    assert qualify_id("fixture", "extra") in {
        item.id for item in read_manifest(uncurated.manifests["train"])
    }


def test_changed_source_creates_distinct_run_and_preserves_old_audio(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    audio = repo / "audio"
    audio.mkdir(parents=True)
    source = audio / "selected.wav"
    sf.write(source, np.zeros(240, dtype=np.float32), 24_000, subtype="PCM_16")
    config = repo / "configs" / "editions" / "fixture.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        """\
schema_version: 1
edition: fixture
datasets:
  - id: fixture
    adapter: fixture
    release: test
    license: test-only
    media_kinds: [speech]
curations:
  - name: quality
    dataset: fixture
    source_ids: [selected]
""",
        encoding="utf-8",
    )

    class OneItemAdapter(DatasetAdapter):
        def fetch(self) -> Path:
            return self.extracted_dir

        def inventory(self) -> list[InventoryItem]:
            return [
                self.item(
                    "selected",
                    MediaKind.SPEECH,
                    source,
                    speaker_id="speaker",
                )
            ]

    monkeypatch.setattr(
        "lrac_data.pipeline.create_adapter",
        lambda *args, **kwargs: OneItemAdapter(*args, **kwargs),
    )
    monkeypatch.setattr(
        "lrac_data.pipeline._require_preparation_requirements", lambda _loaded: None
    )
    workspace = tmp_path / "workspace"
    first = prepare_edition("fixture", workspace=workspace, repo_root=repo)
    first_record = read_manifest(first.manifests["train"])[0]
    first_audio = workspace / first_record.audio_path
    first_digest = sha256_file(first_audio)

    sf.write(source, np.ones(240, dtype=np.float32) * 0.5, 24_000, subtype="PCM_16")
    second = prepare_edition("fixture", workspace=workspace, repo_root=repo)
    second_record = read_manifest(second.manifests["train"])[0]

    assert second.run_id != first.run_id
    assert second_record.audio_path != first_record.audio_path
    assert first_audio.is_file()
    assert sha256_file(first_audio) == first_digest
    assert (workspace / "runs" / first.run_id / "manifests" / "train.jsonl").is_file()
    assert {report["run_id"]: report["complete"] for report in workspace_status(workspace)} == {
        first.run_id: True,
        second.run_id: True,
    }


def test_manifest_publication_restores_stranded_backup_before_staging_failure(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "manifests" / "fixture" / "curated"
    backup = destination.parent / ".curated.old.previous"
    backup.mkdir(parents=True)
    (backup / "marker").write_text("previous", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        _publish_manifest_set(
            {"train": tmp_path / "missing.jsonl"},
            {},
            destination=destination,
            run_id="new",
        )

    assert (destination / "marker").read_text(encoding="utf-8") == "previous"


def test_implementation_fingerprints_are_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "src" / "lrac_data"
    datasets = package / "datasets"
    datasets.mkdir(parents=True)
    for relative_path in (
        "audio.py",
        "state.py",
        "models.py",
        "manifests.py",
        "cli.py",
        "datasets/__init__.py",
        "datasets/base.py",
        "datasets/common.py",
        "datasets/inventory.py",
        "datasets/io.py",
        "datasets/dns5.py",
        "datasets/ears.py",
        "datasets/fma.py",
        "datasets/fsd50k.py",
    ):
        path = package / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative_path, encoding="utf-8")
    (package / "audio.py").write_text(
        """\
def _ffmpeg_command(conversions):
    return ["ffmpeg", str(len(conversions))]

def _materialize_with_soundfile(task, temporary):
    return (task, temporary)

def cache_helper():
    return 1
""",
        encoding="utf-8",
    )
    for adapter in ("dns5", "ears", "fma", "fsd50k"):
        (datasets / f"{adapter}.py").write_text(
            f"""\
class {adapter.title()}Adapter:
    def fetch(self):
        return "first"

    def inventory(self):
        return ["{adapter}"]
""",
            encoding="utf-8",
        )

    audio_before = _audio_implementation_fingerprint(tmp_path, "ffmpeg fixture")
    dns_before = _inventory_implementation_fingerprint(tmp_path, "dns5")
    ears_before = _inventory_implementation_fingerprint(tmp_path, "ears")
    fma_before = _inventory_implementation_fingerprint(tmp_path, "fma")
    fsd50k_before = _inventory_implementation_fingerprint(tmp_path, "fsd50k")

    (package / "cli.py").write_text("changed CLI", encoding="utf-8")
    assert _audio_implementation_fingerprint(tmp_path, "ffmpeg fixture") == audio_before
    assert _inventory_implementation_fingerprint(tmp_path, "dns5") == dns_before

    with (package / "audio.py").open("a", encoding="utf-8") as stream:
        stream.write("\ndef unrelated_cache_change():\n    return 2\n")
    assert _audio_implementation_fingerprint(tmp_path, "ffmpeg fixture") == audio_before
    assert _inventory_implementation_fingerprint(tmp_path, "dns5") == dns_before

    monkeypatch.setattr(
        pipeline_module.audio_module,
        "_ffmpeg_command",
        _alternate_ffmpeg_command,
    )
    assert _audio_implementation_fingerprint(tmp_path, "ffmpeg fixture") != audio_before

    dns_path = datasets / "dns5.py"
    dns_path.write_text(
        dns_path.read_text(encoding="utf-8").replace('return "first"', 'return "faster"'),
        encoding="utf-8",
    )
    assert _inventory_implementation_fingerprint(tmp_path, "dns5") != dns_before
    assert _inventory_implementation_fingerprint(tmp_path, "ears") == ears_before
    dns_after_adapter_change = _inventory_implementation_fingerprint(tmp_path, "dns5")

    (datasets / "inventory.py").write_text("changed inventory", encoding="utf-8")
    assert _inventory_implementation_fingerprint(tmp_path, "fma") != fma_before
    assert _inventory_implementation_fingerprint(tmp_path, "fsd50k") != fsd50k_before
    assert _inventory_implementation_fingerprint(tmp_path, "dns5") != dns_after_adapter_change
    assert _inventory_implementation_fingerprint(tmp_path, "ears") != ears_before


def test_audio_fingerprint_includes_channel_average_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = _audio_implementation_fingerprint(tmp_path, "ffmpeg fixture")

    monkeypatch.setattr(
        pipeline_module.audio_module,
        "_needs_explicit_channel_average",
        _alternate_channel_average,
    )

    assert _audio_implementation_fingerprint(tmp_path, "ffmpeg fixture") != before
