from __future__ import annotations

import json
from pathlib import Path

import pytest
import soundfile as sf
from typer.testing import CliRunner

from lrac_data.audio import probe
from lrac_data.cli import app
from lrac_data.config import load_edition_config, portable_config_path, portable_config_payload
from lrac_data.manifests import write_manifest
from lrac_data.models import ManifestItem, MediaKind, Split, qualify_id
from lrac_data.state import fingerprint, sha256_file


def _write_cli_fixture(root: Path) -> None:
    datasets = root / "configs" / "datasets"
    editions = root / "configs" / "editions"
    datasets.mkdir(parents=True)
    editions.mkdir(parents=True)
    (datasets / "fixture.yaml").write_text(
        """\
id: fixture
adapter: fma
release: fixture-v1
license: fixture-only
media_kinds: [noise]
sources:
  - name: archive
    url: https://example.invalid/fixture.zip
    filename: fixture.zip
expected_files: []
""",
        encoding="utf-8",
    )
    (editions / "fixture.yaml").write_text(
        """\
schema_version: 1
edition: fixture
datasets: [fixture]
exclusions: []
curations: []
""",
        encoding="utf-8",
    )


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.mark.parametrize(
    ("selection", "policy"),
    [("curated", "curated"), ("uncurated", "all-eligible")],
)
def test_cli_plan_succeeds_without_downloads_or_filesystem_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selection: str,
    policy: str,
) -> None:
    repo = tmp_path / "repo"
    _write_cli_fixture(repo)
    before = _snapshot(tmp_path)

    def fail_download(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"plan attempted a download: {args!r}, {kwargs!r}")

    monkeypatch.setattr("lrac_data.datasets.io.download_file", fail_download)
    monkeypatch.setattr("lrac_data.planner.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("lrac_data.planner.importlib.util.find_spec", lambda name: object())
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "plan",
            "--edition",
            "fixture",
            "--selection",
            selection,
            "--repo-root",
            str(repo),
        ],
    )

    assert result.exit_code == 0, result.output
    assert f"Selection: {selection} ({policy})" in result.output
    assert "Plan is complete. No files were written." in result.output
    assert _snapshot(tmp_path) == before
    assert not (tmp_path / "workspace").exists()


def _write_published_group(
    workspace: Path,
    repo: Path,
    *,
    edition: str | Path = "fixture",
) -> tuple[Path, Path]:
    loaded = load_edition_config(edition, repo_root=repo, selection="curated")
    group = workspace / "manifests" / "fixture" / "curated"
    group.mkdir(parents=True)
    audio_path = workspace / "prepared" / "audio.wav"
    audio_path.parent.mkdir(parents=True)
    sf.write(audio_path, [0.0] * 240, 24_000, subtype="PCM_16")
    audio = probe(audio_path)
    item = ManifestItem(
        id=qualify_id("fixture", "train"),
        dataset="fixture",
        media_kind=MediaKind.NOISE,
        audio_path=audio_path.relative_to(workspace).as_posix(),
        source_release="fixture-v1",
        source_id="train",
        split=Split.TRAIN,
        sample_rate_hz=audio.sample_rate_hz,
        channels=audio.channels,
        frame_count=audio.num_frames,
        checksum=audio.sha256,
    )
    records = {"train": [item], "validation": [], "evaluation": []}
    manifests = {name: group / f"{name}.jsonl" for name in records}
    for name, path in manifests.items():
        write_manifest(path, records[name])

    zero_digest = "0" * 64
    metadata = {
        "schema_version": 1,
        "run_id": "fixture-run",
        "edition": "fixture",
        "selection": "curated",
        "selection_policy": "curated",
        "config_path": portable_config_path(loaded),
        "config_fingerprint": fingerprint(portable_config_payload(loaded)),
        "implementation_fingerprint": zero_digest,
        "dependency_lock_digest": None,
        "input_fingerprint": zero_digest,
        "run_fingerprint": zero_digest,
        "counts": {
            "training": 1,
            "validation": 0,
            "evaluation": 0,
            "withheld": 0,
            "quality_rejected": 0,
        },
        "inventory_digests": {"fixture": zero_digest},
        "selected_source_digest": zero_digest,
        "inventory_counts": {"fixture": {"noise": 1}},
        "source_artifacts": {"fixture": []},
        "manifests": {
            name: {
                "path": path.relative_to(workspace).as_posix(),
                "sha256": sha256_file(path),
            }
            for name, path in manifests.items()
        },
        "timings_seconds": {
            "datasets": {"fixture": 0.0},
            "selection_and_source_hashing": 0.0,
            "materialization": 0.0,
            "validation": 0.0,
            "total": 0.0,
        },
        "environment": {},
    }
    (group / "run.json").write_text(json.dumps(metadata), encoding="utf-8")
    return group, manifests["train"]


def test_cli_validate_uses_recorded_noncanonical_repo_config(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_cli_fixture(repo)
    custom_config = repo / "configs" / "custom" / "fixture.yaml"
    custom_config.parent.mkdir()
    custom_config.write_text(
        (repo / "configs" / "editions" / "fixture.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    _write_published_group(workspace, repo, edition=Path("configs/custom/fixture.yaml"))

    result = CliRunner().invoke(
        app, ["validate", "--workspace", str(workspace), "--repo-root", str(repo)]
    )

    assert result.exit_code == 0, result.output
    assert "fixture/curated: 1 records, 1 audio files" in result.output


def test_cli_validate_external_config_requires_matching_override(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_cli_fixture(repo)
    external_config = tmp_path / "external" / "fixture-external.yaml"
    external_config.parent.mkdir()
    external_config.write_text(
        (repo / "configs" / "editions" / "fixture.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    _write_published_group(workspace, repo, edition=external_config)

    missing = CliRunner().invoke(
        app, ["validate", "--workspace", str(workspace), "--repo-root", str(repo)]
    )
    assert missing.exit_code == 1
    assert "requires --edition-config" in missing.output

    valid = CliRunner().invoke(
        app,
        [
            "validate",
            "--workspace",
            str(workspace),
            "--repo-root",
            str(repo),
            "--edition-config",
            str(external_config),
        ],
    )
    assert valid.exit_code == 0, valid.output
    assert "fixture/curated: 1 records, 1 audio files" in valid.output

    wrong_root = tmp_path / "wrong"
    _write_cli_fixture(wrong_root)
    wrong_config = wrong_root / external_config.name
    wrong_config.write_text(
        (wrong_root / "configs" / "editions" / "fixture.yaml")
        .read_text(encoding="utf-8")
        .replace("curations: []", "curations: []\naudio:\n  sample_rate_hz: 16000"),
        encoding="utf-8",
    )
    wrong = CliRunner().invoke(
        app,
        [
            "validate",
            "--workspace",
            str(workspace),
            "--repo-root",
            str(repo),
            "--edition-config",
            str(wrong_config),
        ],
    )
    assert wrong.exit_code == 1
    assert "config fingerprint does not match resolved edition" in wrong.output


def test_cli_validate_verifies_complete_published_run_metadata(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_cli_fixture(repo)
    workspace = tmp_path / "workspace"
    _write_published_group(workspace, repo)

    result = CliRunner().invoke(
        app, ["validate", "--workspace", str(workspace), "--repo-root", str(repo)]
    )

    assert result.exit_code == 0, result.output
    assert "fixture/curated: 1 records, 1 audio files" in result.output


@pytest.mark.parametrize("corruption", ["missing-run", "wrong-digest", "extra-manifest"])
def test_cli_validate_rejects_incomplete_or_mismatched_publication(
    tmp_path: Path, corruption: str
) -> None:
    repo = tmp_path / "repo"
    _write_cli_fixture(repo)
    workspace = tmp_path / "workspace"
    group, _manifest = _write_published_group(workspace, repo)
    if corruption == "missing-run":
        (group / "run.json").unlink()
    elif corruption == "wrong-digest":
        metadata_path = group / "run.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["manifests"]["train"]["sha256"] = "0" * 64
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    else:
        (group / "unexpected.jsonl").write_text("", encoding="utf-8")

    result = CliRunner().invoke(
        app, ["validate", "--workspace", str(workspace), "--repo-root", str(repo)]
    )

    assert result.exit_code == 1
    if corruption == "missing-run":
        assert "run.json is missing" in result.output
    elif corruption == "wrong-digest":
        assert "digest does not match run.json" in result.output
    else:
        assert "manifest is not declared by run.json" in result.output


@pytest.mark.parametrize(
    ("field", "value", "selection_policy"),
    [
        ("edition", "other-edition", "curated"),
        ("selection", "uncurated", "all-eligible"),
    ],
)
def test_cli_validate_rejects_run_identity_mismatched_to_publication_location(
    tmp_path: Path,
    field: str,
    value: str,
    selection_policy: str,
) -> None:
    repo = tmp_path / "repo"
    _write_cli_fixture(repo)
    workspace = tmp_path / "workspace"
    group, _manifest = _write_published_group(workspace, repo)
    metadata_path = group / "run.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] = value
    metadata["selection_policy"] = selection_policy
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    result = CliRunner().invoke(
        app, ["validate", "--workspace", str(workspace), "--repo-root", str(repo)]
    )

    assert result.exit_code == 1
    assert f"{field} {value!r} does not match publication directory" in result.output
