from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lrac_data.cli import app
from lrac_data.state import sha256_file


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


def _write_published_group(workspace: Path) -> tuple[Path, Path]:
    group = workspace / "manifests" / "2026" / "curated"
    manifest = group / "train.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("", encoding="utf-8")
    metadata = {
        "schema_version": 1,
        "run_id": "fixture-run",
        "edition": "2026",
        "selection": "curated",
        "manifests": {
            "train": {
                "path": manifest.relative_to(workspace).as_posix(),
                "sha256": sha256_file(manifest),
            }
        },
    }
    (group / "run.json").write_text(json.dumps(metadata), encoding="utf-8")
    return group, manifest


def test_cli_validate_verifies_complete_published_run_metadata(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _write_published_group(workspace)

    result = CliRunner().invoke(app, ["validate", "--workspace", str(workspace)])

    assert result.exit_code == 0, result.output
    assert "2026/curated: 0 records, 0 audio files" in result.output


@pytest.mark.parametrize("corruption", ["missing-run", "wrong-digest", "extra-manifest"])
def test_cli_validate_rejects_incomplete_or_mismatched_publication(
    tmp_path: Path, corruption: str
) -> None:
    workspace = tmp_path / "workspace"
    group, _manifest = _write_published_group(workspace)
    if corruption == "missing-run":
        (group / "run.json").unlink()
    elif corruption == "wrong-digest":
        metadata_path = group / "run.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["manifests"]["train"]["sha256"] = "0" * 64
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    else:
        (group / "validation.jsonl").write_text("", encoding="utf-8")

    result = CliRunner().invoke(app, ["validate", "--workspace", str(workspace)])

    assert result.exit_code == 1
    if corruption == "missing-run":
        assert "run.json is missing" in result.output
    elif corruption == "wrong-digest":
        assert "digest does not match run.json" in result.output
    else:
        assert "manifest is not declared by run.json" in result.output
