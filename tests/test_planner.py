from __future__ import annotations

from pathlib import Path
from threading import Barrier, Lock
from types import SimpleNamespace

import pytest

from lrac_data.config import load_edition_config
from lrac_data.planner import _remote_artifacts, build_plan


def _write_fixture_repository(root: Path) -> None:
    editions = root / "configs" / "editions"
    datasets = root / "configs" / "datasets"
    editions.mkdir(parents=True)
    datasets.mkdir(parents=True)
    (datasets / "fixture.yaml").write_text(
        """\
id: fixture
adapter: fma
release: fixture-v1
license: fixture-only
media_kinds: [noise]
sources:
  - name: archive
    url: "https://example.invalid/{part}.zip"
    filename: fixture.zip
    options:
      remote_check_url: https://example.invalid/fixture.zip
expected_files: []
""",
        encoding="utf-8",
    )
    (editions / "fixture.yaml").write_text(
        """\
schema_version: 1
edition: fixture
audio:
  sample_rate_hz: 24000
  channels: 1
  sample_format: pcm_s16le
  container: wav
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


def _metadata_state(repo: Path) -> dict[str, tuple[int, int]]:
    return {
        path.relative_to(repo).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for root in (repo / "configs", repo / "metadata")
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.mark.parametrize(
    ("selection", "policy", "selection_stage"),
    [
        ("curated", "curated", "apply quality curation"),
        ("uncurated", "all-eligible", "select all eligible inventory"),
    ],
)
def test_plan_is_metadata_only_and_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selection: str,
    policy: str,
    selection_stage: str,
) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repository(repo)
    before = _snapshot(repo)
    would_be_workspace = tmp_path / "workspace"

    def fail_download(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"plan attempted a download: {args!r}, {kwargs!r}")

    monkeypatch.setattr("lrac_data.datasets.io.download_file", fail_download)

    report = build_plan("fixture", selection=selection, repo_root=repo)

    assert report.selection == selection
    assert report.policy == policy
    assert report.datasets[0].id == "fixture"
    assert selection_stage in report.stages
    assert report.remote_checks == ()
    assert _snapshot(repo) == before
    assert not would_be_workspace.exists()


def test_remote_plan_uses_head_requests_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repository(repo)
    calls: list[tuple[str, str]] = []

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs == {"follow_redirects": True, "timeout": 15.0}

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def request(self, method: str, url: str) -> SimpleNamespace:
            calls.append((method, url))
            return SimpleNamespace(is_success=True, status_code=204)

    monkeypatch.setattr("lrac_data.planner.httpx.Client", FakeClient)

    report = build_plan("fixture", repo_root=repo, check_remote=True)

    assert calls == [("HEAD", "https://example.invalid/fixture.zip")]
    assert report.remote_checks[0].ok
    assert report.remote_checks[0].status == 204


def test_remote_plan_checks_sources_concurrently_in_deterministic_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _write_fixture_repository(repo)
    barrier = Barrier(4)
    lock = Lock()
    calls: list[str] = []
    urls = [f"https://example.invalid/{index}.zip" for index in range(4)]

    class FakeClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def request(self, method: str, url: str) -> SimpleNamespace:
            assert method == "HEAD"
            barrier.wait(timeout=2)
            with lock:
                calls.append(url)
            return SimpleNamespace(is_success=True, status_code=200)

    monkeypatch.setattr("lrac_data.planner.httpx.Client", FakeClient)
    monkeypatch.setattr(
        "lrac_data.planner._remote_artifacts",
        lambda _dataset, _source: [(str(index), url) for index, url in enumerate(urls)],
    )

    report = build_plan("fixture", repo_root=repo, check_remote=True)

    assert set(calls) == set(urls)
    assert [check.url for check in report.remote_checks] == urls


def test_real_2026_plan_resolves_both_selections_without_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = Path(__file__).resolve().parents[1]
    before = _metadata_state(repo)
    monkeypatch.chdir(tmp_path)

    reports = {
        selection: build_plan("2026", selection=selection, repo_root=repo)
        for selection in ("curated", "uncurated")
    }

    expected_datasets = (
        "dns5",
        "ears",
        "fma",
        "fsd50k",
        "globe",
        "libritts",
        "mls",
        "motus",
        "vctk",
        "wham",
    )
    for selection, report in reports.items():
        assert report.edition == "2026"
        assert report.selection == selection
        assert tuple(dataset.id for dataset in report.datasets) == expected_datasets
        curation_targets = sum(dataset.curation_targets for dataset in report.datasets)
        assert curation_targets > 400_000 if selection == "curated" else curation_targets == 0
        assert sum(dataset.exclusion_targets for dataset in report.datasets) > 2_500
        assert report.public_evaluation is not None
        assert report.remote_checks == ()
        for executable in ("ffmpeg", "git", "zip"):
            issue = f"required executable is not installed: {executable}"
            assert (issue in report.unresolved) is (not report.requirements[executable])

    assert _metadata_state(repo) == before
    assert list(tmp_path.iterdir()) == []


def test_real_remote_templates_expand_every_artifact() -> None:
    repo = Path(__file__).resolve().parents[1]
    config = load_edition_config("2026", repo_root=repo).config
    expected = {"dns5": 21, "ears": 107, "globe": 108, "mls": 1395}

    for adapter, count in expected.items():
        dataset = next(item for item in config.datasets if item.adapter == adapter)
        templated = next(source for source in dataset.sources if "{" in (source.url or ""))

        assert len(_remote_artifacts(dataset, templated)) == count
