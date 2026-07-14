import subprocess
from pathlib import Path

import pytest

from lrac_data.models import PublicEvaluationSpec
from lrac_data.public_evaluation import fetch_public_evaluation, inventory_public_evaluation


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _evaluation_repository(tmp_path: Path, name: str = "upstream") -> tuple[Path, str, str]:
    repository = tmp_path / name
    repository.mkdir()
    _git(repository, "init", "--initial-branch=main")
    _git(repository, "config", "user.name", "LRAC tests")
    _git(repository, "config", "user.email", "lrac@example.invalid")
    audio = repository / "open-test-set" / "track_1" / "clean" / "utterance.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"revision one")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "first revision")
    first_revision = _git(repository, "rev-parse", "HEAD")

    audio.write_bytes(b"revision two")
    _git(repository, "commit", "-am", "second revision")
    second_revision = _git(repository, "rev-parse", "HEAD")
    return repository, first_revision, second_revision


def _evaluation_spec(repository: Path, revision: str) -> PublicEvaluationSpec:
    return PublicEvaluationSpec(
        repository_url=str(repository),
        revision=revision,
        subdirectory="open-test-set",
        tracks=("track_1",),
    )


def test_public_evaluation_rejects_download_root_symlink_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (workspace / "downloads").symlink_to(outside, target_is_directory=True)
    spec = PublicEvaluationSpec(
        repository_url="https://example.invalid/evaluation.git",
        revision="a" * 40,
        subdirectory="open-test-set",
    )

    with pytest.raises(ValueError, match="downloads"):
        fetch_public_evaluation(spec, workspace)


def test_public_evaluation_rejects_wav_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "open-test-set" / "track_1" / "clean"
    root.mkdir(parents=True)
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"keep")
    (root / "utterance.wav").symlink_to(outside)
    spec = PublicEvaluationSpec(
        repository_url="https://example.invalid/evaluation.git",
        revision="a" * 40,
        subdirectory="open-test-set",
        tracks=("track_1",),
    )

    with pytest.raises(ValueError, match="WAV"):
        inventory_public_evaluation(spec, tmp_path / "open-test-set")


def test_public_evaluation_rejects_condition_symlink_escape(tmp_path: Path) -> None:
    evaluation_root = tmp_path / "open-test-set"
    track_root = evaluation_root / "track_1"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "utterance.wav").touch()
    track_root.mkdir(parents=True)
    (track_root / "clean").symlink_to(outside, target_is_directory=True)
    spec = PublicEvaluationSpec(
        repository_url="https://example.invalid/evaluation.git",
        revision="a" * 40,
        subdirectory="open-test-set",
        tracks=("track_1",),
    )

    with pytest.raises(ValueError, match="condition"):
        inventory_public_evaluation(spec, evaluation_root)


def test_public_evaluation_inventory_pairs_every_condition(tmp_path: Path) -> None:
    root = tmp_path / "open-test-set" / "track_1"
    for directory in ("clean", "noisy", "reverb", "reference_noisy", "reference_reverb"):
        target = root / directory / "utterance.wav"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    spec = PublicEvaluationSpec(
        repository_url="https://example.invalid/evaluation.git",
        revision="a" * 40,
        subdirectory="open-test-set",
        tracks=("track_1",),
    )

    records = inventory_public_evaluation(spec, tmp_path / "open-test-set")

    assert len(records) == 6
    assert {record.metadata["condition"] for record in records} == {
        "clean",
        "noisy",
        "reverb",
    }
    assert {record.metadata["role"] for record in records} == {"input", "reference"}
    assert len({record.id for record in records}) == len(records)


def test_fetch_public_evaluation_resets_and_cleans_reused_checkout(tmp_path: Path) -> None:
    repository, first_revision, _ = _evaluation_repository(tmp_path)
    workspace = tmp_path / "workspace"
    spec = _evaluation_spec(repository, first_revision)
    root = fetch_public_evaluation(spec, workspace)
    checkout = root.parent
    tracked = root / "track_1" / "clean" / "utterance.wav"

    tracked.write_bytes(b"locally modified")
    untracked = root / "track_1" / "clean" / "injected.wav"
    untracked.write_bytes(b"not from upstream")
    ignored = root / "track_1" / "clean" / "ignored.wav"
    ignored.write_bytes(b"also not from upstream")
    (checkout / ".git" / "info" / "exclude").write_text("ignored.wav\n", encoding="utf-8")

    fetched_root = fetch_public_evaluation(spec, workspace)

    assert fetched_root == root
    assert tracked.read_bytes() == b"revision one"
    assert not untracked.exists()
    assert not ignored.exists()
    assert _git(checkout, "rev-parse", "HEAD") == first_revision
    assert _git(checkout, "status", "--porcelain=v1", "--untracked-files=all") == ""


def test_fetch_public_evaluation_moves_between_pinned_revisions(tmp_path: Path) -> None:
    repository, first_revision, second_revision = _evaluation_repository(tmp_path)
    workspace = tmp_path / "workspace"
    first = _evaluation_spec(repository, first_revision)
    second = _evaluation_spec(repository, second_revision)

    root = fetch_public_evaluation(first, workspace)
    assert (root / "track_1" / "clean" / "utterance.wav").read_bytes() == b"revision one"

    root = fetch_public_evaluation(second, workspace)
    checkout = root.parent
    assert (root / "track_1" / "clean" / "utterance.wav").read_bytes() == b"revision two"
    assert _git(checkout, "rev-parse", "HEAD") == second_revision
    assert (
        subprocess.run(["git", "symbolic-ref", "-q", "HEAD"], cwd=checkout, check=False).returncode
        == 1
    )


def test_fetch_public_evaluation_rejects_wrong_origin_before_cleaning(tmp_path: Path) -> None:
    repository, first_revision, _ = _evaluation_repository(tmp_path, "first-upstream")
    other_repository, _, other_revision = _evaluation_repository(tmp_path, "other-upstream")
    workspace = tmp_path / "workspace"
    root = fetch_public_evaluation(_evaluation_spec(repository, first_revision), workspace)
    sentinel = root / "do-not-delete.wav"
    sentinel.write_bytes(b"keep when ownership check fails")

    with pytest.raises(RuntimeError, match="origin does not match"):
        fetch_public_evaluation(_evaluation_spec(other_repository, other_revision), workspace)

    assert sentinel.exists()


def test_fetch_public_evaluation_preserves_dirty_checkout_when_fetch_fails(
    tmp_path: Path,
) -> None:
    repository, first_revision, _ = _evaluation_repository(tmp_path)
    workspace = tmp_path / "workspace"
    root = fetch_public_evaluation(_evaluation_spec(repository, first_revision), workspace)
    tracked = root / "track_1" / "clean" / "utterance.wav"
    tracked.write_bytes(b"keep when fetch fails")

    with pytest.raises(RuntimeError, match=r"git fetch .* failed"):
        fetch_public_evaluation(_evaluation_spec(repository, "f" * 40), workspace)

    assert tracked.read_bytes() == b"keep when fetch fails"


def test_fetch_public_evaluation_rejects_external_git_directory(tmp_path: Path) -> None:
    repository, first_revision, _ = _evaluation_repository(tmp_path)
    workspace = tmp_path / "workspace"
    checkout = workspace / "downloads" / "lrac-open-evaluation" / "repository"
    checkout.mkdir(parents=True)
    (checkout / ".git").symlink_to(repository / ".git", target_is_directory=True)

    with pytest.raises(RuntimeError, match="checkout is incomplete"):
        fetch_public_evaluation(_evaluation_spec(repository, first_revision), workspace)
