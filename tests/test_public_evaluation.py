from pathlib import Path

import pytest

from lrac_data.models import PublicEvaluationSpec
from lrac_data.public_evaluation import fetch_public_evaluation, inventory_public_evaluation


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
