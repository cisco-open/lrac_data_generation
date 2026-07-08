from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EDITION = ROOT / "metadata" / "editions" / "2026"


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_normalized_exclusions_match_migrated_policy_sources() -> None:
    exclusions = _csv_rows(EDITION / "exclusions.csv")
    assert all(
        row["source_id"] and not row["speaker_id"]
        for row in exclusions
        if row["partition"] == "evaluation"
    )

    speech_expected: set[tuple[str, str]] = set()
    evaluation_only_speech: set[tuple[str, str]] = set()
    noise_expected: set[tuple[str, str]] = set()
    evaluation_only_speech_datasets = {"globe", "libritts"}
    dataset_aliases = {
        "dns5_fullband": "dns5",
        "wham_noise_48k": "wham",
    }
    for track in ("track_1", "track_2"):
        root = EDITION / "evaluation" / "open_testset" / track
        for row in _csv_rows(root / "speech_meta.csv"):
            dataset = row["speech_dataset"]
            source_id = row["speech_uid"]
            target = (dataset, source_id)
            if dataset in evaluation_only_speech_datasets:
                evaluation_only_speech.add(target)
                continue
            if dataset == "vctk" and not source_id.startswith(("p232_", "p257_")):
                continue
            speech_expected.add(target)
        for row in _csv_rows(root / "noise_meta.csv"):
            source_dataset = row["noise_dataset"]
            if source_dataset == "freesound":
                continue
            dataset = dataset_aliases.get(source_dataset, source_dataset)
            noise_expected.add((dataset, row["noise_uid"]))

    speech_actual = {
        (row["dataset"], row["source_id"])
        for row in exclusions
        if row["name"] == "open-evaluation-speech"
    }
    noise_actual = {
        (row["dataset"], row["source_id"])
        for row in exclusions
        if row["name"] == "open-evaluation-noise"
    }
    assert speech_actual == speech_expected
    assert {dataset for dataset, _source_id in evaluation_only_speech} == (
        evaluation_only_speech_datasets
    )
    assert speech_actual.isdisjoint(evaluation_only_speech)
    assert noise_actual == noise_expected

    motus_expected = {
        f"motus_{Path(row['filename']).stem}"
        for row in _csv_rows(EDITION / "evaluation" / "motus_rir_source_ids.csv")
    }
    motus_actual = {
        row["source_id"]
        for row in exclusions
        if row["name"] == "evaluation-rir" and row["dataset"] == "motus"
    }
    assert motus_actual == motus_expected


def test_normalized_validation_noise_and_rir_match_frozen_lists() -> None:
    exclusions = _csv_rows(EDITION / "exclusions.csv")
    dns5_rir_prefix = "SLR28/RIRS_NOISES/real_rirs_isotropic_noises"

    def validation_dataset(source_id: str, *, rir: bool = False) -> str:
        if rir:
            return "motus" if source_id.startswith("motus_") else "dns5"
        if source_id.startswith("fsd50k_"):
            return "fsd50k"
        if source_id.startswith("fma_"):
            return "fma"
        if source_id.startswith("file"):
            return "wham"
        return "dns5"

    for kind in ("noise", "rir"):
        source_ids = {
            line.split()[0]
            for line in (EDITION / "validation" / f"{kind}_source_ids.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        }
        expected = {
            (
                validation_dataset(source_id, rir=kind == "rir"),
                (
                    f"{dns5_rir_prefix}/{source_id}"
                    if kind == "rir" and not source_id.startswith("motus_")
                    else source_id
                ),
            )
            for source_id in source_ids
        }
        actual = {
            (row["dataset"], row["source_id"])
            for row in exclusions
            if row["name"] == f"validation-{kind}"
        }
        assert actual == expected
