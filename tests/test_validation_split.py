from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from lrac_data.config import load_edition_config
from lrac_data.models import ExclusionPartition, InventoryItem, MediaKind, qualify_id

_TOOL_PATH = Path(__file__).resolve().parents[1] / "tools/freeze_validation_speech.py"
_SPEC = importlib.util.spec_from_file_location("freeze_validation_speech", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_TOOL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TOOL
_SPEC.loader.exec_module(_TOOL)

Stratum = _TOOL.Stratum
freeze_validation_split = _TOOL.freeze_validation_split
render_csv = _TOOL.render_csv


def _speech(
    source_id: str, speaker_id: str, *, source_checksum: str | None = None
) -> InventoryItem:
    return InventoryItem(
        id=qualify_id("fixture", source_id),
        dataset="fixture",
        source_id=source_id,
        source_release="fixture-v1",
        media_kind=MediaKind.SPEECH,
        source_path=Path("/fixtures") / f"{source_id}.wav",
        source_checksum=source_checksum or hashlib.sha256(source_id.encode()).hexdigest(),
        speaker_id=speaker_id,
        language="en",
    )


def test_validation_freezer_is_deterministic_and_excludes_evaluation_speakers() -> None:
    items = [
        _speech("a-1", "a"),
        _speech("a-2", "a"),
        _speech("a-3", "a"),
        _speech("b-1", "b"),
        _speech("b-2", "b"),
        _speech("evaluation-1", "evaluation"),
        _speech("evaluation-2", "evaluation"),
    ]
    strata = (
        Stratum(
            name="fixture",
            dataset="fixture",
            target=3,
            minimum_speakers=2,
            contribution_cap=2,
        ),
    )

    first = freeze_validation_split(
        items,
        evaluation_speakers=frozenset({("fixture", "evaluation")}),
        seed=2026,
        strata=strata,
    )
    second = freeze_validation_split(
        reversed(items),
        evaluation_speakers=frozenset({("fixture", "evaluation")}),
        seed=2026,
        strata=strata,
    )

    assert [item.id for item in first.validation_items] == [
        item.id for item in second.validation_items
    ]
    assert len(first.validation_items) == 3
    assert {item.speaker_id for item in first.validation_items} == {"a", "b"}
    assert first.withheld_speakers == (("fixture", "a"), ("fixture", "b"))
    assert "evaluation" not in render_csv(first)


def test_validation_freezer_canonicalizes_duplicate_audio() -> None:
    checksum = hashlib.sha256(b"same audio").hexdigest()
    items = [
        _speech("alias-b", "a", source_checksum=checksum),
        _speech("alias-a", "a", source_checksum=checksum),
        _speech("other", "b"),
    ]
    strata = (
        Stratum(
            name="fixture",
            dataset="fixture",
            target=2,
            minimum_speakers=2,
            contribution_cap=1,
        ),
    )

    first = freeze_validation_split(items, seed=2026, strata=strata)
    second = freeze_validation_split(reversed(items), seed=2026, strata=strata)

    assert [item.id for item in first.validation_items] == [
        item.id for item in second.validation_items
    ]
    assert "fixture:alias-a" in {item.id for item in first.validation_items}
    assert "fixture:alias-b" not in {item.id for item in first.validation_items}
    assert len({item.source_checksum for item in first.validation_items}) == 2


def test_validation_freezer_requires_source_checksums() -> None:
    item = _speech("missing", "speaker").model_copy(update={"source_checksum": None})
    strata = (
        Stratum(
            name="fixture",
            dataset="fixture",
            target=1,
            minimum_speakers=1,
            contribution_cap=1,
        ),
    )

    with pytest.raises(ValueError, match="has no source checksum"):
        freeze_validation_split([item], strata=strata)


def test_validation_freezer_applies_speaker_limit_before_deduplication() -> None:
    checksum = hashlib.sha256(b"repeated audio").hexdigest()
    aliases = [
        _speech(f"alias-{index:03d}", "too-many", source_checksum=checksum) for index in range(251)
    ]
    strata = (
        Stratum(
            name="fixture",
            dataset="fixture",
            target=1,
            minimum_speakers=1,
            contribution_cap=1,
        ),
    )

    with pytest.raises(ValueError, match="insufficient eligible capacity"):
        freeze_validation_split(aliases, strata=strata)


def test_2026_speech_split_metadata_has_frozen_targets() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    loaded = load_edition_config("2026", repo_root=repo_root, selection="uncurated")

    validation = [
        exclusion
        for exclusion in loaded.config.exclusions
        if exclusion.name == "validation-speech-items"
    ]
    withheld = [
        exclusion
        for exclusion in loaded.config.exclusions
        if exclusion.name == "validation-speech-speakers"
    ]

    assert all(rule.partition is ExclusionPartition.VALIDATION for rule in validation)
    assert all(rule.partition is ExclusionPartition.WITHHELD for rule in withheld)
    assert Counter({rule.dataset: len(rule.source_ids) for rule in validation}) == Counter(
        dns5=250,
        ears=250,
        globe=250,
        libritts=250,
        mls=300,
        vctk=250,
    )
    assert Counter({rule.dataset: len(rule.speaker_ids) for rule in withheld}) == Counter(
        dns5=23,
        ears=10,
        globe=49,
        libritts=16,
        mls=17,
        vctk=10,
    )

    digest = hashlib.sha256()
    source_ids = sorted(
        (rule.dataset or "", source_id) for rule in validation for source_id in rule.source_ids
    )
    for dataset, source_id in source_ids:
        digest.update(f"{dataset}\t{source_id}\n".encode())
    expected_checksum = "824215a59c6a0e23e73caa7393ce3d12cd0533ff79875c4613c6842b8e9fd49a"
    assert digest.hexdigest() == expected_checksum

    report_path = repo_root / "metadata/editions/2026/validation/speech_split.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["edition"] == "2026"
    assert report["policy"]["seed"] == 2026
    assert report["result"]["validation_source_ids"] == 1_550
    assert report["result"]["withheld_speakers"] == 125
    assert report["result"]["validation_source_id_sha256"] == expected_checksum
    assert report["result"]["unique_validation_source_checksums"] == 1_550
    assert (
        report["result"]["validation_audio_identity_sha256"]
        == "ae72456c625cf04caa6b9cd711c41eab85eac1d9661266ea13d6c1810edb7a0c"
    )

    minimum_speakers = {
        "dns5": 10,
        "ears/f": 5,
        "ears/m": 5,
        "globe/f": 5,
        "globe/m": 5,
        "libritts/f": 5,
        "libritts/m": 5,
        "vctk/f": 5,
        "vctk/m": 5,
        "mls/french": 4,
        "mls/german": 4,
        "mls/spanish": 4,
    }
    for allocation in report["result"]["allocations"]:
        assert len(allocation["selected_speakers"]) >= minimum_speakers[allocation["stratum"]]
        assert max(allocation["validation_items_per_speaker"].values()) <= 25
