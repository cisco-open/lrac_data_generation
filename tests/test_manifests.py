from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import pytest

from lrac_data.manifests import (
    ManifestError,
    read_manifest,
    write_manifest,
    write_ordered_manifest,
)
from lrac_data.models import ManifestItem, MediaKind, Split, qualify_id


def _manifest_item(source_id: str) -> ManifestItem:
    return ManifestItem(
        id=qualify_id("fixture", source_id),
        dataset="fixture",
        media_kind=MediaKind.SPEECH,
        audio_path=PurePosixPath(f"prepared/fixture/{source_id}.wav"),
        source_release="fixture-v1",
        source_id=source_id,
        split=Split.TRAIN,
        sample_rate_hz=24_000,
        channels=1,
        frame_count=240,
        checksum=source_id * 16,
        speaker_id="speaker",
        text=f"text for {source_id}",
    )


def test_manifest_is_byte_deterministic_and_sorted_by_stable_id(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    a = _manifest_item("a")
    b = _manifest_item("b")

    write_manifest(first, [b, a])
    write_manifest(second, [a, b])

    assert first.read_bytes() == second.read_bytes()
    lines = first.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["fixture:a", "fixture:b"]
    assert lines[0] == json.dumps(
        a.model_dump(mode="json", exclude_none=True),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert read_manifest(first) == (a, b)


def test_ordered_manifest_streams_atomically_and_rejects_disorder(tmp_path: Path) -> None:
    destination = tmp_path / "manifest.jsonl"
    destination.write_text("original\n", encoding="utf-8")
    consumed: list[str] = []

    def records():
        for source_id in ("a", "c", "b"):
            consumed.append(source_id)
            yield _manifest_item(source_id)

    with pytest.raises(ManifestError, match="not strictly ordered"):
        write_ordered_manifest(destination, records())

    assert consumed == ["a", "c", "b"]
    assert destination.read_text(encoding="utf-8") == "original\n"


def test_manifest_rejects_duplicate_ids_before_publication(tmp_path: Path) -> None:
    destination = tmp_path / "manifest.jsonl"
    item = _manifest_item("duplicate")

    with pytest.raises(ManifestError, match="duplicate manifest ID"):
        write_manifest(destination, [item, item])

    assert not destination.exists()
