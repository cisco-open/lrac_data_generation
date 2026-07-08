from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from lrac_data.models import (
    DatasetConfig,
    EditionConfig,
    InventoryItem,
    MediaKind,
    PathSegment,
    PublicEvaluationSpec,
    SourceSpec,
)

UNSAFE_SEGMENTS = (".", "..", "../victim", "/tmp/victim", "nested/value", r"..\victim")


@pytest.mark.parametrize("value", UNSAFE_SEGMENTS)
def test_path_segment_rejects_traversal_and_hierarchical_values(value: str) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(PathSegment).validate_python(value)


def test_path_segment_preserves_required_filename_templates() -> None:
    source = SourceSpec(
        name="shards",
        url="https://example.invalid/{index}",
        filename="{index:04d}.parquet",
    )

    assert source.filename == "{index:04d}.parquet"


@pytest.mark.parametrize("value", UNSAFE_SEGMENTS)
def test_path_bearing_configuration_fields_use_safe_segments(value: str) -> None:
    dataset = DatasetConfig(
        id="safe",
        adapter="safe",
        release="fixture",
        license="fixture",
        media_kinds=(MediaKind.NOISE,),
    )
    with pytest.raises(ValidationError):
        DatasetConfig(
            id=value,
            adapter="safe",
            release="fixture",
            license="fixture",
            media_kinds=(MediaKind.NOISE,),
        )
    with pytest.raises(ValidationError):
        DatasetConfig(
            id="safe",
            adapter=value,
            release="fixture",
            license="fixture",
            media_kinds=(MediaKind.NOISE,),
        )
    with pytest.raises(ValidationError):
        SourceSpec(
            name="archive",
            url="https://example.invalid/archive",
            filename=value,
        )
    with pytest.raises(ValidationError):
        EditionConfig(edition=value, datasets=(dataset,))
    with pytest.raises(ValidationError):
        PublicEvaluationSpec(
            id=value,
            repository_url="https://example.invalid/evaluation.git",
            revision="a" * 40,
            subdirectory="evaluation",
        )
    with pytest.raises(ValidationError):
        PublicEvaluationSpec(
            repository_url="https://example.invalid/evaluation.git",
            revision="a" * 40,
            subdirectory="evaluation",
            tracks=(value,),
        )
    with pytest.raises(ValidationError):
        PublicEvaluationSpec(
            repository_url="https://example.invalid/evaluation.git",
            revision="a" * 40,
            subdirectory="evaluation",
            conditions=(value,),
        )
    with pytest.raises(ValidationError):
        InventoryItem(
            id=f"{value}:clip",
            dataset=value,
            source_id="clip",
            source_release="fixture",
            media_kind=MediaKind.NOISE,
            source_path=Path("clip.wav"),
        )


def test_source_ids_remain_upstream_identifiers_not_path_segments(tmp_path: Path) -> None:
    source_id = "speaker/chapter/clip"
    item = InventoryItem(
        id=f"safe:{source_id}",
        dataset="safe",
        source_id=source_id,
        source_release="fixture",
        media_kind=MediaKind.NOISE,
        source_path=tmp_path / "clip.wav",
    )

    assert item.source_id == source_id
