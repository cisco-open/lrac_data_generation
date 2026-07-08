from __future__ import annotations

from pathlib import Path

import pytest

from lrac_data.datasets.inventory import FileTreeRule, build_file_inventory
from lrac_data.models import DatasetConfig, InventoryItem, MediaKind, qualify_id


class _Owner:
    def __init__(self, extracted_dir: Path) -> None:
        self.config = DatasetConfig(
            id="fixture",
            adapter="fixture",
            release="fixture-v1",
            license="fixture-only",
            media_kinds=(MediaKind.NOISE, MediaKind.RIR),
        )
        self.extracted_dir = extracted_dir
        self.expected_calls = 0
        self.item_calls = 0
        self.expected_error: Exception | None = None

    def ensure_expected_files(self) -> None:
        self.expected_calls += 1
        if self.expected_error is not None:
            raise self.expected_error

    def item(
        self,
        source_id: str,
        media_kind: MediaKind,
        source_path: Path,
        **_kwargs: object,
    ) -> InventoryItem:
        self.item_calls += 1
        return InventoryItem(
            id=qualify_id(self.config.id, source_id),
            dataset=self.config.id,
            source_id=source_id,
            source_release=self.config.release,
            media_kind=media_kind,
            source_path=source_path.resolve(),
        )


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")
    return path


def test_file_inventory_orders_matches_by_relative_path(tmp_path: Path) -> None:
    owner = _Owner(tmp_path)
    _touch(tmp_path / "clips" / "nested" / "b.wav")
    _touch(tmp_path / "clips" / "a.wav")

    records = build_file_inventory(
        owner,
        (FileTreeRule(Path("clips"), "**/*.wav", MediaKind.NOISE),),
    )

    assert [record.source_id for record in records] == ["a", "b"]
    assert owner.expected_calls == 1


def test_file_inventory_applies_prefixes_and_media_kinds(tmp_path: Path) -> None:
    owner = _Owner(tmp_path)
    _touch(tmp_path / "noise" / "clip.mp3")
    _touch(tmp_path / "rirs" / "room.wav")

    records = build_file_inventory(
        owner,
        (
            FileTreeRule(Path("rirs"), "*.wav", MediaKind.RIR, "motus_"),
            FileTreeRule(Path("noise"), "*.mp3", MediaKind.NOISE, "fma_"),
        ),
    )

    assert [(record.source_id, record.media_kind) for record in records] == [
        ("fma_clip", MediaKind.NOISE),
        ("motus_room", MediaKind.RIR),
    ]


def test_file_inventory_rejects_duplicate_ids_across_rules(tmp_path: Path) -> None:
    owner = _Owner(tmp_path)
    first = _touch(tmp_path / "first" / "same.wav").resolve()
    second = _touch(tmp_path / "second" / "same.wav").resolve()

    with pytest.raises(ValueError, match="Duplicate source ID 'same'") as error:
        build_file_inventory(
            owner,
            (
                FileTreeRule(Path("first"), "*.wav", MediaKind.NOISE),
                FileTreeRule(Path("second"), "*.wav", MediaKind.RIR),
            ),
        )

    assert str(first) in str(error.value)
    assert str(second) in str(error.value)


def test_file_inventory_ignores_matching_directories(tmp_path: Path) -> None:
    owner = _Owner(tmp_path)
    (tmp_path / "clips" / "directory.wav").mkdir(parents=True)
    _touch(tmp_path / "clips" / "audio.wav")

    records = build_file_inventory(
        owner,
        (FileTreeRule(Path("clips"), "*.wav", MediaKind.NOISE),),
    )

    assert [record.source_id for record in records] == ["audio"]


def test_file_inventory_delegates_expected_file_validation(tmp_path: Path) -> None:
    owner = _Owner(tmp_path)
    owner.expected_error = FileNotFoundError("incomplete fixture")

    with pytest.raises(FileNotFoundError, match="incomplete fixture"):
        build_file_inventory(
            owner,
            (FileTreeRule(Path("clips"), "*.wav", MediaKind.NOISE),),
        )

    assert owner.expected_calls == 1
    assert owner.item_calls == 0


@pytest.mark.parametrize(
    ("relative_root", "pattern", "message"),
    [
        (Path("../outside"), "*.wav", "safe relative path"),
        (Path("clips"), "../*.wav", "safe relative pattern"),
        (Path("clips"), "", "non-empty"),
    ],
)
def test_file_inventory_rejects_unsafe_or_invalid_rules(
    tmp_path: Path,
    relative_root: Path,
    pattern: str,
    message: str,
) -> None:
    owner = _Owner(tmp_path)

    with pytest.raises(ValueError, match=message):
        build_file_inventory(
            owner,
            (FileTreeRule(relative_root, pattern, MediaKind.NOISE),),
        )

    assert owner.expected_calls == 0


def test_file_inventory_requires_at_least_one_rule(tmp_path: Path) -> None:
    owner = _Owner(tmp_path)

    with pytest.raises(ValueError, match="at least one rule"):
        build_file_inventory(owner, ())

    assert owner.expected_calls == 0
