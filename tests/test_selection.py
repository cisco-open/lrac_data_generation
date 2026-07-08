from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest

from lrac_data.models import (
    CurationSpec,
    ExclusionSpec,
    InventoryItem,
    MediaKind,
    SelectionMode,
    Split,
    qualify_id,
)
from lrac_data.selection import SelectionError, select_inventory


def _item(
    source_id: str,
    *,
    dataset: str = "speech",
    speaker_id: str | None = None,
    media_kind: MediaKind = MediaKind.SPEECH,
) -> InventoryItem:
    return InventoryItem(
        id=qualify_id(dataset, source_id),
        dataset=dataset,
        source_id=source_id,
        source_release="fixture-v1",
        media_kind=media_kind,
        source_path=Path("/fixtures") / dataset / f"{source_id}.wav",
        speaker_id=speaker_id,
    )


def _ids(items: tuple[InventoryItem, ...]) -> tuple[str, ...]:
    return tuple(item.id for item in items)


def test_curated_and_uncurated_share_mandatory_exclusions() -> None:
    inventory = [
        _item("quality-keep", speaker_id="train-speaker"),
        _item("quality-drop", speaker_id="train-speaker"),
        _item("validation-a", speaker_id="validation-speaker"),
        _item("validation-b", speaker_id="validation-speaker"),
        _item("evaluation", speaker_id="evaluation-speaker"),
    ]
    exclusions = [
        ExclusionSpec(
            name="frozen-validation-speaker",
            partition=Split.VALIDATION,
            dataset="speech",
            speaker_ids=("validation-speaker",),
        ),
        ExclusionSpec(
            name="frozen-open-evaluation",
            partition=Split.EVALUATION,
            dataset="speech",
            source_ids=("evaluation",),
        ),
    ]
    curations = [
        CurationSpec(
            name="speech-quality-allowlist",
            dataset="speech",
            source_ids=("quality-keep", "validation-a"),
        )
    ]

    curated = select_inventory(
        inventory,
        selection=SelectionMode.CURATED,
        exclusions=exclusions,
        curations=curations,
    )
    uncurated = select_inventory(
        inventory,
        selection=SelectionMode.UNCURATED,
        exclusions=exclusions,
        curations=curations,
    )

    assert _ids(curated.validation) == ("speech:validation-a",)
    assert _ids(uncurated.validation) == (
        "speech:validation-a",
        "speech:validation-b",
    )
    assert _ids(curated.evaluation) == _ids(uncurated.evaluation) == ("speech:evaluation",)
    assert _ids(curated.training) == ("speech:quality-keep",)
    assert _ids(curated.quality_rejected) == (
        "speech:quality-drop",
        "speech:validation-b",
    )
    assert _ids(uncurated.training) == (
        "speech:quality-drop",
        "speech:quality-keep",
    )
    assert uncurated.quality_rejected == ()
    assert uncurated.selection.policy_name == "all-eligible"


def test_exact_and_speaker_exclusions_partition_the_complete_inventory() -> None:
    inventory = [
        _item("speaker-a-1", speaker_id="speaker-a"),
        _item("speaker-a-2", speaker_id="speaker-a"),
        _item("speaker-b-1", speaker_id="speaker-b"),
        _item("rir-1", dataset="rir", media_kind=MediaKind.RIR),
    ]
    result = select_inventory(
        inventory,
        selection="uncurated",
        exclusions=[
            ExclusionSpec(
                name="speaker-validation",
                partition="validation",
                dataset="speech",
                speaker_ids=("speaker-a",),
            ),
            ExclusionSpec(
                name="rir-evaluation",
                partition="evaluation",
                dataset="rir",
                source_ids=("rir-1",),
            ),
        ],
    )

    assert _ids(result.training) == ("speech:speaker-b-1",)
    assert _ids(result.validation) == (
        "speech:speaker-a-1",
        "speech:speaker-a-2",
    )
    assert _ids(result.evaluation) == ("rir:rir-1",)


def test_media_scoped_allowlists_do_not_reject_other_media_in_mixed_dataset() -> None:
    inventory = [
        _item("speech-keep", dataset="dns5", speaker_id="speaker"),
        _item("speech-drop", dataset="dns5", speaker_id="speaker"),
        _item("noise-keep", dataset="dns5", media_kind=MediaKind.NOISE),
        _item("noise-drop", dataset="dns5", media_kind=MediaKind.NOISE),
        _item("rir-a", dataset="dns5", media_kind=MediaKind.RIR),
        _item("rir-b", dataset="dns5", media_kind=MediaKind.RIR),
    ]

    result = select_inventory(
        inventory,
        selection="curated",
        curations=[
            CurationSpec(
                name="dns5-speech-quality",
                dataset="dns5",
                media_kind=MediaKind.SPEECH,
                source_ids=("speech-keep",),
            ),
            CurationSpec(
                name="dns5-noise-quality",
                dataset="dns5",
                media_kind=MediaKind.NOISE,
                source_ids=("noise-keep",),
            ),
        ],
    )

    assert _ids(result.training) == (
        "dns5:noise-keep",
        "dns5:rir-a",
        "dns5:rir-b",
        "dns5:speech-keep",
    )
    assert _ids(result.quality_rejected) == (
        "dns5:noise-drop",
        "dns5:speech-drop",
    )


def test_duplicate_inventory_id_fails() -> None:
    item = _item("duplicate", speaker_id="speaker")

    with pytest.raises(SelectionError, match="duplicate inventory ID"):
        select_inventory([item, item])


def test_duplicate_policy_target_across_exclusions_fails() -> None:
    inventory = [_item("one", speaker_id="speaker")]
    exclusions = [
        ExclusionSpec(
            name="first",
            partition="validation",
            dataset="speech",
            source_ids=("one",),
        ),
        ExclusionSpec(
            name="second",
            partition="evaluation",
            dataset="speech",
            source_ids=("one",),
        ),
    ]

    with pytest.raises(SelectionError, match="duplicate source exclusion"):
        select_inventory(inventory, exclusions=exclusions)


def test_unresolved_exclusion_fails_but_uncurated_does_not_resolve_curations() -> None:
    inventory = [_item("known", speaker_id="speaker")]
    exclusion = ExclusionSpec(
        name="missing-exclusion",
        partition="validation",
        dataset="speech",
        source_ids=("missing",),
    )
    curation = CurationSpec(
        name="missing-curation",
        dataset="speech",
        source_ids=("missing",),
    )

    with pytest.raises(SelectionError, match="unresolved exclusion source ID"):
        select_inventory(inventory, selection="curated", exclusions=[exclusion])
    with pytest.raises(SelectionError, match=r"unresolved curation .* source ID"):
        select_inventory(inventory, selection="curated", curations=[curation])

    uncurated = select_inventory(inventory, selection="uncurated", curations=[curation])

    assert _ids(uncurated.training) == ("speech:known",)


def test_uncurated_does_not_iterate_curation_policy() -> None:
    def fail_if_iterated() -> Iterable[CurationSpec]:
        raise AssertionError("uncurated mode inspected curation policy")
        yield

    result = select_inventory(
        [_item("known", speaker_id="speaker")],
        selection="uncurated",
        curations=fail_if_iterated(),
    )

    assert _ids(result.training) == ("speech:known",)


def test_unscoped_source_and_speaker_ids_must_be_unambiguous() -> None:
    inventory = [
        _item("shared", dataset="one", speaker_id="shared-speaker"),
        _item("shared", dataset="two", speaker_id="shared-speaker"),
    ]

    with pytest.raises(SelectionError, match="ambiguous exclusion source ID"):
        select_inventory(
            inventory,
            exclusions=[
                ExclusionSpec(
                    name="ambiguous-source",
                    partition="validation",
                    source_ids=("shared",),
                )
            ],
        )

    with pytest.raises(SelectionError, match="ambiguous speaker exclusion"):
        select_inventory(
            inventory,
            exclusions=[
                ExclusionSpec(
                    name="ambiguous-speaker",
                    partition="validation",
                    speaker_ids=("shared-speaker",),
                )
            ],
        )
