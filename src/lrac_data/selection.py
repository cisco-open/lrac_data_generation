# Copyright 2026 Cisco Systems, Inc. and its affiliates
# Apache-2.0

"""Edition selection policy over a normalized source inventory."""

from __future__ import annotations

from collections.abc import Iterable

from lrac_data.models import (
    CurationSpec,
    ExclusionPartition,
    ExclusionSpec,
    InventoryItem,
    MediaKind,
    SelectionMode,
    SelectionResult,
    qualify_id,
)

SourceKey = tuple[str, str]
SpeakerKey = tuple[str, str]
CurationScope = tuple[str, MediaKind]


class SelectionError(ValueError):
    """Raised when policy cannot partition an inventory unambiguously."""


def select_inventory(
    inventory: Iterable[InventoryItem],
    *,
    selection: SelectionMode | str = SelectionMode.CURATED,
    exclusions: Iterable[ExclusionSpec] = (),
    curations: Iterable[CurationSpec] = (),
) -> SelectionResult:
    """Apply frozen exclusions and optional quality allowlists."""

    try:
        mode = SelectionMode(selection)
    except ValueError as error:
        supported = ", ".join(item.value for item in SelectionMode)
        raise SelectionError(
            f"unknown selection mode {selection!r}; expected one of {supported}"
        ) from error

    exact, excluded_speakers = _exclusion_targets(exclusions)
    curation_scopes: set[CurationScope] = set()
    curation_targets: dict[SourceKey, MediaKind] = {}
    if mode is SelectionMode.CURATED:
        curation_scopes, curation_targets = _curation_targets(curations)

    training: list[InventoryItem] = []
    validation: list[InventoryItem] = []
    evaluation: list[InventoryItem] = []
    withheld: list[InventoryItem] = []
    quality_rejected: list[InventoryItem] = []
    validation_speakers: set[SpeakerKey] = set()
    training_speakers: set[SpeakerKey] = set()
    resolved_speakers: set[SpeakerKey] = set()
    previous_id: str | None = None

    for item in sorted(inventory, key=lambda candidate: candidate.id):
        if item.id == previous_id:
            raise SelectionError(
                "duplicate inventory source ID: "
                f"dataset={item.dataset!r}, source_id={item.source_id!r}"
            )
        previous_id = item.id

        source_key = (item.dataset, item.source_id)
        speaker_key = (item.dataset, item.speaker_id) if item.speaker_id is not None else None
        if speaker_key in excluded_speakers:
            resolved_speakers.add(speaker_key)

        scope = (item.dataset, item.media_kind)
        curated = scope not in curation_scopes
        expected_kind = curation_targets.get(source_key)
        if expected_kind is item.media_kind:
            curated = True
            del curation_targets[source_key]

        partition = exact.pop(source_key, None)
        if partition is ExclusionPartition.VALIDATION:
            validation.append(item)
            if speaker_key is not None:
                validation_speakers.add(speaker_key)
        elif partition is ExclusionPartition.EVALUATION:
            evaluation.append(item)
        elif partition is ExclusionPartition.WITHHELD or speaker_key in excluded_speakers:
            withheld.append(item)
        else:
            if speaker_key is not None:
                training_speakers.add(speaker_key)
            (training if curated else quality_rejected).append(item)

    _require_resolved(exact, excluded_speakers - resolved_speakers, curation_targets)
    leakage = sorted(validation_speakers & training_speakers)
    if leakage:
        names = ", ".join(qualify_id(*speaker) for speaker in leakage)
        raise SelectionError(
            "validation speakers remain eligible for training: "
            f"{names}; add speaker-withheld exclusions"
        )

    return SelectionResult(
        training=tuple(training),
        validation=tuple(validation),
        evaluation=tuple(evaluation),
        withheld=tuple(withheld),
        quality_rejected=tuple(quality_rejected),
    )


def _exclusion_targets(
    exclusions: Iterable[ExclusionSpec],
) -> tuple[dict[SourceKey, ExclusionPartition], set[SpeakerKey]]:
    exact: dict[SourceKey, ExclusionPartition] = {}
    speakers: set[SpeakerKey] = set()
    for exclusion in exclusions:
        for source_id in exclusion.source_ids:
            key = (exclusion.dataset, source_id)
            if key in exact:
                raise SelectionError(
                    f"duplicate source exclusion {source_id!r} in dataset {exclusion.dataset!r}"
                )
            exact[key] = exclusion.partition
        for speaker_id in exclusion.speaker_ids:
            key = (exclusion.dataset, speaker_id)
            if key in speakers:
                raise SelectionError(
                    f"duplicate speaker exclusion {speaker_id!r} in dataset {exclusion.dataset!r}"
                )
            speakers.add(key)
    return exact, speakers


def _curation_targets(
    curations: Iterable[CurationSpec],
) -> tuple[set[CurationScope], dict[SourceKey, MediaKind]]:
    scopes: set[CurationScope] = set()
    targets: dict[SourceKey, MediaKind] = {}
    for curation in curations:
        scope = (curation.dataset, curation.media_kind)
        if scope in scopes:
            raise SelectionError(
                f"duplicate curation scope for dataset {curation.dataset!r}, "
                f"media kind {curation.media_kind.value!r}"
            )
        scopes.add(scope)
        for source_id in curation.source_ids:
            key = (curation.dataset, source_id)
            if key in targets:
                raise SelectionError(
                    f"duplicate curation target {source_id!r} in dataset {curation.dataset!r}"
                )
            targets[key] = curation.media_kind
    return scopes, targets


def _require_resolved(
    exact: dict[SourceKey, ExclusionPartition],
    speakers: set[SpeakerKey],
    curations: dict[SourceKey, MediaKind],
) -> None:
    if exact:
        dataset, source_id = min(exact)
        raise SelectionError(f"unresolved exclusion source ID {source_id!r} in dataset {dataset!r}")
    if speakers:
        dataset, speaker_id = min(speakers)
        raise SelectionError(f"unresolved speaker exclusion {speaker_id!r} in dataset {dataset!r}")
    if curations:
        (dataset, source_id), media_kind = min(curations.items())
        raise SelectionError(
            f"unresolved curation source ID {source_id!r} in dataset {dataset!r}, "
            f"media kind {media_kind.value!r}"
        )
