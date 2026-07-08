"""Edition selection policy over a normalized source inventory."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from lrac_data.models import (
    CurationAction,
    CurationSpec,
    ExclusionSpec,
    InventoryItem,
    MediaKind,
    SelectionMode,
    SelectionResult,
    Split,
    qualify_id,
)


class SelectionError(ValueError):
    """Raised when policy cannot partition an inventory unambiguously."""


def select_inventory(
    inventory: Iterable[InventoryItem],
    *,
    selection: SelectionMode | str = SelectionMode.CURATED,
    exclusions: Iterable[ExclusionSpec] = (),
    curations: Iterable[CurationSpec] = (),
) -> SelectionResult:
    """Apply frozen split exclusions and optional quality curation.

    Validation and evaluation exclusions are resolved first and are identical in
    both public selection modes.  Curated mode validates quality rules and applies
    them to training and validation items; evaluation membership is preserved.
    Uncurated mode does not inspect quality rules.  The returned tuples are sorted
    by stable item ID regardless of adapter iteration order.

    Raises:
        SelectionError: If inventory IDs are duplicated, an exclusion is invalid,
            or a curation target used by curated mode is duplicated, unresolved,
            or ambiguous.
    """

    try:
        mode = SelectionMode(selection)
    except ValueError as error:
        supported = ", ".join(item.value for item in SelectionMode)
        raise SelectionError(
            f"unknown selection mode {selection!r}; expected one of {supported}"
        ) from error

    items = tuple(sorted(inventory, key=lambda item: item.id))
    index = _index_inventory(items)
    exclusions = tuple(exclusions)

    partition_by_id = _resolve_exclusions(index, exclusions)

    validation_candidates: list[InventoryItem] = []
    evaluation: list[InventoryItem] = []
    training_candidates: list[InventoryItem] = []
    for item in items:
        partition = partition_by_id.get(item.id)
        if partition is Split.VALIDATION:
            validation_candidates.append(item)
        elif partition is Split.EVALUATION:
            evaluation.append(item)
        else:
            training_candidates.append(item)

    if mode is SelectionMode.UNCURATED:
        training = training_candidates
        validation = validation_candidates
        quality_rejected: list[InventoryItem] = []
    else:
        include_by_scope, quality_excluded = _resolve_curations(index, tuple(curations))
        training = []
        validation = []
        quality_rejected = []
        for candidates, accepted_items in (
            (training_candidates, training),
            (validation_candidates, validation),
        ):
            for item in candidates:
                if _is_curation_eligible(item, include_by_scope, quality_excluded):
                    accepted_items.append(item)
                else:
                    quality_rejected.append(item)
        quality_rejected.sort(key=lambda item: item.id)

    return SelectionResult(
        selection=mode,
        training=tuple(training),
        validation=tuple(validation),
        evaluation=tuple(evaluation),
        quality_rejected=tuple(quality_rejected),
    )


def _is_curation_eligible(
    item: InventoryItem,
    include_by_scope: dict[tuple[str, MediaKind | None], set[str]],
    excluded: set[str],
) -> bool:
    allowlist = include_by_scope.get((item.dataset, item.media_kind))
    if allowlist is None:
        allowlist = include_by_scope.get((item.dataset, None))
    return item.id not in excluded and (allowlist is None or item.id in allowlist)


@dataclass(frozen=True)
class _InventoryIndex:
    items: tuple[InventoryItem, ...]
    by_id: dict[str, InventoryItem]
    by_source_id: dict[str, tuple[InventoryItem, ...]]
    by_dataset_source: dict[tuple[str, str], InventoryItem]
    speaker_groups: dict[tuple[str, str], tuple[InventoryItem, ...]]
    speaker_aliases: dict[str, tuple[tuple[str, str], ...]]


def _index_inventory(items: tuple[InventoryItem, ...]) -> _InventoryIndex:
    by_id: dict[str, InventoryItem] = {}
    by_source_id_lists: defaultdict[str, list[InventoryItem]] = defaultdict(list)
    by_dataset_source: dict[tuple[str, str], InventoryItem] = {}
    speaker_group_lists: defaultdict[tuple[str, str], list[InventoryItem]] = defaultdict(list)
    for item in items:
        if item.id in by_id:
            raise SelectionError(f"duplicate inventory ID: {item.id!r}")
        by_id[item.id] = item

        source_key = (item.dataset, item.source_id)
        if source_key in by_dataset_source:
            raise SelectionError(
                "duplicate inventory source ID: "
                f"dataset={item.dataset!r}, source_id={item.source_id!r}"
            )
        by_dataset_source[source_key] = item
        by_source_id_lists[item.source_id].append(item)
        if item.speaker_id is not None:
            speaker_group_lists[(item.dataset, item.speaker_id)].append(item)

    speaker_alias_lists: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    for speaker_key in speaker_group_lists:
        dataset, speaker_id = speaker_key
        speaker_alias_lists[speaker_id].append(speaker_key)
        speaker_alias_lists[qualify_id(dataset, speaker_id)].append(speaker_key)

    return _InventoryIndex(
        items=items,
        by_id=by_id,
        by_source_id={key: tuple(value) for key, value in by_source_id_lists.items()},
        by_dataset_source=by_dataset_source,
        speaker_groups={key: tuple(value) for key, value in speaker_group_lists.items()},
        speaker_aliases={
            key: tuple(dict.fromkeys(value)) for key, value in speaker_alias_lists.items()
        },
    )


def _resolve_exclusions(
    index: _InventoryIndex,
    exclusions: tuple[ExclusionSpec, ...],
) -> dict[str, Split]:
    partition_by_id: dict[str, Split] = {}
    reason_by_id: dict[str, str] = {}
    seen_source_targets: set[tuple[str | None, str]] = set()
    seen_speaker_targets: set[tuple[str | None, str]] = set()

    for exclusion in exclusions:
        for target in exclusion.source_ids:
            key = (exclusion.dataset, target)
            if key in seen_source_targets:
                raise SelectionError(
                    f"duplicate source exclusion {target!r} in dataset {exclusion.dataset!r}"
                )
            seen_source_targets.add(key)
            matches = _resolve_source(index, target, exclusion.dataset, "exclusion")
            _assign_partition(
                matches,
                exclusion.partition,
                f"source exclusion {exclusion.name!r}:{target!r}",
                partition_by_id,
                reason_by_id,
            )

        for target in exclusion.speaker_ids:
            key = (exclusion.dataset, target)
            if key in seen_speaker_targets:
                raise SelectionError(
                    f"duplicate speaker exclusion {target!r} in dataset {exclusion.dataset!r}"
                )
            seen_speaker_targets.add(key)
            matches = _resolve_speaker(index, target, exclusion.dataset)
            _assign_partition(
                matches,
                exclusion.partition,
                f"speaker exclusion {exclusion.name!r}:{target!r}",
                partition_by_id,
                reason_by_id,
            )

    return partition_by_id


def _assign_partition(
    matches: tuple[InventoryItem, ...],
    partition: Split,
    reason: str,
    partition_by_id: dict[str, Split],
    reason_by_id: dict[str, str],
) -> None:
    for item in matches:
        previous = partition_by_id.get(item.id)
        if previous is not None:
            raise SelectionError(
                f"item {item.id!r} is excluded more than once: "
                f"{reason_by_id[item.id]} ({previous.value}) and {reason} "
                f"({partition.value})"
            )
        partition_by_id[item.id] = partition
        reason_by_id[item.id] = reason


def _resolve_curations(
    index: _InventoryIndex,
    curations: tuple[CurationSpec, ...],
) -> tuple[dict[tuple[str, MediaKind | None], set[str]], set[str]]:
    include_by_scope: defaultdict[tuple[str, MediaKind | None], set[str]] = defaultdict(set)
    excluded: set[str] = set()
    seen_targets: dict[tuple[str, MediaKind | None, str], CurationAction] = {}
    resolved_targets: dict[tuple[str, MediaKind | None], CurationAction] = {}

    for curation in curations:
        for target in curation.source_ids:
            key = (curation.dataset, curation.media_kind, target)
            previous_action = seen_targets.get(key)
            if previous_action is not None:
                raise SelectionError(
                    f"duplicate curation target {target!r} in dataset "
                    f"{curation.dataset!r}, media kind {curation.media_kind!r} "
                    f"({previous_action.value} and "
                    f"{curation.action.value})"
                )
            seen_targets[key] = curation.action

            match = _resolve_source(
                index,
                target,
                curation.dataset,
                f"curation {curation.name!r}",
                media_kind=curation.media_kind,
            )[0]
            resolved_key = (match.id, curation.media_kind)
            previous_action = resolved_targets.get(resolved_key)
            if previous_action is not None:
                raise SelectionError(
                    f"item {match.id!r} is curated more than once "
                    f"({previous_action.value} and {curation.action.value})"
                )
            resolved_targets[resolved_key] = curation.action

            if curation.action is CurationAction.INCLUDE:
                include_by_scope[(curation.dataset, curation.media_kind)].add(match.id)
            else:
                excluded.add(match.id)

    return dict(include_by_scope), excluded


def _resolve_source(
    index: _InventoryIndex,
    target: str,
    dataset: str | None,
    policy_name: str,
    *,
    media_kind: MediaKind | None = None,
) -> tuple[InventoryItem, ...]:
    candidates: dict[str, InventoryItem] = {}
    qualified_match = index.by_id.get(target)
    if qualified_match is not None:
        candidates[qualified_match.id] = qualified_match
    if dataset is None:
        for item in index.by_source_id.get(target, ()):
            candidates[item.id] = item
    else:
        local_match = index.by_dataset_source.get((dataset, target))
        if local_match is not None:
            candidates[local_match.id] = local_match
    matches = tuple(
        item
        for item in sorted(candidates.values(), key=lambda candidate: candidate.id)
        if (dataset is None or item.dataset == dataset)
        and (media_kind is None or item.media_kind is media_kind)
    )
    if not matches:
        scope = f" in dataset {dataset!r}" if dataset is not None else ""
        if media_kind is not None:
            scope += f", media kind {media_kind.value!r}"
        raise SelectionError(f"unresolved {policy_name} source ID {target!r}{scope}")
    if len(matches) > 1:
        candidate_ids = ", ".join(item.id for item in matches)
        raise SelectionError(
            f"ambiguous {policy_name} source ID {target!r}; matches {candidate_ids}"
        )
    return matches


def _resolve_speaker(
    index: _InventoryIndex,
    target: str,
    dataset: str | None,
) -> tuple[InventoryItem, ...]:
    speaker_keys = tuple(
        speaker_key
        for speaker_key in index.speaker_aliases.get(target, ())
        if dataset is None or speaker_key[0] == dataset
    )
    if not speaker_keys:
        scope = f" in dataset {dataset!r}" if dataset is not None else ""
        raise SelectionError(f"unresolved speaker exclusion {target!r}{scope}")

    if len(speaker_keys) > 1:
        candidates = ", ".join(
            qualify_id(item_dataset, speaker_id)
            for item_dataset, speaker_id in sorted(speaker_keys)
        )
        raise SelectionError(f"ambiguous speaker exclusion {target!r}; matches {candidates}")
    return index.speaker_groups[speaker_keys[0]]


__all__ = ["SelectionError", "select_inventory"]
