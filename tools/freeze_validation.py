#!/usr/bin/env python3
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# Apache-2.0

"""Freeze the deterministic LRAC 2026 validation split.

This maintenance tool reads completed normalized inventories. It is not called by
``lrac-data prepare``: production preparation only consumes the checked-in CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from lrac_data.config import load_edition_config
from lrac_data.models import (
    CurationSpec,
    ExclusionPartition,
    ExclusionSpec,
    InventoryItem,
    MediaKind,
)
from lrac_data.selection import select_inventory
from lrac_data.state import atomic_write_text, sha256_file

HASH_NAMESPACE = "lrac-2026-validation"
GroupMode = Literal["speaker", "item"]
GroupKey = tuple[str, str]
AudioIdentity = tuple[str, int, int]


@dataclass(frozen=True, slots=True)
class Stratum:
    name: str
    dataset: str
    media_kind: MediaKind
    target: int
    minimum_groups: int
    contribution_cap: int
    group_mode: GroupMode
    language: str | None = None


SPEECH_SPEAKER_DATASET_LANGUAGES = (
    ("commonvoice_v26", "ar"),
    ("commonvoice_v26", "de"),
    ("commonvoice_v26", "es"),
    ("commonvoice_v26", "fa"),
    ("commonvoice_v26", "fr"),
    ("commonvoice_v26", "it"),
    ("commonvoice_v26", "ja"),
    ("commonvoice_v26", "pt"),
    ("commonvoice_v26", "ru"),
    ("commonvoice_v26", "sw"),
    ("commonvoice_v26", "zh-CN"),
    ("commonvoice_v26", "zh-HK"),
    ("commonvoice_v26", "zh-TW"),
    ("dns5", "en"),
    ("ears", "en"),
    ("globe", "en"),
    ("libritts", "en"),
    ("mls", "french"),
    ("mls", "german"),
    ("mls", "spanish"),
    ("openslr93", "cmn"),
    ("vctk", "en"),
)

SPEECH_STRATA = tuple(
    Stratum(
        f"{dataset}/{language}",
        dataset,
        MediaKind.SPEECH,
        250,
        10,
        25,
        "speaker",
        language,
    )
    for dataset, language in SPEECH_SPEAKER_DATASET_LANGUAGES
)

NOISE_STRATA = tuple(
    Stratum(f"{dataset}/noise", dataset, MediaKind.NOISE, 100, 100, 1, "item")
    for dataset in ("dns5", "fma", "fsd50k", "wham")
)

RIR_STRATA = tuple(
    Stratum(f"{dataset}/rir", dataset, MediaKind.RIR, 100, 100, 1, "item")
    for dataset in ("dns5", "motus")
)

STRATA = SPEECH_STRATA + NOISE_STRATA + RIR_STRATA


@dataclass(frozen=True, slots=True)
class CandidatePool:
    curated_items: tuple[InventoryItem, ...]
    eligible_items: tuple[InventoryItem, ...]
    evaluation_items: tuple[InventoryItem, ...]
    inventory_items: tuple[InventoryItem, ...]
    seed: int
    evaluation_exclusions: tuple[ExclusionSpec, ...] = ()
    curations: tuple[CurationSpec, ...] = ()


@dataclass(frozen=True, slots=True)
class FrozenSplit:
    validation_items: tuple[InventoryItem, ...]
    withheld_speakers: tuple[GroupKey, ...]
    withheld_source_items: tuple[InventoryItem, ...]
    evaluation_alias_items: tuple[InventoryItem, ...]


def stable_hash(*parts: object) -> bytes:
    """Return the byte ordering used by the frozen edition policy."""

    value = "\0".join(map(str, parts)).encode("utf-8")
    return hashlib.sha256(value).digest()


def freeze_validation_split(
    curated_items: Iterable[InventoryItem],
    *,
    eligible_items: Iterable[InventoryItem] | None = None,
    evaluation_items: Iterable[InventoryItem] = (),
    seed: int = 2026,
    strata: tuple[Stratum, ...] = STRATA,
) -> FrozenSplit:
    """Select exact validation items and everything needed for disjointness."""

    raw_curated = tuple(curated_items)
    raw_eligible = tuple(eligible_items) if eligible_items is not None else raw_curated
    evaluation = tuple(evaluation_items)
    evaluation_alias_items = _evaluation_aliases(raw_eligible, evaluation)
    evaluation_alias_ids = {item.id for item in evaluation_alias_items}
    eligible = tuple(item for item in raw_eligible if item.id not in evaluation_alias_ids)
    curated = _canonical_candidates(
        item for item in raw_curated if item.id not in evaluation_alias_ids
    )

    chosen_items: list[InventoryItem] = []
    selected_speakers: set[GroupKey] = set()
    for stratum in strata:
        candidates = tuple(item for item in curated if _matches_stratum(item, stratum))
        groups = _group_candidates(candidates, stratum)
        for group_id in _evaluation_group_ids(evaluation, stratum):
            groups.pop(group_id, None)
        selected = _choose_groups(groups, stratum, seed)
        selected_items = _select_from_groups(groups, selected, stratum, seed)
        chosen_items.extend(selected_items)

        if stratum.group_mode == "speaker":
            selected_speakers.update((stratum.dataset, group_id) for group_id in selected)

    validation_items = tuple(sorted(chosen_items, key=lambda item: (item.dataset, item.source_id)))
    identities = [(item.dataset, item.source_id) for item in validation_items]
    if len(identities) != len(set(identities)):
        raise ValueError("validation strata selected a source ID more than once")
    physical_audio = list(map(_audio_identity, validation_items))
    if len(physical_audio) != len(set(physical_audio)):
        raise ValueError(
            "validation strata selected identical source audio across dataset namespaces"
        )

    withheld_source_items = _withheld_source_items(
        eligible,
        validation_items,
        selected_speakers,
    )
    return FrozenSplit(
        validation_items=validation_items,
        withheld_speakers=tuple(sorted(selected_speakers)),
        withheld_source_items=withheld_source_items,
        evaluation_alias_items=evaluation_alias_items,
    )


def _matches_stratum(item: InventoryItem, stratum: Stratum) -> bool:
    return (
        item.dataset == stratum.dataset
        and item.media_kind is stratum.media_kind
        and (stratum.language is None or item.language == stratum.language)
    )


def _canonical_candidates(items: Iterable[InventoryItem]) -> tuple[InventoryItem, ...]:
    """Keep one deterministic source ID for each physical source or segment."""

    canonical: list[InventoryItem] = []
    for identity, aliases in sorted(_identity_index(items).items()):
        by_dataset: defaultdict[str, list[InventoryItem]] = defaultdict(list)
        for item in aliases:
            by_dataset[item.dataset].append(item)
        for dataset, dataset_aliases in sorted(by_dataset.items()):
            signatures = {_alias_signature(item) for item in dataset_aliases}
            if len(signatures) != 1:
                source_ids = ", ".join(sorted(item.id for item in dataset_aliases))
                raise ValueError(
                    f"{dataset}: audio identity {identity!r} has conflicting normalized "
                    f"metadata: {source_ids}"
                )
        canonical.append(min(aliases, key=lambda item: (item.dataset, item.source_id)))
    return tuple(sorted(canonical, key=lambda item: (item.dataset, item.source_id)))


def _alias_signature(item: InventoryItem) -> tuple[object, ...]:
    return (
        item.media_kind,
        item.speaker_id,
        item.text,
        item.language,
        item.gender,
    )


def _audio_identity(item: InventoryItem) -> AudioIdentity:
    if item.source_checksum is None:
        raise ValueError(f"validation candidate {item.id!r} has no source checksum")
    if item.source_segment is None:
        return (item.source_checksum, -1, -1)
    return (
        item.source_checksum,
        item.source_segment.start_us,
        item.source_segment.end_us,
    )


def _identity_index(
    items: Iterable[InventoryItem],
) -> dict[AudioIdentity, tuple[InventoryItem, ...]]:
    grouped: defaultdict[AudioIdentity, list[InventoryItem]] = defaultdict(list)
    for item in items:
        grouped[_audio_identity(item)].append(item)
    return {
        identity: tuple(sorted(aliases, key=lambda item: item.id))
        for identity, aliases in grouped.items()
    }


def _evaluation_aliases(
    eligible_items: Iterable[InventoryItem],
    evaluation_items: Iterable[InventoryItem],
) -> tuple[InventoryItem, ...]:
    eligible = _identity_index(eligible_items)
    evaluation = _identity_index(evaluation_items)
    return tuple(
        sorted(
            (item for identity in set(eligible) & set(evaluation) for item in eligible[identity]),
            key=lambda item: item.id,
        )
    )


def _withheld_source_items(
    eligible_items: Iterable[InventoryItem],
    validation_items: Iterable[InventoryItem],
    selected_speakers: set[GroupKey],
) -> tuple[InventoryItem, ...]:
    validation = tuple(validation_items)
    validation_ids = {item.id for item in validation}
    validation_audio = {_audio_identity(item) for item in validation}
    withheld: dict[str, InventoryItem] = {}
    for item in eligible_items:
        if item.id in validation_ids:
            continue
        covered_by_speaker = (
            item.speaker_id is not None and (item.dataset, item.speaker_id) in selected_speakers
        )
        physical_alias = _audio_identity(item) in validation_audio
        if not covered_by_speaker and physical_alias:
            withheld[item.id] = item
    return tuple(sorted(withheld.values(), key=lambda item: (item.dataset, item.source_id)))


def _group_candidates(
    candidates: Iterable[InventoryItem],
    stratum: Stratum,
) -> dict[str, tuple[InventoryItem, ...]]:
    grouped: defaultdict[str, list[InventoryItem]] = defaultdict(list)
    for item in candidates:
        if stratum.group_mode == "speaker":
            if item.speaker_id is None:
                continue
            group_id = item.speaker_id
        else:
            group_id = item.source_id
        grouped[group_id].append(item)
    return {
        group_id: tuple(sorted(items, key=lambda item: item.source_id))
        for group_id, items in grouped.items()
    }

def _evaluation_group_ids(evaluation_items: Iterable[InventoryItem], stratum: Stratum) -> set[str]:
    if stratum.media_kind is not MediaKind.SPEECH:
        return set()
    group_ids: set[str] = set()
    for item in evaluation_items:
        if item.dataset != stratum.dataset or item.media_kind is not MediaKind.SPEECH:
            continue
        if stratum.group_mode == "speaker" and item.speaker_id is not None:
            group_ids.add(item.speaker_id)
    return group_ids


def _choose_groups(
    groups: dict[str, tuple[InventoryItem, ...]],
    stratum: Stratum,
    seed: int,
) -> tuple[str, ...]:
    ordered = sorted(
        groups,
        key=lambda group_id: (
            stable_hash(HASH_NAMESPACE, seed, "group", stratum.name, group_id),
            group_id,
        ),
    )
    selected: list[str] = []
    capacity = 0
    for group_id in ordered:
        selected.append(group_id)
        capacity += min(len(groups[group_id]), stratum.contribution_cap)
        if len(selected) >= stratum.minimum_groups and capacity >= stratum.target:
            break
    if len(selected) < stratum.minimum_groups or capacity < stratum.target:
        raise ValueError(
            f"{stratum.name}: insufficient curated capacity "
            f"({capacity} items from {len(selected)} groups; "
            f"requires {stratum.target} items from at least {stratum.minimum_groups} groups)"
        )
    return tuple(selected)


def _select_from_groups(
    groups: dict[str, tuple[InventoryItem, ...]],
    selected_groups: tuple[str, ...],
    stratum: Stratum,
    seed: int,
) -> list[InventoryItem]:
    def item_key(item: InventoryItem) -> tuple[bytes, str]:
        return (
            stable_hash(HASH_NAMESPACE, seed, "item", stratum.name, item.source_id),
            item.source_id,
        )

    chosen: list[InventoryItem] = []
    chosen_ids: set[str] = set()
    counts: defaultdict[str, int] = defaultdict(int)
    for group_id in selected_groups:
        item = min(groups[group_id], key=item_key)
        chosen.append(item)
        chosen_ids.add(item.id)
        counts[group_id] += 1

    remaining = sorted(
        (
            (item_key(item), group_id, item)
            for group_id in selected_groups
            for item in groups[group_id]
            if item.id not in chosen_ids
        ),
        key=lambda value: (value[0], value[1]),
    )
    for _, group_id, item in remaining:
        if len(chosen) == stratum.target:
            break
        if counts[group_id] == stratum.contribution_cap:
            continue
        chosen.append(item)
        counts[group_id] += 1
    if len(chosen) != stratum.target:
        raise ValueError(f"{stratum.name}: selected {len(chosen)} of {stratum.target} items")
    return chosen


def load_candidate_pool(
    inventory_root: Path,
    repo_root: Path,
) -> CandidatePool:
    """Load one normalized inventory per dataset and apply curated eligibility."""

    inventory_root = inventory_root.expanduser().resolve()
    loaded = load_edition_config("2026", repo_root=repo_root, selection="curated")
    inventory: list[InventoryItem] = []
    for dataset in loaded.config.datasets:
        path = inventory_root / f"{dataset.id}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"normalized inventory is missing: {path}")
        items = _read_inventory(path)
        unexpected = sorted({item.dataset for item in items if item.dataset != dataset.id})
        if unexpected:
            raise ValueError(
                f"{path}: inventory for {dataset.id!r} contains datasets: " + ", ".join(unexpected)
            )
        inventory.extend(items)

    inventory = list(_hydrate_source_checksums(inventory))

    evaluation_exclusions = tuple(
        exclusion
        for exclusion in loaded.config.exclusions
        if exclusion.partition is ExclusionPartition.EVALUATION
    )
    selected = select_inventory(
        inventory,
        selection="curated",
        exclusions=evaluation_exclusions,
        curations=loaded.config.curations,
    )
    eligible = tuple(
        sorted((*selected.training, *selected.quality_rejected), key=lambda item: item.id)
    )
    configured = {
        (dataset.id, kind) for dataset in loaded.config.datasets for kind in dataset.media_kinds
    }
    policy = {(stratum.dataset, stratum.media_kind) for stratum in STRATA}
    languages = {
        (item.dataset, item.language) for item in eligible if item.media_kind is MediaKind.SPEECH
    }
    expected_languages = {(stratum.dataset, stratum.language) for stratum in SPEECH_STRATA}
    if configured != policy or languages != expected_languages:
        raise ValueError("inventories do not exactly match the 2026 validation policy")
    return CandidatePool(
        curated_items=selected.training,
        eligible_items=eligible,
        evaluation_items=selected.evaluation,
        inventory_items=tuple(sorted(inventory, key=lambda item: item.id)),
        seed=loaded.config.seed,
        evaluation_exclusions=evaluation_exclusions,
        curations=loaded.config.curations,
    )


def _read_inventory(path: Path) -> tuple[InventoryItem, ...]:
    """Read normalized inventory using the current release contract."""

    normalized: list[InventoryItem] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                normalized.append(InventoryItem.model_validate_json(line))
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid inventory row: {error}") from error
    return tuple(normalized)


def _hydrate_source_checksums(
    items: Iterable[InventoryItem],
) -> tuple[InventoryItem, ...]:
    """Fill checksum gaps once per source file."""

    by_path: defaultdict[Path, list[InventoryItem]] = defaultdict(list)
    for item in items:
        by_path[item.source_path.expanduser().resolve()].append(item)

    hydrated: list[InventoryItem] = []
    for source_path, source_items in sorted(by_path.items(), key=lambda value: str(value[0])):
        supplied = {item.source_checksum for item in source_items if item.source_checksum}
        if len(supplied) > 1:
            raise ValueError(f"conflicting source checksums for {source_path}")
        if supplied:
            checksum = next(iter(supplied))
        else:
            if not source_path.is_file():
                raise ValueError(
                    f"source checksum is missing and source file is unavailable: {source_path}"
                )
            checksum = sha256_file(source_path)
        hydrated.extend(
            item
            if item.source_checksum == checksum
            else item.model_copy(update={"source_checksum": checksum})
            for item in source_items
        )
    return tuple(sorted(hydrated, key=lambda item: item.id))


def generated_exclusions(frozen: FrozenSplit) -> tuple[ExclusionSpec, ...]:
    """Build the typed policy represented by the generated exclusion CSV."""

    exclusions: list[ExclusionSpec] = []
    sources = (
        ("validation-items", ExclusionPartition.VALIDATION, frozen.validation_items),
        (
            "validation-source-withholding",
            ExclusionPartition.WITHHELD,
            frozen.withheld_source_items,
        ),
        (
            "evaluation-source-withholding",
            ExclusionPartition.WITHHELD,
            frozen.evaluation_alias_items,
        ),
    )
    for name, partition, items in sources:
        grouped: defaultdict[str, list[str]] = defaultdict(list)
        for item in items:
            grouped[item.dataset].append(item.source_id)
        for dataset, source_ids in sorted(grouped.items()):
            exclusions.append(
                ExclusionSpec(
                    name=name,
                    partition=partition,
                    dataset=dataset,
                    source_ids=tuple(sorted(source_ids)),
                )
            )

    speakers: defaultdict[str, list[str]] = defaultdict(list)
    for dataset, speaker_id in frozen.withheld_speakers:
        speakers[dataset].append(speaker_id)
    for dataset, speaker_ids in sorted(speakers.items()):
        exclusions.append(
            ExclusionSpec(
                name="validation-speakers",
                partition=ExclusionPartition.WITHHELD,
                dataset=dataset,
                speaker_ids=tuple(sorted(speaker_ids)),
            )
        )
    return tuple(exclusions)


def validate_frozen_policy(pool: CandidatePool, frozen: FrozenSplit) -> None:
    """Resolve the generated policy against both public selection modes."""

    exclusions = (*pool.evaluation_exclusions, *generated_exclusions(frozen))
    results = {
        mode: select_inventory(
            pool.inventory_items,
            selection=mode,
            exclusions=exclusions,
            curations=pool.curations,
        )
        for mode in ("curated", "uncurated")
    }
    expected = {
        "validation": {item.id for item in frozen.validation_items},
        "evaluation": {item.id for item in pool.evaluation_items},
    }
    for mode, selected in results.items():
        actual = {
            "validation": {item.id for item in selected.validation},
            "evaluation": {item.id for item in selected.evaluation},
        }
        for partition in expected:
            if actual[partition] != expected[partition]:
                raise ValueError(f"{mode}: {partition} membership changed during freeze")
        _ensure_partition_disjointness(selected.training, selected.validation, selected.evaluation)

    curated_withheld = {item.id for item in results["curated"].withheld}
    uncurated_withheld = {item.id for item in results["uncurated"].withheld}
    if curated_withheld != uncurated_withheld:
        raise ValueError("curated and uncurated modes resolved different withheld membership")


def _ensure_partition_disjointness(
    training: Iterable[InventoryItem],
    validation: Iterable[InventoryItem],
    evaluation: Iterable[InventoryItem],
) -> None:
    partitions = tuple(map(tuple, (training, validation, evaluation)))
    identities = tuple({_audio_identity(item) for item in items} for items in partitions)
    names = ("training", "validation", "evaluation")
    for left, right in ((0, 1), (0, 2), (1, 2)):
        if identities[left] & identities[right]:
            raise ValueError(f"physical audio overlaps {names[left]} and {names[right]}")


def render_csv(frozen: FrozenSplit) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("name", "partition", "dataset", "source_id", "speaker_id"))
    for exclusion in generated_exclusions(frozen):
        for source_id in exclusion.source_ids:
            writer.writerow(
                (exclusion.name, exclusion.partition.value, exclusion.dataset, source_id, "")
            )
        for speaker_id in exclusion.speaker_ids:
            writer.writerow(
                (exclusion.name, exclusion.partition.value, exclusion.dataset, "", speaker_id)
            )
    return output.getvalue()


def render_report(
    frozen: FrozenSplit,
    pool: CandidatePool,
    csv_text: str,
    *,
    strata: tuple[Stratum, ...] = STRATA,
) -> str:
    report = {
        "edition": "2026",
        "policy": {
            "hash_namespace": HASH_NAMESPACE,
            "seed": pool.seed,
            "strata": [asdict(stratum) for stratum in strata],
            "group_order": "sha256(NUL-joined namespace, seed, group, stratum, group_id)",
            "item_order": "sha256(NUL-joined namespace, seed, item, stratum, source_id)",
            "speech_disjointness": "speaker where available",
        },
        "input": {
            "inventory_items": len(pool.inventory_items),
            "curated_eligible_items": len(pool.curated_items),
            "evaluation_items": len(pool.evaluation_items),
            "evaluation_source_id_sha256": _source_id_digest(pool.evaluation_items),
        },
        "result": {
            "validation_source_ids": len(frozen.validation_items),
            "withheld_speakers": len(frozen.withheld_speakers),
            "withheld_source_ids": len(frozen.withheld_source_items),
            "withheld_evaluation_alias_source_ids": len(frozen.evaluation_alias_items),
            "validation_source_id_sha256": _source_id_digest(frozen.validation_items),
            "withheld_source_id_sha256": _source_id_digest(frozen.withheld_source_items),
            "withheld_evaluation_alias_source_id_sha256": _source_id_digest(
                frozen.evaluation_alias_items
            ),
            "exclusions_csv_sha256": hashlib.sha256(csv_text.encode("utf-8")).hexdigest(),
        },
    }
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def _source_id_digest(items: Iterable[InventoryItem]) -> str:
    digest = hashlib.sha256()
    for item in sorted(items, key=lambda value: (value.dataset, value.source_id)):
        digest.update(f"{item.dataset}\t{item.source_id}\n".encode())
    return digest.hexdigest()


def _write_or_check(path: Path, expected: str, *, check: bool) -> None:
    if check:
        if not path.is_file() or path.read_text(encoding="utf-8") != expected:
            raise SystemExit(f"frozen validation file differs: {path}")
    else:
        atomic_write_text(path, expected)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory-root",
        type=Path,
        required=True,
        help="directory containing one <dataset>.jsonl inventory per configured dataset",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the checked-in files without changing them",
    )
    args = parser.parse_args()

    pool = load_candidate_pool(args.inventory_root, repo_root)
    if pool.seed != 2026:
        raise ValueError(f"edition seed must be 2026, got {pool.seed}")
    frozen = freeze_validation_split(
        pool.curated_items,
        eligible_items=pool.eligible_items,
        evaluation_items=pool.evaluation_items,
        seed=pool.seed,
    )
    validate_frozen_policy(pool, frozen)
    csv_text = render_csv(frozen)
    report_text = render_report(frozen, pool, csv_text)
    output_root = repo_root / "metadata/editions/2026/validation"
    _write_or_check(output_root / "exclusions.csv", csv_text, check=args.check)
    _write_or_check(output_root / "split.json", report_text, check=args.check)

    print(
        f"validation={len(frozen.validation_items)} "
        f"withheld_speakers={len(frozen.withheld_speakers)} "
        f"withheld_source_items={len(frozen.withheld_source_items)} "
        f"withheld_evaluation_aliases={len(frozen.evaluation_alias_items)} "
        f"sha256={_source_id_digest(frozen.validation_items)}"
    )


if __name__ == "__main__":
    main()
