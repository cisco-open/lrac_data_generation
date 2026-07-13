#!/usr/bin/env python3
"""Freeze the deterministic LRAC 2026 speech validation split.

This maintenance tool reads completed normalized inventories. It is not called by
`lrac-data prepare`: production preparation only consumes the checked-in CSV.
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

from lrac_data.config import load_edition_config
from lrac_data.manifests import read_jsonl
from lrac_data.models import ExclusionSpec, InventoryItem, MediaKind
from lrac_data.pipeline import WorkspaceLayout
from lrac_data.selection import select_inventory
from lrac_data.state import atomic_write_text

HASH_NAMESPACE = "lrac-2026-validation-v1"
MAX_CURATED_ITEMS_PER_SPEAKER = 250
SpeakerKey = tuple[str, str]


@dataclass(frozen=True)
class Stratum:
    name: str
    dataset: str
    target: int
    minimum_speakers: int
    contribution_cap: int
    language: str | None = None
    gender: str | None = None


STRATA = (
    Stratum("dns5", "dns5", 250, 10, 25),
    Stratum("ears/f", "ears", 125, 5, 25, gender="f"),
    Stratum("ears/m", "ears", 125, 5, 25, gender="m"),
    Stratum("globe/f", "globe", 125, 5, 25, gender="f"),
    Stratum("globe/m", "globe", 125, 5, 25, gender="m"),
    Stratum("libritts/f", "libritts", 125, 5, 25, gender="f"),
    Stratum("libritts/m", "libritts", 125, 5, 25, gender="m"),
    Stratum("vctk/f", "vctk", 125, 5, 25, gender="f"),
    Stratum("vctk/m", "vctk", 125, 5, 25, gender="m"),
    Stratum("mls/french", "mls", 100, 4, 25, language="french"),
    Stratum("mls/german", "mls", 100, 4, 25, language="german"),
    Stratum("mls/spanish", "mls", 100, 4, 25, language="spanish"),
)


@dataclass(frozen=True)
class Allocation:
    stratum: str
    target_source_ids: int
    selected_speakers: tuple[str, ...]
    validation_items_per_speaker: dict[str, int]


@dataclass(frozen=True)
class FrozenSplit:
    validation_items: tuple[InventoryItem, ...]
    withheld_speakers: tuple[SpeakerKey, ...]
    allocations: tuple[Allocation, ...]

    @property
    def source_id_checksum(self) -> str:
        digest = hashlib.sha256()
        for item in self.validation_items:
            digest.update(f"{item.dataset}\t{item.source_id}\n".encode())
        return digest.hexdigest()

    @property
    def audio_identity_checksum(self) -> str:
        digest = hashlib.sha256()
        identities = sorted((item.dataset, item.source_checksum) for item in self.validation_items)
        for dataset, source_checksum in identities:
            assert source_checksum is not None
            digest.update(f"{dataset}\t{source_checksum}\n".encode())
        return digest.hexdigest()


def stable_hash(*parts: object) -> bytes:
    """Return the stable byte ordering used by the frozen edition policy."""

    value = "\0".join(map(str, parts)).encode()
    return hashlib.sha256(value).digest()


def freeze_validation_split(
    items: Iterable[InventoryItem],
    *,
    evaluation_speakers: frozenset[SpeakerKey] = frozenset(),
    seed: int = 2026,
    strata: tuple[Stratum, ...] = STRATA,
) -> FrozenSplit:
    """Select exact validation items and the speakers withheld around them."""

    grouped: defaultdict[SpeakerKey, list[InventoryItem]] = defaultdict(list)
    canonical_items, raw_speaker_counts = _canonical_validation_candidates(items, strata)
    for item in canonical_items:
        assert item.speaker_id is not None
        grouped[(item.dataset, item.speaker_id)].append(item)

    chosen_items: list[InventoryItem] = []
    withheld_speakers: set[SpeakerKey] = set()
    allocations: list[Allocation] = []
    for stratum in strata:
        candidates: list[tuple[bytes, str, SpeakerKey, tuple[InventoryItem, ...]]] = []
        for speaker_key, speaker_items in grouped.items():
            dataset, speaker_id = speaker_key
            if (
                dataset != stratum.dataset
                or speaker_key in evaluation_speakers
                or raw_speaker_counts[speaker_key] > MAX_CURATED_ITEMS_PER_SPEAKER
            ):
                continue
            matching = tuple(item for item in speaker_items if _matches_stratum(item, stratum))
            if matching:
                candidates.append(
                    (
                        stable_hash(
                            HASH_NAMESPACE,
                            seed,
                            "speaker",
                            stratum.name,
                            speaker_id,
                        ),
                        speaker_id,
                        speaker_key,
                        matching,
                    )
                )
        candidates.sort(key=lambda value: (value[0], value[1]))

        selected: list[tuple[SpeakerKey, tuple[InventoryItem, ...]]] = []
        capacity = 0
        for _, _, speaker_key, matching in candidates:
            selected.append((speaker_key, matching))
            capacity += min(len(matching), stratum.contribution_cap)
            if len(selected) >= stratum.minimum_speakers and capacity >= stratum.target:
                break
        if len(selected) < stratum.minimum_speakers or capacity < stratum.target:
            raise ValueError(
                f"{stratum.name}: insufficient eligible capacity "
                f"({capacity} items from {len(selected)} speakers)"
            )

        selected_items, counts = _select_items(selected, stratum, seed)
        chosen_items.extend(selected_items)
        withheld_speakers.update(speaker_key for speaker_key, _ in selected)
        allocations.append(
            Allocation(
                stratum=stratum.name,
                target_source_ids=stratum.target,
                selected_speakers=tuple(speaker_id for (_, speaker_id), _ in selected),
                validation_items_per_speaker=dict(sorted(counts.items())),
            )
        )

    ordered_items = tuple(sorted(chosen_items, key=lambda item: (item.dataset, item.source_id)))
    item_keys = [(item.dataset, item.source_id) for item in ordered_items]
    if len(item_keys) != len(set(item_keys)):
        raise ValueError("validation strata selected a source ID more than once")
    source_checksums = [item.source_checksum for item in ordered_items]
    if len(source_checksums) != len(set(source_checksums)):
        raise ValueError(
            "validation strata selected byte-identical audio more than once; "
            "review cross-dataset checksum aliases"
        )
    return FrozenSplit(
        validation_items=ordered_items,
        withheld_speakers=tuple(sorted(withheld_speakers)),
        allocations=tuple(allocations),
    )


def _canonical_validation_candidates(
    items: Iterable[InventoryItem],
    strata: tuple[Stratum, ...],
) -> tuple[tuple[InventoryItem, ...], dict[SpeakerKey, int]]:
    """Keep one deterministic source ID for each dataset-qualified audio digest."""

    targeted_datasets = {stratum.dataset for stratum in strata}
    raw_speaker_counts: defaultdict[SpeakerKey, int] = defaultdict(int)
    checksum_groups: defaultdict[tuple[str, str], list[InventoryItem]] = defaultdict(list)
    for item in items:
        if item.media_kind is not MediaKind.SPEECH or item.dataset not in targeted_datasets:
            continue
        if item.speaker_id is None:
            raise ValueError(f"speech item {item.id!r} has no speaker ID")
        raw_speaker_counts[(item.dataset, item.speaker_id)] += 1
        if item.source_checksum is None:
            raise ValueError(f"speech item {item.id!r} has no source checksum")
        checksum_groups[(item.dataset, item.source_checksum)].append(item)

    canonical: list[InventoryItem] = []
    for (dataset, source_checksum), aliases in sorted(checksum_groups.items()):
        signatures = {(item.speaker_id, item.text, item.language, item.gender) for item in aliases}
        if len(signatures) != 1:
            source_ids = ", ".join(sorted(item.source_id for item in aliases))
            raise ValueError(
                f"{dataset}: checksum {source_checksum} has conflicting speech metadata: "
                f"{source_ids}"
            )
        canonical.append(min(aliases, key=lambda item: (item.source_id, item.id)))
    return tuple(canonical), dict(raw_speaker_counts)


def _matches_stratum(item: InventoryItem, stratum: Stratum) -> bool:
    return (stratum.language is None or item.language == stratum.language) and (
        stratum.gender is None or item.gender == stratum.gender
    )


def _select_items(
    selected: list[tuple[SpeakerKey, tuple[InventoryItem, ...]]],
    stratum: Stratum,
    seed: int,
) -> tuple[list[InventoryItem], dict[str, int]]:
    def item_key(item: InventoryItem) -> tuple[bytes, str]:
        return (
            stable_hash(HASH_NAMESPACE, seed, "item", stratum.name, item.source_id),
            item.source_id,
        )

    chosen: list[InventoryItem] = []
    chosen_ids: set[str] = set()
    counts: defaultdict[str, int] = defaultdict(int)
    for (_, speaker_id), items in selected:
        item = min(items, key=item_key)
        chosen.append(item)
        chosen_ids.add(item.id)
        counts[speaker_id] += 1

    remaining = sorted(
        (
            (item_key(item), speaker_id, item)
            for (_, speaker_id), items in selected
            for item in items
            if item.id not in chosen_ids
        ),
        key=lambda value: (value[0], value[1]),
    )
    for _, speaker_id, item in remaining:
        if len(chosen) == stratum.target:
            break
        if counts[speaker_id] == stratum.contribution_cap:
            continue
        chosen.append(item)
        counts[speaker_id] += 1
    if len(chosen) != stratum.target:
        raise ValueError(f"{stratum.name}: selected {len(chosen)} of {stratum.target} items")
    return chosen, dict(counts)


def load_curated_speech(
    workspace: Path,
    edition: str,
    repo_root: Path,
) -> tuple[tuple[InventoryItem, ...], frozenset[SpeakerKey], int]:
    """Resolve the curated speech pool without any previous speech validation split."""

    loaded = load_edition_config(
        edition,
        repo_root=repo_root,
        selection="curated",
    )
    layout = WorkspaceLayout.at(workspace)
    inventory: list[InventoryItem] = []
    for dataset in loaded.config.datasets:
        path = layout.inventories / f"{dataset.id}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"normalized inventory is missing: {path}")
        inventory.extend(read_jsonl(path, InventoryItem))

    exclusions: tuple[ExclusionSpec, ...] = tuple(
        exclusion
        for exclusion in loaded.config.exclusions
        if not exclusion.name.startswith("validation-speech")
    )
    selected = select_inventory(
        inventory,
        selection="curated",
        exclusions=exclusions,
        curations=loaded.config.curations,
    )
    pool = tuple(item for item in selected.training if item.media_kind is MediaKind.SPEECH)
    evaluation_speakers = frozenset(
        (item.dataset, item.speaker_id)
        for item in selected.evaluation
        if item.media_kind is MediaKind.SPEECH and item.speaker_id is not None
    )
    return pool, evaluation_speakers, loaded.config.seed


def render_csv(frozen: FrozenSplit) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("name", "partition", "dataset", "source_id", "speaker_id"))
    for item in frozen.validation_items:
        writer.writerow(("validation-speech-items", "validation", item.dataset, item.source_id, ""))
    for dataset, speaker_id in frozen.withheld_speakers:
        writer.writerow(("validation-speech-speakers", "withheld", dataset, "", speaker_id))
    return output.getvalue()


def render_report(frozen: FrozenSplit, pool: tuple[InventoryItem, ...], seed: int) -> str:
    inventory_digest = hashlib.sha256()
    for item in sorted(pool, key=lambda value: (value.dataset, value.source_id)):
        inventory_digest.update(
            (
                f"{item.dataset}\t{item.source_id}\t{item.speaker_id or ''}\t"
                f"{item.language or ''}\t{item.gender or ''}\t{item.source_checksum or ''}\n"
            ).encode()
        )
    report = {
        "schema_version": 1,
        "edition": "2026",
        "policy": {
            "hash_namespace": HASH_NAMESPACE,
            "seed": seed,
            "maximum_curated_items_per_speaker": MAX_CURATED_ITEMS_PER_SPEAKER,
            "strata": [asdict(stratum) for stratum in STRATA],
            "speaker_order": "sha256(NUL-joined namespace, seed, speaker, stratum, speaker_id)",
            "item_order": "sha256(NUL-joined namespace, seed, item, stratum, source_id)",
            "speaker_disjointness": "withhold every non-validation item from selected speakers",
            "audio_deduplication": "one lexical source ID per dataset-qualified source checksum",
            "audio_alias_contract": "checksum aliases must have identical speech metadata",
        },
        "input": {
            "curated_speech_items": len(pool),
            "curated_speech_identity_sha256": inventory_digest.hexdigest(),
        },
        "result": {
            "validation_source_ids": len(frozen.validation_items),
            "withheld_speakers": len(frozen.withheld_speakers),
            "validation_source_id_sha256": frozen.source_id_checksum,
            "unique_validation_source_checksums": len(frozen.validation_items),
            "validation_audio_identity_sha256": frozen.audio_identity_checksum,
            "allocations": [asdict(allocation) for allocation in frozen.allocations],
        },
    }
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "metadata/editions/2026/validation/speech_exclusions.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "metadata/editions/2026/validation/speech_split.json",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the checked-in files without changing them",
    )
    args = parser.parse_args()

    pool, evaluation_speakers, seed = load_curated_speech(
        args.workspace,
        "2026",
        args.repo_root.resolve(),
    )
    if seed != 2026:
        raise ValueError(f"edition seed must be 2026, got {seed}")
    frozen = freeze_validation_split(
        pool,
        evaluation_speakers=evaluation_speakers,
        seed=seed,
    )
    csv_text = render_csv(frozen)
    report_text = render_report(frozen, pool, seed)

    if args.check:
        mismatches = [
            path
            for path, expected in ((args.output, csv_text), (args.report, report_text))
            if not path.is_file() or path.read_text(encoding="utf-8") != expected
        ]
        if mismatches:
            raise SystemExit("frozen split differs: " + ", ".join(map(str, mismatches)))
    else:
        atomic_write_text(args.output, csv_text)
        atomic_write_text(args.report, report_text)

    print(
        f"validation={len(frozen.validation_items)} "
        f"withheld_speakers={len(frozen.withheld_speakers)} "
        f"sha256={frozen.source_id_checksum}"
    )


if __name__ == "__main__":
    main()
