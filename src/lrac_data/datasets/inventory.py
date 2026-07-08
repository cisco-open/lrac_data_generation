"""Composable inventory construction for simple file-tree datasets."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from lrac_data.models import InventoryItem, MediaKind

from .base import DatasetAdapter


@dataclass(frozen=True, slots=True)
class FileTreeRule:
    """Select files below one extracted-data root and derive IDs from their stems."""

    root: str | Path
    pattern: str
    media_kind: MediaKind
    source_id_prefix: str = ""


def build_file_inventory(
    owner: DatasetAdapter,
    rules: Iterable[FileTreeRule],
) -> list[InventoryItem]:
    """Build a deterministic inventory from validated file-tree rules."""

    rule_list = tuple(rules)
    if not rule_list:
        raise ValueError("file-tree inventory requires at least one rule")

    extracted_root = owner.extracted_dir.resolve()
    validated: list[tuple[FileTreeRule, Path]] = []
    for rule in rule_list:
        relative_root = Path(rule.root)
        _validate_rule(rule, relative_root)
        search_root = (extracted_root / relative_root).resolve()
        if not search_root.is_relative_to(extracted_root):
            raise ValueError(
                f"Unsafe file-tree root for dataset {owner.config.id!r}: {rule.root!s}"
            )
        validated.append((rule, search_root))

    owner.ensure_expected_files()

    matches: list[tuple[str, int, Path, FileTreeRule]] = []
    for rule_index, (rule, search_root) in enumerate(validated):
        for path in search_root.glob(rule.pattern):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(extracted_root):
                raise ValueError(
                    f"File-tree rule for dataset {owner.config.id!r} matched a file outside "
                    f"the extracted directory: {path}"
                )
            relative_path = resolved.relative_to(extracted_root).as_posix()
            matches.append((relative_path, rule_index, resolved, rule))

    records: list[InventoryItem] = []
    seen: dict[str, Path] = {}
    for _, _, path, rule in sorted(matches):
        source_id = f"{rule.source_id_prefix}{path.stem}"
        previous = seen.get(source_id)
        if previous is not None:
            raise ValueError(
                f"Duplicate source ID {source_id!r} in {owner.config.id}: {previous} and {path}"
            )
        seen[source_id] = path
        records.append(owner.item(source_id, rule.media_kind, path))
    return records


def _validate_rule(rule: FileTreeRule, relative_root: Path) -> None:
    if relative_root.is_absolute() or ".." in relative_root.parts:
        raise ValueError(f"File-tree root must be a safe relative path: {rule.root!s}")
    if not isinstance(rule.pattern, str) or not rule.pattern.strip():
        raise ValueError("File-tree glob pattern must be non-empty")
    pattern_path = Path(rule.pattern)
    if pattern_path.is_absolute() or ".." in pattern_path.parts:
        raise ValueError(f"File-tree glob must be a safe relative pattern: {rule.pattern!r}")
    if not isinstance(rule.media_kind, MediaKind):
        raise ValueError(f"Invalid file-tree media kind: {rule.media_kind!r}")
    if not isinstance(rule.source_id_prefix, str) or any(
        character.isspace() for character in rule.source_id_prefix
    ):
        raise ValueError("File-tree source ID prefix may not contain whitespace")
