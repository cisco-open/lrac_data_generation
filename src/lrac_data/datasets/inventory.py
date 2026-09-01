# Copyright 2026 Cisco Systems, Inc. and its affiliates
# Apache-2.0

"""Inventory construction for the four simple file-tree datasets."""

from __future__ import annotations

from lrac_data.models import InventoryItem, MediaKind

from .base import DatasetAdapter
from .common import require_unique


def build_file_inventory(
    owner: DatasetAdapter,
    root: str,
    pattern: str,
    media_kind: MediaKind,
    source_id_prefix: str = "",
) -> list[InventoryItem]:
    """Build an inventory from one fixed directory and glob."""

    extracted_root = owner.extracted_dir.resolve()
    paths = []
    for path in (extracted_root / root).glob(pattern):
        resolved = path.resolve()
        if path.is_file() and resolved.is_relative_to(extracted_root):
            paths.append(resolved)
    pairs = require_unique(
        ((f"{source_id_prefix}{path.stem}", path) for path in sorted(paths)),
        owner.config.id,
    )
    return [owner.item(source_id, media_kind, path) for source_id, path in pairs]
