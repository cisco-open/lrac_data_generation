"""Dataset adapter registry.

Importing this package only registers adapter classes; acquisition starts solely
when a caller invokes :meth:`DatasetAdapter.fetch`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from lrac_data.models import DatasetConfig

from .base import DatasetAdapter
from .dns5 import DNS5Adapter
from .ears import EARSAdapter
from .fma import FMAAdapter
from .fsd50k import FSD50KAdapter
from .globe import GLOBEAdapter
from .libritts import LibriTTSAdapter
from .mls import MLSAdapter
from .motus import MOTUSAdapter
from .vctk import VCTKAdapter
from .wham import WHAMAdapter

AdapterType: TypeAlias = type[DatasetAdapter]

ADAPTERS: dict[str, AdapterType] = {
    "dns5": DNS5Adapter,
    "ears": EARSAdapter,
    "fma": FMAAdapter,
    "fsd50k": FSD50KAdapter,
    "globe": GLOBEAdapter,
    "libritts": LibriTTSAdapter,
    "mls": MLSAdapter,
    "motus": MOTUSAdapter,
    "vctk": VCTKAdapter,
    "wham": WHAMAdapter,
}


def create_adapter(
    config: DatasetConfig,
    repo_root: Path,
    workspace: Path,
    *,
    workers: int = 4,
) -> DatasetAdapter:
    try:
        adapter_type = ADAPTERS[config.adapter]
    except KeyError as exc:
        available = ", ".join(sorted(ADAPTERS))
        raise ValueError(
            f"Unknown dataset adapter {config.adapter!r}; available: {available}"
        ) from exc
    return adapter_type(config, repo_root, workspace, workers=workers)


__all__ = ["ADAPTERS", "DatasetAdapter", "create_adapter"]
