# Copyright 2026 Cisco Systems, Inc. and its affiliates
# Apache-2.0

"""Dataset adapter registry.

Importing this package only registers adapter classes; acquisition starts solely
when a caller invokes :meth:`DatasetAdapter.fetch`.
"""

from __future__ import annotations

from pathlib import Path

from lrac_data.models import DatasetConfig

from .base import DatasetAdapter
from .commonvoice_v26 import CommonVoiceV26Adapter
from .dns5 import DNS5Adapter
from .ears import EARSAdapter
from .fma import FMAAdapter
from .fsd50k import FSD50KAdapter
from .globe import GLOBEAdapter
from .libritts import LibriTTSAdapter
from .mls import MLSAdapter
from .motus import MOTUSAdapter
from .openslr93 import OpenSLR93Adapter
from .vctk import VCTKAdapter
from .wham import WHAMAdapter

ADAPTERS: dict[str, type[DatasetAdapter]] = {
    "commonvoice_v26": CommonVoiceV26Adapter,
    "dns5": DNS5Adapter,
    "ears": EARSAdapter,
    "fma": FMAAdapter,
    "fsd50k": FSD50KAdapter,
    "globe": GLOBEAdapter,
    "libritts": LibriTTSAdapter,
    "mls": MLSAdapter,
    "motus": MOTUSAdapter,
    "openslr93": OpenSLR93Adapter,
    "vctk": VCTKAdapter,
    "wham": WHAMAdapter,
}


def create_adapter(
    config: DatasetConfig,
    workspace: Path,
    *,
    workers: int = 4,
) -> DatasetAdapter:
    try:
        adapter_type = ADAPTERS[config.id]
    except KeyError as exc:
        available = ", ".join(sorted(ADAPTERS))
        raise ValueError(f"Unknown dataset {config.id!r}; available: {available}") from exc
    return adapter_type(config, workspace, workers=workers)
