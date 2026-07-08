"""Reproducible data preparation for the LRAC Challenge."""

from lrac_data.config import (
    ConfigError,
    LoadedEdition,
    load_dataset_config,
    load_edition_config,
)
from lrac_data.manifests import (
    ManifestError,
    read_jsonl,
    read_manifest,
    write_jsonl,
    write_manifest,
)
from lrac_data.models import (
    AudioFormat,
    CurationAction,
    CurationSpec,
    DatasetConfig,
    EditionConfig,
    ExclusionSpec,
    InventoryItem,
    ManifestItem,
    MediaKind,
    PublicEvaluationSpec,
    SelectionMode,
    SelectionResult,
    SourceSpec,
    Split,
    qualify_id,
)
from lrac_data.selection import SelectionError, select_inventory

__version__ = "2.0.0.dev0"

__all__ = [
    "AudioFormat",
    "ConfigError",
    "CurationAction",
    "CurationSpec",
    "DatasetConfig",
    "EditionConfig",
    "ExclusionSpec",
    "InventoryItem",
    "LoadedEdition",
    "ManifestError",
    "ManifestItem",
    "MediaKind",
    "PublicEvaluationSpec",
    "SelectionError",
    "SelectionMode",
    "SelectionResult",
    "SourceSpec",
    "Split",
    "load_dataset_config",
    "load_edition_config",
    "qualify_id",
    "read_jsonl",
    "read_manifest",
    "select_inventory",
    "write_jsonl",
    "write_manifest",
]
