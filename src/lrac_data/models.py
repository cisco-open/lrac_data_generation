# Copyright 2026 Cisco Systems, Inc. and its affiliates
# Apache-2.0

"""Typed contracts shared by LRAC configuration and pipeline stages.

The models in this module are deliberately free of file-system behavior.  Dataset
adapters describe a complete :class:`InventoryItem` inventory, edition policy
partitions that inventory, and materialization produces :class:`ManifestItem`
records.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, TypedDict

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

Identifier = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, pattern=r"^\S+$"),
]


def _safe_path_segment(value: str) -> str:
    if value in {".", ".."}:
        raise ValueError("path segment may not be '.' or '..'")
    return value


PathSegment = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, pattern=r"^[^\s/\\]+$"),
    AfterValidator(_safe_path_segment),
]
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
PositiveInt = Annotated[int, Field(gt=0)]


class SelectionMode(StrEnum):
    """Public inventory-selection modes."""

    CURATED = "curated"
    UNCURATED = "uncurated"

    @property
    def policy_name(self) -> str:
        """Return the internal policy name used in provenance records."""

        return "curated" if self is self.CURATED else "all-eligible"


class MediaKind(StrEnum):
    """Kinds of source media used by the challenge mixer."""

    SPEECH = "speech"
    NOISE = "noise"
    RIR = "rir"


class ChannelMode(StrEnum):
    """Supported channel handling for prepared audio."""

    PRESERVE = "preserve"
    DOWNMIX = "downmix"


class Split(StrEnum):
    """A materialized manifest partition."""

    TRAIN = "train"
    VALIDATION = "validation"
    EVALUATION = "evaluation"


class ExclusionPartition(StrEnum):
    """How an edition exclusion partitions source inventory."""

    VALIDATION = "validation"
    EVALUATION = "evaluation"
    WITHHELD = "withheld"


class ContractModel(BaseModel):
    """Strict, immutable base for externally persisted contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class ChannelPolicy(ContractModel):
    """Channel handling selected independently for each media kind."""

    speech: ChannelMode
    noise: ChannelMode
    rir: ChannelMode

    def for_media_kind(self, kind: MediaKind) -> ChannelMode:
        """Return the channel mode configured for one media kind."""

        modes = {
            MediaKind.SPEECH: self.speech,
            MediaKind.NOISE: self.noise,
            MediaKind.RIR: self.rir,
        }
        return modes[kind]


class AudioFormat(ContractModel):
    """Fixed audio representation published for LRAC 2026."""

    sample_rate_hz: Literal[24_000] = 24_000
    channels: ChannelPolicy = ChannelPolicy(
        speech=ChannelMode.PRESERVE,
        noise=ChannelMode.PRESERVE,
        rir=ChannelMode.PRESERVE,
    )
    sample_format: Literal["float32"] = "float32"
    container: Literal["wav"] = "wav"


class SourceSegment(ContractModel):
    """Half-open time range within an original source recording."""

    start_us: NonNegativeInt
    end_us: PositiveInt

    @model_validator(mode="after")
    def validate_bounds(self) -> SourceSegment:
        if self.end_us <= self.start_us:
            raise ValueError("source segment end_us must be greater than start_us")
        return self


class SourceSpec(ContractModel):
    """A downloadable or already-local dataset source.

    ``checksum`` accepts either a bare digest or an algorithm-prefixed value such
    as ``sha256:abc...``.  Verification is performed by the download layer.
    """

    name: Identifier
    url: NonEmptyText | None = None
    path: Path | None = None
    filename: PathSegment | None = None
    checksum: NonEmptyText | None = None
    artifact_checksums: dict[str, Sha256] = Field(default_factory=dict)

    @field_validator("artifact_checksums")
    @classmethod
    def validate_artifact_checksums(cls, checksums: dict[str, str]) -> dict[str, str]:
        for name in checksums:
            path = PurePosixPath(name)
            if (
                not name
                or "\\" in name
                or path.is_absolute()
                or ".." in path.parts
                or path == PurePosixPath(".")
            ):
                raise ValueError(f"artifact checksum path must be safe and relative: {name!r}")
        return checksums

    @model_validator(mode="after")
    def require_location(self) -> SourceSpec:
        if self.url is None and self.path is None:
            raise ValueError("a source must define at least one of 'url' or 'path'")
        return self


class DatasetConfig(ContractModel):
    """Resolved configuration for one normalized dataset adapter."""

    id: PathSegment
    release: NonEmptyText
    license: NonEmptyText
    media_kinds: tuple[MediaKind, ...]
    sources: tuple[SourceSpec, ...] = ()
    expected_inventory: dict[MediaKind, PositiveInt] = Field(default_factory=dict)

    @field_validator("media_kinds")
    @classmethod
    def validate_media_kinds(cls, media_kinds: tuple[MediaKind, ...]) -> tuple[MediaKind, ...]:
        if not media_kinds:
            raise ValueError("a dataset must declare at least one media kind")
        if len(media_kinds) != len(set(media_kinds)):
            raise ValueError("dataset media kinds must be unique")
        return media_kinds

    @field_validator("sources")
    @classmethod
    def validate_source_names(cls, sources: tuple[SourceSpec, ...]) -> tuple[SourceSpec, ...]:
        names = [source.name for source in sources]
        if len(names) != len(set(names)):
            raise ValueError("dataset source names must be unique")
        return sources

    @model_validator(mode="after")
    def validate_expected_inventory(self) -> DatasetConfig:
        unexpected = set(self.expected_inventory) - set(self.media_kinds)
        if unexpected:
            raise ValueError(
                "expected inventory references undeclared media kinds: "
                + ", ".join(sorted(kind.value for kind in unexpected))
            )
        return self


class PublicEvaluationSpec(ContractModel):
    """Pinned evaluation checkout and its LRAC directory layout."""

    id: PathSegment = "lrac-open-evaluation"
    repository_url: NonEmptyText
    revision: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
    subdirectory: PurePosixPath
    open_subdirectory: PurePosixPath
    test_subdirectory: PurePosixPath
    tracks: tuple[PathSegment, ...] = ("track_1", "track_2")
    conditions: tuple[PathSegment, ...] = ("clean", "noisy", "reverb")
    test_conditions: tuple[PathSegment, ...] = (
        "clean",
        "realworld",
        "simultaneous_talkers",
    )

    @field_validator("subdirectory", "open_subdirectory", "test_subdirectory")
    @classmethod
    def validate_subdirectory(cls, value: PurePosixPath) -> PurePosixPath:
        if value.is_absolute() or ".." in value.parts or value == PurePosixPath("."):
            raise ValueError("evaluation subdirectories must be safe relative paths")
        return value

    @field_validator("tracks", "conditions", "test_conditions")
    @classmethod
    def validate_groups(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values or len(values) != len(set(values)):
            raise ValueError("evaluation groups must be nonempty and unique")
        return values


class ExclusionSpec(ContractModel):
    """Frozen validation, evaluation, or training-withholding policy.

    Validation and evaluation use exact source IDs. Withheld exclusions use
    either exact source IDs (for example, other segments from a validation
    recording) or speaker IDs (to keep every non-validation utterance from a
    selected validation speaker out of training). Each exclusion targets exactly
    one identifier type. Every identifier is scoped to its source dataset.
    """

    name: Identifier
    partition: ExclusionPartition
    dataset: PathSegment
    source_ids: tuple[Identifier, ...] = ()
    speaker_ids: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def validate_exclusion(self) -> ExclusionSpec:
        if bool(self.source_ids) == bool(self.speaker_ids):
            raise ValueError("an exclusion must contain exactly one of source IDs or speaker IDs")
        if self.partition is not ExclusionPartition.WITHHELD and self.speaker_ids:
            raise ValueError(f"{self.partition.value} exclusions must use exact source IDs")
        if len(self.source_ids) != len(set(self.source_ids)):
            raise ValueError("source IDs in an exclusion must be unique")
        if len(self.speaker_ids) != len(set(self.speaker_ids)):
            raise ValueError("speaker IDs in an exclusion must be unique")
        return self


class CurationSpec(ContractModel):
    """Quality allowlist for one dataset and media kind."""

    dataset: PathSegment
    media_kind: MediaKind
    source_ids: tuple[Identifier, ...]

    @field_validator("source_ids")
    @classmethod
    def validate_source_ids(cls, source_ids: tuple[str, ...]) -> tuple[str, ...]:
        if not source_ids:
            raise ValueError("a curation rule must contain at least one source ID")
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source IDs in a curation rule must be unique")
        return source_ids


class EditionConfig(ContractModel):
    """Fully resolved configuration for one reproducible challenge edition."""

    edition: PathSegment
    audio: AudioFormat = Field(default_factory=AudioFormat)
    datasets: tuple[DatasetConfig, ...]
    exclusions: tuple[ExclusionSpec, ...] = ()
    curations: tuple[CurationSpec, ...] = ()
    public_evaluation: PublicEvaluationSpec | None = None
    seed: int = 42

    @field_validator("datasets")
    @classmethod
    def validate_datasets(cls, datasets: tuple[DatasetConfig, ...]) -> tuple[DatasetConfig, ...]:
        if not datasets:
            raise ValueError("an edition must configure at least one dataset")
        ids = [dataset.id for dataset in datasets]
        if len(ids) != len(set(ids)):
            raise ValueError("edition dataset IDs must be unique")
        return datasets

    @field_validator("exclusions")
    @classmethod
    def validate_exclusion_names(
        cls, exclusions: tuple[ExclusionSpec, ...]
    ) -> tuple[ExclusionSpec, ...]:
        identities = [
            (exclusion.name, exclusion.partition, exclusion.dataset) for exclusion in exclusions
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("edition exclusion name/partition/dataset combinations must be unique")
        return exclusions

    @field_validator("curations")
    @classmethod
    def validate_curation_scopes(
        cls, curations: tuple[CurationSpec, ...]
    ) -> tuple[CurationSpec, ...]:
        scopes = [(curation.dataset, curation.media_kind) for curation in curations]
        if len(scopes) != len(set(scopes)):
            raise ValueError("edition curation dataset/media-kind scopes must be unique")
        return curations

    @model_validator(mode="after")
    def validate_policy_datasets(self) -> EditionConfig:
        datasets_by_id = {dataset.id: dataset for dataset in self.datasets}
        dataset_ids = set(datasets_by_id)
        referenced = {exclusion.dataset for exclusion in self.exclusions}
        referenced.update(curation.dataset for curation in self.curations)
        unknown = sorted(referenced - dataset_ids)
        if unknown:
            raise ValueError("edition policy references unknown datasets: " + ", ".join(unknown))
        invalid_kinds = sorted(
            f"{curation.dataset}/{curation.media_kind.value}"
            for curation in self.curations
            if curation.media_kind not in datasets_by_id[curation.dataset].media_kinds
        )
        if invalid_kinds:
            raise ValueError(
                "edition curation references media kinds not provided by its dataset: "
                + ", ".join(invalid_kinds)
            )
        return self


class InventoryItem(ContractModel):
    """One normalized source item before edition selection or materialization."""

    id: Identifier
    dataset: PathSegment
    source_id: Identifier
    source_release: NonEmptyText
    media_kind: MediaKind
    source_path: Path
    source_checksum: Sha256 | None = None
    speaker_id: Identifier | None = None
    text: str | None = None
    language: Identifier | None = None
    gender: Identifier | None = None
    source_segment: SourceSegment | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_qualified_id(self) -> InventoryItem:
        expected = qualify_id(self.dataset, self.source_id)
        if self.id != expected:
            raise ValueError(
                f"inventory ID must be dataset-qualified as {expected!r}, got {self.id!r}"
            )
        if self.media_kind is not MediaKind.SPEECH and any(
            value is not None for value in (self.speaker_id, self.text, self.language, self.gender)
        ):
            raise ValueError("speech metadata may only be set on speech inventory items")
        if "source_segment" in self.metadata:
            raise ValueError("inventory metadata key 'source_segment' is reserved")
        return self


class ManifestItem(ContractModel):
    """One deterministic, materialized JSONL manifest record."""

    id: Identifier
    dataset: PathSegment
    media_kind: MediaKind
    audio_path: PurePosixPath
    source_release: NonEmptyText
    source_id: Identifier
    split: Split
    sample_rate_hz: PositiveInt
    channels: PositiveInt
    frame_count: PositiveInt
    checksum: Sha256
    speaker_id: Identifier | None = None
    text: str | None = None
    language: Identifier | None = None
    gender: Identifier | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("audio_path")
    @classmethod
    def validate_relative_audio_path(cls, audio_path: PurePosixPath) -> PurePosixPath:
        if audio_path.is_absolute() or ".." in audio_path.parts:
            raise ValueError("manifest audio paths must be relative and may not contain '..'")
        if audio_path == PurePosixPath("."):
            raise ValueError("manifest audio path may not be empty")
        return audio_path

    @model_validator(mode="after")
    def validate_record(self) -> ManifestItem:
        expected = qualify_id(self.dataset, self.source_id)
        if self.id != expected:
            raise ValueError(
                f"manifest ID must be dataset-qualified as {expected!r}, got {self.id!r}"
            )
        if self.media_kind is not MediaKind.SPEECH and any(
            value is not None for value in (self.speaker_id, self.text, self.language, self.gender)
        ):
            raise ValueError("speech metadata may only be set on speech manifest records")
        return self

    @classmethod
    def from_inventory(
        cls,
        item: InventoryItem,
        *,
        audio_path: str | PurePosixPath,
        split: Split,
        sample_rate_hz: int,
        channels: int,
        frame_count: int,
        checksum: str,
    ) -> ManifestItem:
        """Build a final record while retaining normalized source metadata."""

        metadata = dict(item.metadata)
        if item.source_segment is not None:
            metadata["source_segment"] = item.source_segment.model_dump(mode="json")
        return cls(
            id=item.id,
            dataset=item.dataset,
            media_kind=item.media_kind,
            audio_path=PurePosixPath(audio_path),
            source_release=item.source_release,
            source_id=item.source_id,
            split=split,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            frame_count=frame_count,
            checksum=checksum,
            speaker_id=item.speaker_id,
            text=item.text,
            language=item.language,
            gender=item.gender,
            metadata=metadata,
        )


class PreparationRunMetadata(TypedDict):
    """Internal run record written verbatim to ``run.json``."""

    run_id: str
    edition: str
    selection: str
    selection_policy: str
    config_path: str
    config_fingerprint: str
    implementation_fingerprint: str
    dependency_lock_digest: str | None
    input_fingerprint: str
    run_fingerprint: str
    counts: dict[str, int | None]
    inventory_digests: dict[str, str]
    selected_source_digest: str
    inventory_counts: dict[str, dict[str, int]]
    source_artifacts: dict[str, list[dict[str, str]]]
    manifests: dict[str, dict[str, str]]
    timings_seconds: dict[str, Any]
    environment: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Complete inventory partition produced by edition selection policy."""

    training: tuple[InventoryItem, ...]
    validation: tuple[InventoryItem, ...]
    evaluation: tuple[InventoryItem, ...]
    withheld: tuple[InventoryItem, ...] = ()
    quality_rejected: tuple[InventoryItem, ...] = ()

    @property
    def counts(self) -> dict[str, int]:
        """Return stable partition counts for logs and provenance."""

        return {
            "training": len(self.training),
            "validation": len(self.validation),
            "evaluation": len(self.evaluation),
            "withheld": len(self.withheld),
            "quality_rejected": len(self.quality_rejected),
        }


def qualify_id(dataset: str, source_id: str) -> str:
    """Return the canonical stable ID for a dataset-local source ID."""

    dataset = dataset.strip()
    source_id = source_id.strip()
    if not dataset or not source_id:
        raise ValueError("dataset and source ID must be non-empty")
    if any(character.isspace() for character in dataset + source_id):
        raise ValueError("dataset and source ID may not contain whitespace")
    return f"{dataset}:{source_id}"
