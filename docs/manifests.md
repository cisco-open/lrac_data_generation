# Manifest Contract

A completed build publishes a UTF-8 JSON Lines manifest. Each non-empty line is
one JSON object, and records are sorted by their dataset-qualified `id`. The
same resolved inputs always produce byte-identical manifest content.

## Required Fields

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | integer | Manifest schema version, currently `1` |
| `id` | string | Stable, dataset-qualified item ID |
| `dataset` | string | Dataset configuration name |
| `source_release` | string | Upstream release identifier |
| `source_id` | string | ID in the upstream dataset |
| `media_kind` | string | `speech`, `noise`, or `rir` |
| `split` | string | Edition-owned split assignment |
| `audio_path` | string | POSIX path relative to the workspace |
| `sample_rate_hz` | integer | Prepared sample rate in hertz |
| `channels` | integer | Prepared channel count |
| `frame_count` | integer | Number of prepared PCM frames |
| `checksum` | string | SHA-256 digest of the prepared WAV bytes |

Speech records may additionally contain `speaker_id`, `text`, `language`, and
`gender`. Fields that do not apply to a media kind are omitted rather than
filled with sentinel values.

## Invariants

- IDs are unique across the manifest and include the dataset namespace.
- `audio_path` is relative and cannot escape the workspace.
- Prepared audio is mono, 24 kHz, PCM16 WAV unless the edition explicitly
  declares a different target format.
- Multichannel inputs without a channel layout are reduced to an equal-weight
  channel average. This is a 2026 materialization policy, not the representation
  used by the historical 2025 intermediate RIR files.
- Validation and evaluation members never appear in a training manifest.
- The complete manifest set and its `run.json` are published as one directory
  generation only after all records validate.

Consumers should reject unknown schema versions and unknown fields. Adding,
removing, renaming, or changing the meaning of a field requires a new
`schema_version`.
