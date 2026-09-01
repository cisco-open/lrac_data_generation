# Manifest Contract

A completed build publishes each manifest as UTF-8 JSON Lines. Every non-empty
line contains one JSON object. Records are sorted by their dataset-qualified
`id`, so the same resolved inputs produce byte-identical manifest content.

## Required Fields

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | string | Stable, dataset-qualified item ID |
| `dataset` | string | Dataset configuration name |
| `source_release` | string | Upstream release identifier |
| `source_id` | string | ID in the upstream dataset |
| `media_kind` | string | `speech`, `noise`, or `rir` |
| `split` | string | Edition-owned split assignment |
| `audio_path` | string | POSIX path relative to the release directory |
| `sample_rate_hz` | integer | Prepared sample rate in hertz |
| `channels` | integer | Prepared channel count |
| `frame_count` | integer | Number of prepared audio frames |
| `checksum` | string | SHA-256 digest of the prepared WAV bytes |
| `metadata` | object | Adapter-specific source metadata |

Speech records may additionally contain `speaker_id`, `text`, `language`, and
`gender`. Fields that do not apply to a media kind are omitted rather than
filled with sentinel values.

Items cut from longer recordings record integer `start_us` and `end_us` bounds
in `metadata.source_segment`. Bounds are expressed in microseconds relative to
the original recording and form a half-open interval: the start is included and
the end is excluded. The published `audio_path` already contains the selected
segment, so consumers do not cut it again.

## Invariants

- IDs are unique across the manifest and include the dataset namespace.
- `audio_path` is relative and cannot escape the release directory.
- Prepared audio is 24 kHz IEEE float32 WAV. Resampled samples are written
  without clipping, normalization, limiting, or gain adjustment. Channel
  handling is configured per media kind as `preserve` or `downmix`; the latter
  produces deterministic equal-weight mono.
- The checked-in 2026 policy preserves every source channel for speech, noise,
  and RIR media.
- Validation and evaluation members never appear in a training manifest.
- Training speech is speaker-disjoint from validation where speaker identities
  are available.
- The complete release directory is published atomically only after every
  manifest, Kaldi view, and audio file validates.
- Every manifest has a matching `kaldi/<partition>/` directory derived from the
  same records. Bundled SCP audio paths are relative to the release root.
- Kaldi `wav.scp` and utterance sidecars contain speech; noise and RIR records
  are kept separately in `noise.scp` and `rirs.scp`.
- Open-evaluation role metadata is exported as paired condition-specific
  `wav.scp` and `reference.scp` files.
- The `test-clean`, `test-realworld`, and `test-simultaneous-talkers` manifests
  are input-only evaluation partitions with independent Kaldi directories.

The fields above define the manifest contract for this release. Consumers
should reject unknown top-level fields; the contents of `metadata` are
adapter-specific.
