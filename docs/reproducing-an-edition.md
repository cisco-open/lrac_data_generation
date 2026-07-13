# Reproducing An Edition

An edition is defined by its edition YAML, referenced dataset YAML files,
curation and exclusion metadata, dependency lock, and repository revision.

## Metadata-Only Check

From a clean checkout, install the locked environment and resolve the plan:

```bash
uv sync --locked --extra prep
uv run lrac-data plan --edition 2026 --selection curated
uv run lrac-data plan --edition 2026 --selection uncurated
```

Planning reads repository metadata only. Without `--check-remote` it performs
no network requests; in either form it must not create or alter a workspace.
The remote check expands every configured multipart or sharded URL and sends
HEAD requests only. Sources whose publishers do not provide archive digests are
called out as integrity notes; preparation records the bytes received with a
SHA-256 digest in the run provenance.

The initial migration plan may also report unpinned canonical inventory counts.
The first complete candidate build records per-dataset/media counts and every
downloaded artifact digest in `run.json`. Those values must be reviewed and
copied into the dataset metadata before declaring the 2026 corpus accepted;
until then the edition branch is a build candidate, not a frozen data release.

## Frozen 2026 Speech Split

The 2026 edition freezes validation membership by exact source ID in
`metadata/editions/2026/validation/speech_exclusions.csv`. The same 1,550
utterances are validation data in curated and uncurated builds:

| Source | Validation items |
| --- | ---: |
| DNS5 | 250 |
| EARS | 250 |
| GLOBE | 250 |
| LibriTTS | 250 |
| VCTK | 250 |
| MLS French | 100 |
| MLS German | 100 |
| MLS Spanish | 100 |

The edition seed is `2026`. Validation candidates are canonicalized by
dataset-qualified source checksum; checksum aliases must have identical speech
metadata, and the lexical source ID represents the audio. The result contains
1,550 byte-unique validation clips.

Candidate speakers have at most 250 curated utterances and are ordered with a
namespaced SHA-256 key. Each speaker contributes at most 25 validation items.
The policy requires at least 10 DNS5 speakers, five speakers in each
known-gender stratum, and four speakers in each MLS language stratum.
Known-gender sources use equal female and male targets. The canonical source-ID
checksum is
`824215a59c6a0e23e73caa7393ce3d12cd0533ff79875c4613c6842b8e9fd49a`;
the dataset-qualified audio-identity checksum is
`ae72456c625cf04caa6b9cd711c41eab85eac1d9661266ea13d6c1810edb7a0c`.

All non-validation utterances belonging to the 125 selected speakers are
classified as `withheld`, not as quality rejections. The validated inventory
contains 18,164 such items in either mode; 3,695 of them would otherwise pass
the curated allowlists and enter curated training. This keeps every training
manifest speaker-disjoint from validation. Exact validation and evaluation
assignments take precedence over speaker withholding.

The checked-in split is generated only when edition policy changes:

```bash
uv run python tools/freeze_validation_speech.py --workspace /data/lrac
uv run python tools/freeze_validation_speech.py --workspace /data/lrac --check
```

The tool reads completed normalized inventories, applies the edition's curated
training policy after removing any previous speech-validation rules, and writes
the frozen CSV plus `speech_split.json`. Normal preparation never samples a
split; it only consumes those reviewed artifacts.

## Complete Build

Choose a workspace outside the source checkout and run one complete selection:

```bash
uv run lrac-data prepare \
  --edition 2026 \
  --selection curated \
  --workspace /data/lrac
```

The workspace separates immutable downloads, extracted sources, prepared
audio, run state, and final manifests. A rerun resumes work only when the saved
fingerprint matches the current configuration, code, and inputs. Changed
fingerprints invalidate the affected stage rather than trusting stale output.
Fingerprints are scoped to their behavior: adapter or shared-inventory changes
invalidate only the affected dataset inventories, while audio-materializer or
FFmpeg changes select a new prepared-audio generation. Unrelated CLI, planner,
or exporter edits remain recorded in run provenance without rebuilding audio.
Each normalized inventory item records its source-media SHA-256, and reuse of a
prepared WAV additionally requires the source release, source digest, target
format, and materializer implementation fingerprint to match.
Large archives derived solely from immutable downloads, such as joined DNS and
FSD50K archives, are removed after successful extraction and regenerated on
demand after an interrupted build.
When configured, public evaluation inputs and references are fetched through a
sparse Git checkout pinned to the exact revision recorded in the edition YAML.
Normalized exclusions name only items that can occur in the configured source
inventories. Provenance from upstream evaluation-only splits remains in the
frozen public-evaluation metadata and is materialized through that checkout; it
is not duplicated as an unresolved training-inventory exclusion.

Run validation after preparation and archive the provenance record with the
manifest:

```bash
uv run lrac-data validate --workspace /data/lrac
uv run lrac-data status --workspace /data/lrac
```

Reproduction is successful when the manifest and prepared-file checksums match
the published edition provenance. Upstream credentials and license acceptance
are environmental prerequisites and are not embedded in the repository.
