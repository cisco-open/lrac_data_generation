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
