# LRAC Data Generation

[![CI](https://github.com/cisco-open/lrac_data_generation/actions/workflows/ci.yml/badge.svg)](https://github.com/cisco-open/lrac_data_generation/actions/workflows/ci.yml)
[![Challenge Website](https://img.shields.io/badge/Challenge-Website-blue)](https://lrac.short.gy/)

This repository contains the reproducible data preparation pipeline for the Low
Resource Audio Codec (LRAC) Challenge. The current pipeline describes the 2026
edition. The original 2025 shell recipe remains available at the
[`lrac-2025-final`](https://github.com/cisco-open/lrac_data_generation/tree/lrac-2025-final)
tag and on the `archive/2025` branch.

The pipeline has two deliberate operating modes:

- `plan` resolves and validates local configuration without downloading,
  extracting, converting, or writing data.
- `prepare` materializes the complete configured edition. It does not support
  partial corpora or a storage budget.

## Requirements

- Linux
- Python 3.11 or 3.12
- [`uv`](https://docs.astral.sh/uv/)
- `ffmpeg` on `PATH`
- Info-ZIP `zip` on `PATH` (required to assemble the FSD50K split archive)
- Access to every corpus configured for the selected edition

Install the CLI and preparation dependencies:

```bash
git clone https://github.com/cisco-open/lrac_data_generation.git
cd lrac_data_generation
uv sync --extra prep
```

For development, include the test and static-analysis tools:

```bash
uv sync --extra prep --group dev
```

## Plan Without Audio

Planning is metadata-only and does not create a workspace:

```bash
uv run lrac-data plan --edition 2026 --selection curated
uv run lrac-data plan --edition 2026 --selection uncurated
```

Add `--check-remote` to issue header-only availability checks for configured
URLs, including every configured shard of templated sources. Response bodies
are never downloaded by `plan`. The report also identifies publishers that do
not provide an archive checksum; successful preparation records the received
artifact SHA-256 in provenance.

`curated` applies the edition's quality allowlists to training candidates.
`uncurated` skips those quality allowlists. Both modes use the same frozen,
exact validation and evaluation IDs. Non-validation speech from validation
speakers is withheld in both modes, preserving speaker disjointness without
counting those items as quality failures.

## Prepare An Edition

Preparation always processes all datasets in the edition:

```bash
uv run lrac-data prepare \
  --edition 2026 \
  --selection curated \
  --workspace /data/lrac \
  --workers 8
```

`--workers` bounds concurrent downloads, extraction jobs, checksum work, and
audio conversion. The default is four; increase it only after measuring the
target storage and source hosts.

Use `--selection uncurated` to materialize all eligible training items. Stable
audio IDs allow curated and uncurated manifests to share already prepared
audio in the same workspace. Reuse is accepted only when the source digest,
source release, target format, and materializer implementation still match.

Inspect or validate a run:

```bash
uv run lrac-data status --workspace /data/lrac
uv run lrac-data validate --workspace /data/lrac --workers 8
```

Export a completed JSONL manifest for a Kaldi-compatible baseline:

```bash
uv run lrac-data export-kaldi \
  --manifest /data/lrac/manifests/2026/curated/train.jsonl \
  --output /data/lrac/kaldi/curated
```

Final prepared audio is mono, 24 kHz, PCM16 WAV. A successful run records its
resolved configuration, source and output digests, tool versions, and counts.
Interrupted runs retain verified downloads and sharded audio checkpoints for
resumption. Suspect extractions are rebuilt, and final manifests are published
only after validation succeeds.

The initial 2026 configuration also materializes the pinned LRAC 2025 public
open-test set as a compatibility set. Its repository revision and directory are
declared explicitly in the edition YAML; a newly released second-edition public
set should be introduced through a new immutable edition configuration.

The checked-in 2026 policy is a migration candidate until the first complete
build is reviewed. `plan` reports the remaining unpinned upstream checksums and
inventory-count baselines. A complete candidate run records both in `run.json`
so they can be frozen in dataset metadata before the edition is accepted.

## Documentation

- [Manifest contract](docs/manifests.md)
- [Adding a dataset](docs/adding-a-dataset.md)
- [Reproducing an edition](docs/reproducing-an-edition.md)
- [2025 historical recipe](docs/history/2025.md)

The datasets and their upstream licenses are listed in the edition and dataset
configuration. Challenge data information is also available from the
[LRAC dataset page](https://lrac.short.gy/datasets).

## License

The pipeline is licensed under the Apache License 2.0. Individual datasets
remain subject to their upstream licenses. See [LICENSE](LICENSE).
