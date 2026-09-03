# LRAC 2026 Data Preparation

[![Challenge Website](https://img.shields.io/badge/Challenge-Website-blue)](https://lrac.short.gy/)

This repository builds the data release for the 2026 Low Resource Audio Codec
(LRAC) Challenge. It downloads a fixed set of upstream sources, applies the
checked-in selection and split policies, converts the selected media, and
publishes one self-contained release directory.

The LRAC 2026 data pipeline is a standalone implementation and is not based on
a fork of URGENT 2025.

The historical 2025 shell recipe remains available from the
[`lrac-2025-final` GitHub Release](https://github.com/cisco-open/lrac_data_generation/releases/tag/lrac-2025-final).

## Release Data

| Media | Sources |
| --- | --- |
| Speech | Common Voice 26.0, DNS Challenge 5, EARS, GLOBE, LibriTTS, MLS, OpenSLR93/AISHELL-3, VCTK 0.92 |
| Noise | DNS Challenge 5, FMA Medium, FSD50K, WHAM! |
| Room impulse responses | DNS Challenge 5, MOTUS |

Prepared audio is 24 kHz IEEE float32 WAV. Resampled samples are written without
clipping, normalization, limiting, or gain adjustment. The 2026 release preserves
the source channel count for speech, noise, and room impulse responses.

Two training selections are available:

- `curated` is the release default and applies checked-in quality lists where
  configured.
- `uncurated` retains every eligible source item without applying quality lists.

Both selections use the same fixed validation and evaluation exclusions. The
2026 validation split is frozen from the curated inventories, with 250 clips
per represented speech dataset-language pair and 100 items per noise or RIR
dataset. Speech is speaker-disjoint where speaker identities are available.
Configured public evaluation material is published in a separate
`open-evaluation` manifest.

The `evaluation` partition contains source recordings reserved from training
for constructing the LRAC 2025 evaluation material. It covers only the
corresponding 2025 speech, noise, and RIR sources and does not add the new 2026
languages. The `open-evaluation` partition instead contains the published LRAC
2025 open test signals, paired with their references and organized by track and
condition.

The release also includes the LRAC 2025 Track 1 blind-test audio as three
input-only test sets: `test-clean`, `test-realworld`, and
`test-simultaneous-talkers`. Each has its own manifest and Kaldi data directory.

Upstream releases and terms are declared in
[`configs/datasets`](configs/datasets). Challenge data information is also
available from the [LRAC dataset page](https://lrac.short.gy/datasets).

## Requirements

- Linux
- Python 3.11 or 3.12
- [`uv`](https://docs.astral.sh/uv/)
- `git` on `PATH`
- Info-ZIP `zip` on `PATH`
- Access to the configured upstream datasets and sufficient space for the
  complete build
- A Mozilla Data Collective account and API key for Common Voice

The build workspace and release output must be on the same filesystem, and the
release output must be outside the workspace.

## Installation

```bash
git clone https://github.com/cisco-open/lrac_data_generation.git
cd lrac_data_generation
uv sync --locked --extra prep --no-dev
```

## Data Access

### Common Voice

Common Voice 26.0 is downloaded through Mozilla Data Collective (MDC). MDC
treats every locale as a separate dataset: open every dataset URL listed in
[`configs/datasets/commonvoice_v26.yaml`](configs/datasets/commonvoice_v26.yaml)
and complete its terms acceptance individually. Creating an API key does not
grant access until those acceptances are complete. If you register or accept
terms for a company or another organization, first confirm that you are
authorized to bind that entity; see the
[MDC Data Consumer Terms](https://mozilladatacollective.com/terms/consumers).

Create an [MDC API key](https://mozilladatacollective.com/profile/credentials)
and expose it only to the preparation process. Do not commit the key, place it
in command arguments, or include it in logs. Prefer an approved secret manager,
or enter it interactively without putting the value in shell history:

```bash
read -rsp 'MDC API key: ' MDC_API_KEY
export MDC_API_KEY
printf '\n'
```

`plan` only checks whether the variable is set; it does not send the key or
download data. `prepare` sends the key to MDC to request temporary archive URLs
and publisher checksums.

### Hugging Face (Optional)

GLOBE and MLS use public, pinned Hugging Face sources and work without
an account. Setting `HF_TOKEN` associates those requests with an account and can
reduce anonymous rate-limit interruptions. Create a dedicated read-only or
fine-grained token using the
[Hugging Face token settings](https://huggingface.co/settings/tokens), then
expose it in the same way:

```bash
read -rsp 'Hugging Face read token: ' HF_TOKEN
export HF_TOKEN
printf '\n'
```

The token is sent only to the initial HTTPS `huggingface.co` resolver request;
it is not forwarded to redirected storage hosts. Leaving it unset never blocks
planning or preparation. See the Hugging Face
[rate-limit documentation](https://huggingface.co/docs/hub/en/rate-limits) and
[token security guidance](https://huggingface.co/docs/hub/en/security-tokens).

## Check The Release Plan

Resolve the complete release without creating a workspace or downloading data:

```bash
.venv/bin/lrac-data plan --edition 2026 --selection curated
```

Use `--check-remote` to make one representative header-only check per public
source. It does not test authenticated MDC access or download response bodies;
`prepare` still verifies every configured artifact. Planning never downloads
audio or writes manifests.

## Prepare The Release

> [!WARNING]
> `prepare` can create sustained network and disk I/O, high filesystem metadata
> activity, and a large number of files. Run it on dedicated, adequately sized
> storage rather than a shared or operationally critical filesystem, and monitor
> storage health and free space throughout the build.

```bash
# Change this path to your destination.
LRAC_DATA_ROOT=/data

.venv/bin/lrac-data prepare \
  --edition 2026 \
  --selection curated \
  --workspace "${LRAC_DATA_ROOT}/lrac-work" \
  --output "${LRAC_DATA_ROOT}/releases/LRAC-2026" \
  --workers 8
```

Preparation always produces the complete configured release. Interrupted runs
can resume from the workspace. The output directory becomes visible only after
the complete release validates successfully.

Add `--low-storage` when the workspace cannot retain source caches. Each dataset
is inventoried, selected, and materialized before its downloads and extracted
sources are removed, preventing caches from different datasets from
accumulating. Inventories and prepared audio remain available for resume. A
resumed dataset may still need to fetch its sources again when its materialized
audio is incomplete. Omit the flag when retaining source caches for later runs
matters more than storage.

To build the uncurated selection, reuse the workspace and choose a different
output directory:

```bash
.venv/bin/lrac-data prepare \
  --edition 2026 \
  --selection uncurated \
  --workspace "${LRAC_DATA_ROOT}/lrac-work" \
  --output "${LRAC_DATA_ROOT}/releases/LRAC-2026-uncurated" \
  --workers 8
```

## Release Directory

```text
LRAC-2026/
├── audio/
│   └── <dataset>/<prefix>/*.wav
├── manifests/
│   ├── train.jsonl
│   ├── validation.jsonl
│   ├── evaluation.jsonl
│   ├── open-evaluation.jsonl
│   ├── test-clean.jsonl
│   ├── test-realworld.jsonl
│   └── test-simultaneous-talkers.jsonl
├── kaldi/
│   ├── train/
│   ├── validation/
│   ├── evaluation/
│   ├── open-evaluation/
│   ├── test-clean/
│   ├── test-realworld/
│   └── test-simultaneous-talkers/
├── metadata/
│   ├── datasets.json
│   └── provenance.json
├── licenses/README.md
├── README.md
├── release.json
└── SHA256SUMS
```

All manifest and bundled Kaldi audio paths are relative to the release
directory. `release.json` records partition counts, the audio contract, and the
release fingerprint. `SHA256SUMS` covers every other published file. See the
[manifest contract](docs/manifests.md) for the JSONL fields and invariants.

## Validate

Validate a release independently of its build workspace:

```bash
.venv/bin/lrac-data validate \
  --release "${LRAC_DATA_ROOT}/releases/LRAC-2026" \
  --workers 8
```

Inspect resumable preparation state separately:

```bash
.venv/bin/lrac-data status --workspace "${LRAC_DATA_ROOT}/lrac-work"
```

## Kaldi And ESPnet

`prepare` publishes a matching ESPnet/Kaldi data directory under
`kaldi/<partition>/` for every manifest. These bundled views use paths relative
to the release root so the complete release remains portable. ESPnet recipes
require absolute paths because they run from their own recipe directories; use
`export-kaldi` to create the directories consumed by the baseline.

The command defaults to absolute audio paths. Export every baseline partition
as follows:

```bash
LRAC_RELEASE="${LRAC_DATA_ROOT}/releases/LRAC-2026"
LRAC_ESP_DATA="${LRAC_DATA_ROOT}/lrac-espnet-data"

export_split() {
  .venv/bin/lrac-data export-kaldi \
    --manifest "${LRAC_RELEASE}/manifests/$1.jsonl" \
    --output "${LRAC_ESP_DATA}/$2"
}

export_split train train
export_split validation train_validation
export_split evaluation evaluation
export_split open-evaluation open-evaluation
export_split test-clean test-clean
export_split test-realworld test-realworld
export_split test-simultaneous-talkers test-simultaneous-talkers
```

Both forms include the media-specific `noise.scp` and `rirs.scp` files used by
the LRAC ESPnet baseline when those media occur in the supplied manifest.
`wav.scp` and the utterance sidecars contain speech only, matching the 2025
baseline layout. The optional `spk2gender` is included only when every speech
speaker has a Kaldi-supported `m` or `f` value.
The `evaluation`, `open-evaluation`, and three test views include both `wav.scp`
and `reference.scp`.

## License

The preparation code is licensed under the Apache License 2.0. Each upstream
dataset remains governed by its own license and access terms. The generated
release includes a dataset-specific index in `licenses/README.md`. See
[`LICENSE`](LICENSE) for the repository license.
