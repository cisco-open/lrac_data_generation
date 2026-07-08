# Network-Free Materialization Benchmark

This harness measures the real LRAC audio materializer and manifest validator
with deterministic, locally generated WAV input. It performs no downloads and
does not alter the production CLI. Each worker count and repetition receives a
fresh output directory with the same hash-sharded layout as production, then
exercises that directory through these phases:

1. `cold_materialization` converts every source into prepared audio.
2. `warm_materialization` reruns the same tasks through journal reuse.
3. `warm_checksum_validation` validates a generated manifest and hashes the
   prepared audio using the requested worker count.
4. `torn_journal_recovery` truncates the final journal record, then resumes.
5. `recovery_checksum_validation` validates and hashes the recovered result.

The recovery phase must regenerate exactly the item named by the torn record.
This keeps the failure representative while bounding recovery work to one file.

Run it from the repository root with the preparation environment installed:

```bash
uv run python -m benchmarks.materialize \
  --workspace /work/scratch/commonsw/lrac_data_generation_v2/benchmark-work \
  --items 1000 \
  --duration-seconds 0.25 \
  --workers 1 2 4 8 \
  --repetitions 3 \
  --require-ffmpeg
```

The command prints the path to `result.json`. All generated sources, prepared
audio, journals, manifests, and default results remain below `--workspace`.
Set `--json-output` to place the JSON report at another explicit path.

Every phase reports wall and CPU time, sampled aggregate process-tree RSS, and
item and audio throughput. The report also includes input/output sizes, backend
and tool versions, checksum-set digests, validation outcomes, journal health,
and the paths regenerated during recovery. A checksum mismatch, validation
error, incomplete repaired journal, or unexpected regenerated path is recorded
in JSON and makes the command exit with status 2.

For a quick server check, use 100 items and one repetition. For worker tuning,
increase to at least 1,000 items and choose the smallest worker count within a
few percent of peak throughput. Compare individual phase timings: cold and
recovery exercise conversion, warm materialization exercises state lookup, and
checksum validation exercises final-run I/O. Recovery traverses every task but
converts only the item whose journal record was torn. Short clips emphasize
process and filesystem overhead; increasing `--duration-seconds` shifts cold
conversion and validation toward codec and sequential I/O throughput.

Do not compare a run using `ffmpeg` with one using the development soundfile
fallback. Use `--require-ffmpeg` for production-representative measurements.
The RSS sampler reads Linux `/proc`; it reports zero on systems without that
filesystem.
