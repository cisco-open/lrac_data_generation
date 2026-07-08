from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_synthetic_materialization_is_deterministic_across_workers(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.materialize",
            "--workspace",
            str(tmp_path / "benchmark"),
            "--items",
            "3",
            "--duration-seconds",
            "0.01",
            "--workers",
            "1",
            "2",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    result_path = Path(completed.stdout.strip())
    report = json.loads(result_path.read_text(encoding="utf-8"))
    assert report["parity"]["ok"] is True
    assert [case["workers"] for case in report["cases"]] == [1, 2]
    assert {case["checksum_set_sha256"] for case in report["cases"]} == {
        report["cases"][0]["checksum_set_sha256"]
    }
    assert report["sources"]["files"] == 3
    assert report["configuration"]["source_sample_rate_hz"] == 48_000
    assert report["configuration"]["target_sample_rate_hz"] == 24_000
    assert report["schema_version"] == 2
    expected_phases = {
        "cold_materialization",
        "warm_materialization",
        "warm_checksum_validation",
        "torn_journal_recovery",
        "recovery_checksum_validation",
    }
    for case in report["cases"]:
        assert set(case["phases"]) == expected_phases
        assert all(phase["wall_seconds"] >= 0 for phase in case["phases"].values())
        assert case["parity"]["warm_validation"] is True
        assert case["parity"]["recovery_validation"] is True
        assert case["parity"]["torn_output_checksum"] is True
        assert case["recovery"]["reprocessed_paths"] == [case["recovery"]["torn_destination"]]
        assert case["recovery"]["journal_records"] == 3
        assert case["recovery"]["journal_files"] >= 1
        assert case["recovery"]["journal_complete"] is True
