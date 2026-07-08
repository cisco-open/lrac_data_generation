from __future__ import annotations

from pathlib import Path

import pytest

from lrac_data.state import FileIdentity, StateStore, fingerprint, sha256_file


def test_fingerprint_is_independent_of_mapping_order() -> None:
    left = {"edition": 2026, "selection": "curated", "nested": {"b": 2, "a": 1}}
    right = {"nested": {"a": 1, "b": 2}, "selection": "curated", "edition": 2026}

    assert fingerprint(left) == fingerprint(right)


def test_file_identity_round_trips_and_detects_replacement(tmp_path: Path) -> None:
    path = tmp_path / "source.bin"
    path.write_bytes(b"first")
    identity = FileIdentity.from_stat(path.stat())

    assert FileIdentity.from_dict(identity.as_dict()) == identity
    assert identity.matches_stat(path.stat())

    replacement = path.with_suffix(".replacement")
    replacement.write_bytes(b"other")
    replacement.replace(path)

    assert not identity.matches_stat(path.stat())


def test_state_resumes_only_matching_complete_and_unchanged_output(tmp_path: Path) -> None:
    output = tmp_path / "prepared.wav"
    output.write_bytes(b"complete output")
    store = StateStore(tmp_path / "state")
    stage_fingerprint = fingerprint({"dataset": "fixture", "target_rate": 24_000})

    running = store.mark_running("materialize/fixture", stage_fingerprint)
    assert running.status == "running"
    assert not store.is_complete("materialize/fixture", stage_fingerprint)

    completed = store.mark_complete(
        "materialize/fixture",
        stage_fingerprint,
        [output],
        started_at=running.started_at,
    )
    assert completed.status == "complete"
    assert completed.finished_at is not None
    assert store.is_complete("materialize/fixture", stage_fingerprint)
    assert not store.is_complete("materialize/fixture", fingerprint({"changed": True}))

    output.write_bytes(b"changed output")
    assert not store.is_complete("materialize/fixture", stage_fingerprint)


def test_failed_or_interrupted_stage_is_not_resumable(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state")
    stage_fingerprint = fingerprint({"stage": "inventory"})

    running = store.mark_running("inventory", stage_fingerprint)
    assert not store.is_complete("inventory", stage_fingerprint)

    failed = store.mark_failed(
        "inventory", stage_fingerprint, "fixture failure", started_at=running.started_at
    )
    assert failed.status == "failed"
    assert failed.error == "fixture failure"
    assert not store.is_complete("inventory", stage_fingerprint)


def test_state_can_verify_only_requested_outputs(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.jsonl"
    archive = tmp_path / "archive.tar"
    inventory.write_text("inventory", encoding="utf-8")
    archive.write_text("archive", encoding="utf-8")
    store = StateStore(tmp_path / "state")
    stage_fingerprint = fingerprint({"stage": "inventory"})
    store.mark_complete("inventory", stage_fingerprint, [inventory, archive])

    archive.write_text("changed archive", encoding="utf-8")

    assert store.is_complete("inventory", stage_fingerprint, verify_paths=[inventory])
    assert not store.is_complete("inventory", stage_fingerprint)


def test_state_reuses_known_output_digests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    known = tmp_path / "known.bin"
    unknown = tmp_path / "unknown.bin"
    known.write_bytes(b"known")
    unknown.write_bytes(b"unknown")
    known_digest = sha256_file(known)
    calls: list[Path] = []

    def tracking_digest(path: Path) -> str:
        calls.append(path)
        return sha256_file(path)

    monkeypatch.setattr("lrac_data.state.sha256_file", tracking_digest)
    store = StateStore(tmp_path / "state")
    completed = store.mark_complete(
        "inventory",
        fingerprint({"stage": "inventory"}),
        [known, unknown],
        known_digests={known: known_digest},
    )

    assert calls == [unknown]
    assert completed.outputs is not None
    assert completed.outputs[str(known.resolve())] == known_digest
