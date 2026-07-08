"""Pinned acquisition and inventory of public LRAC evaluation pairs."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .models import InventoryItem, MediaKind, PublicEvaluationSpec, qualify_id


def fetch_public_evaluation(spec: PublicEvaluationSpec, workspace: Path) -> Path:
    """Materialize a sparse checkout at the exact configured Git revision."""

    workspace_root = workspace.expanduser().resolve()
    downloads_root = (workspace_root / "downloads").resolve()
    _require_descendant(downloads_root, workspace_root, "public evaluation downloads")
    checkout = (downloads_root / spec.id / "repository").resolve()
    _require_descendant(checkout, downloads_root, "public evaluation checkout")
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        _git(
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            spec.repository_url,
            str(checkout),
        )
    if not (checkout / ".git").exists():
        raise RuntimeError(f"public evaluation checkout is incomplete: {checkout}")

    revision_exists = (
        subprocess.run(
            ["git", "cat-file", "-e", f"{spec.revision}^{{commit}}"],
            cwd=checkout,
            capture_output=True,
            text=True,
        ).returncode
        == 0
    )
    if not revision_exists:
        _git("fetch", "--depth", "1", "origin", spec.revision, cwd=checkout)
    _git("sparse-checkout", "init", "--cone", cwd=checkout)
    _git("sparse-checkout", "set", spec.subdirectory.as_posix(), cwd=checkout)
    _git("checkout", "--detach", spec.revision, cwd=checkout)
    root = (checkout / Path(spec.subdirectory.as_posix())).resolve()
    _require_descendant(root, checkout, "public evaluation data")
    if not root.is_dir():
        raise FileNotFoundError(f"public evaluation directory is absent at {spec.revision}: {root}")
    return root


def inventory_public_evaluation(spec: PublicEvaluationSpec, root: Path) -> list[InventoryItem]:
    """Return paired input/reference records for every track and condition."""

    evaluation_root = root.resolve()
    records: list[InventoryItem] = []
    seen: set[str] = set()
    for track in spec.tracks:
        track_root = root / track
        for condition in spec.conditions:
            input_dir = track_root / condition
            reference_dir = (
                track_root / "clean"
                if condition == "clean"
                else track_root / f"reference_{condition}"
            )
            inputs = _wav_by_stem(input_dir, evaluation_root)
            references = _wav_by_stem(reference_dir, evaluation_root)
            if inputs.keys() != references.keys():
                missing_references = sorted(inputs.keys() - references.keys())
                missing_inputs = sorted(references.keys() - inputs.keys())
                raise ValueError(
                    f"unpaired public evaluation data for {track}/{condition}; "
                    f"missing references={missing_references[:5]}, "
                    f"missing inputs={missing_inputs[:5]}"
                )
            for stem in sorted(inputs):
                pair_id = f"{track}:{condition}:{stem}"
                for role, path in (
                    ("input", inputs[stem]),
                    ("reference", references[stem]),
                ):
                    source_id = f"{track}-{condition}-{role}-{stem}"
                    item_id = qualify_id(spec.id, source_id)
                    if item_id in seen:
                        raise ValueError(f"duplicate public evaluation ID: {item_id}")
                    seen.add(item_id)
                    records.append(
                        InventoryItem(
                            id=item_id,
                            dataset=spec.id,
                            source_id=source_id,
                            source_release=spec.revision,
                            media_kind=MediaKind.SPEECH,
                            source_path=path.resolve(),
                            metadata={
                                "track": track,
                                "condition": condition,
                                "role": role,
                                "pair_id": pair_id,
                            },
                        )
                    )
    return sorted(records, key=lambda item: item.id)


def _wav_by_stem(directory: Path, evaluation_root: Path) -> dict[str, Path]:
    root = directory.resolve()
    _require_descendant(root, evaluation_root, "public evaluation condition")
    if not root.is_dir():
        raise FileNotFoundError(f"public evaluation condition is missing: {directory}")
    result: dict[str, Path] = {}
    for candidate in sorted(root.rglob("*.wav")):
        path = candidate.resolve()
        _require_descendant(path, root, "public evaluation WAV")
        if path.stem in result:
            raise ValueError(f"duplicate evaluation utterance ID in {directory}: {path.stem}")
        result[path.stem] = path
    if not result:
        raise ValueError(f"public evaluation condition contains no WAV files: {directory}")
    return result


def _require_descendant(path: Path, root: Path, label: str) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes {root}: {path}") from error
    if relative == Path("."):
        raise ValueError(f"{label} must not be its root: {root}")


def _git(*arguments: str, cwd: Path | None = None) -> None:
    try:
        subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("git is required to fetch public evaluation data") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed: {detail}") from error


__all__ = ["fetch_public_evaluation", "inventory_public_evaluation"]
