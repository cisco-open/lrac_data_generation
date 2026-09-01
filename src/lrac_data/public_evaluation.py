# Copyright 2026 Cisco Systems, Inc. and its affiliates
# Apache-2.0

"""Pinned acquisition and inventory of public LRAC evaluation pairs."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .models import InventoryItem, MediaKind, PublicEvaluationSpec, qualify_id


def fetch_public_evaluation(spec: PublicEvaluationSpec, workspace: Path) -> Path:
    """Materialize a sparse checkout at the exact configured Git revision."""

    workspace_root = workspace.expanduser().resolve()
    downloads_root = (workspace_root / "downloads").resolve()
    _require_descendant(downloads_root, workspace_root, "public evaluation downloads")
    checkout_parent = downloads_root / spec.id
    if checkout_parent.is_symlink():
        raise RuntimeError(f"public evaluation directory must not be a symlink: {checkout_parent}")
    checkout = checkout_parent / "repository"
    _require_descendant(checkout, downloads_root, "public evaluation checkout")
    if checkout.is_symlink():
        raise RuntimeError(f"public evaluation checkout must not be a symlink: {checkout}")
    if checkout.exists() and not _checkout_is_complete(checkout):
        if checkout.is_dir():
            shutil.rmtree(checkout)
        else:
            checkout.unlink()
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        _git(
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            spec.repository_url,
            str(checkout),
        )
    if not _checkout_is_complete(checkout):
        raise RuntimeError(f"public evaluation checkout is incomplete: {checkout}")
    _validate_managed_checkout(checkout, spec.repository_url)

    root = (checkout / Path(spec.subdirectory.as_posix())).resolve()
    _require_descendant(root, checkout.resolve(), "public evaluation data")
    head = _git("rev-parse", "--verify", "HEAD^{commit}", cwd=checkout)
    sparse_paths = _sparse_paths(spec)
    if head == spec.revision and root.is_dir() and all(
        (checkout / path).is_dir() for path in sparse_paths
    ):
        status = _git("status", "--porcelain=v1", "--untracked-files=all", cwd=checkout)
        if not status:
            return root

    _git(
        "fetch",
        "--depth",
        "1",
        "--force",
        "--no-tags",
        "origin",
        spec.revision,
        cwd=checkout,
    )
    fetched_revision = _git("rev-parse", "--verify", "FETCH_HEAD^{commit}", cwd=checkout)
    if fetched_revision != spec.revision:
        raise RuntimeError(
            "public evaluation fetch did not resolve to the configured revision: "
            f"expected {spec.revision}, got {fetched_revision}"
        )

    # Activate sparse mode before moving HEAD so a new checkout never materializes
    # unrelated repository content. The final clean also removes ignored files.
    _git("sparse-checkout", "init", "--cone", cwd=checkout)
    _git("clean", "-ffdx", cwd=checkout)
    _git("checkout", "--detach", "--force", fetched_revision, cwd=checkout)
    _git("sparse-checkout", "set", *sparse_paths, cwd=checkout)
    _git("reset", "--hard", fetched_revision, cwd=checkout)
    _git("clean", "-ffdx", cwd=checkout)

    head = _git("rev-parse", "--verify", "HEAD^{commit}", cwd=checkout)
    if head != spec.revision:
        raise RuntimeError(
            "public evaluation checkout is not at the configured revision: "
            f"expected {spec.revision}, got {head}"
        )
    status = _git("status", "--porcelain=v1", "--untracked-files=all", cwd=checkout)
    if status:
        raise RuntimeError(f"public evaluation checkout is dirty after reset: {checkout}")

    if not root.is_dir():
        raise FileNotFoundError(f"public evaluation directory is absent at {spec.revision}: {root}")
    return root


def inventory_public_evaluation(spec: PublicEvaluationSpec, root: Path) -> list[InventoryItem]:
    """Return paired input/reference records for every track and condition."""

    evaluation_root = (root / Path(spec.open_subdirectory.as_posix())).resolve()
    _require_descendant(evaluation_root, root.resolve(), "open evaluation data")
    records: list[InventoryItem] = []
    seen: set[str] = set()
    for track in spec.tracks:
        track_root = evaluation_root / track
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


def inventory_kaldi_testsets(
    spec: PublicEvaluationSpec,
    root: Path,
) -> dict[str, tuple[InventoryItem, ...]]:
    """Return each input-only blind condition as a separate test partition."""

    evaluation_root = root.resolve()
    test_root = (root / Path(spec.test_subdirectory.as_posix())).resolve()
    _require_descendant(test_root, evaluation_root, "Kaldi test data")
    partitions: dict[str, tuple[InventoryItem, ...]] = {}
    seen: set[str] = set()
    for condition in spec.test_conditions:
        records: list[InventoryItem] = []
        for stem, path in sorted(_wav_by_stem(test_root / condition, evaluation_root).items()):
            source_id = f"test-{condition}-{stem}"
            item_id = qualify_id(spec.id, source_id)
            if item_id in seen:
                raise ValueError(f"duplicate Kaldi test ID: {item_id}")
            seen.add(item_id)
            records.append(
                InventoryItem(
                    id=item_id,
                    dataset=spec.id,
                    source_id=source_id,
                    source_release=spec.revision,
                    media_kind=MediaKind.SPEECH,
                    source_path=path,
                    metadata={"test_condition": condition},
                )
            )
        partition = f"test-{condition.replace('_', '-')}"
        partitions[partition] = tuple(records)
    return partitions


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


def _sparse_paths(spec: PublicEvaluationSpec) -> tuple[str, ...]:
    root = Path(spec.subdirectory.as_posix())
    paths = [root / Path(spec.open_subdirectory.as_posix())]
    test_root = root / Path(spec.test_subdirectory.as_posix())
    paths.extend(test_root / condition for condition in spec.test_conditions)
    return tuple(path.as_posix() for path in paths)


def _checkout_is_complete(checkout: Path) -> bool:
    git_directory = checkout / ".git"
    if git_directory.is_symlink() or not git_directory.is_dir():
        return False

    try:
        repository_root = Path(_git("rev-parse", "--show-toplevel", cwd=checkout)).resolve()
        absolute_git_directory = Path(
            _git("rev-parse", "--absolute-git-dir", cwd=checkout)
        ).resolve()
    except RuntimeError:
        return False
    return (
        repository_root == checkout.resolve() and absolute_git_directory == git_directory.resolve()
    )


def _validate_managed_checkout(checkout: Path, repository_url: str) -> None:
    origin_urls = _git("remote", "get-url", "--all", "origin", cwd=checkout).splitlines()
    if origin_urls != [repository_url]:
        raise RuntimeError(
            "public evaluation checkout origin does not match the configured repository URL"
        )


def _git(*arguments: str, cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
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
    return result.stdout.strip()
