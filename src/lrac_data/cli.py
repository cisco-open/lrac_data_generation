# Copyright 2026 Cisco Systems, Inc. and its affiliates
# Apache-2.0

"""Command-line interface for one LRAC data preparation lifecycle."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .exporters import export_kaldi
from .models import SelectionMode
from .pipeline import prepare_edition, workspace_status
from .planner import build_plan
from .validation import validate_release


def plan(
    *,
    edition: str,
    selection: SelectionMode,
    check_remote: bool,
    repo_root: Path | None,
) -> int:
    """Resolve a complete edition without downloading or writing anything."""

    report = build_plan(
        edition,
        selection=selection,
        repo_root=repo_root,
        check_remote=check_remote,
    )
    print(f"Edition: {report.edition}")
    print(f"Selection: {report.selection} ({report.policy})")
    print(f"Configuration: {report.config_path}")
    audio = report.target_audio
    channels = audio["channels"]
    print(
        "Target audio: "
        f"{audio['sample_rate_hz']} Hz {audio['sample_format']} {audio['container']}; "
        f"channels speech={channels['speech']}, noise={channels['noise']}, "
        f"rir={channels['rir']}"
    )
    print("Datasets:")
    for dataset in report.datasets:
        kinds = ",".join(dataset.media_kinds)
        print(
            f"  {dataset.id}: {dataset.release} [{kinds}], "
            f"sources={dataset.sources}, curation={dataset.curation_targets}, "
            f"exclusions={dataset.exclusion_targets}"
        )
    if report.public_evaluation is not None:
        print(
            "Public evaluation: "
            f"{report.public_evaluation['id']} at {report.public_evaluation['revision']}"
        )
    print("Stages:")
    for index, stage in enumerate(report.stages, start=1):
        print(f"  {index}. {stage}")
    print("Requirements:")
    for name, available in sorted(report.requirements.items()):
        print(f"  {name}: {'available' if available else 'missing'}")
    if report.remote_checks:
        print("Remote checks:")
        for check in report.remote_checks:
            status_code = check.status if check.status is not None else "error"
            print(f"  {check.dataset}/{check.source}: {status_code} {check.url}")
    if report.unresolved:
        print("Unresolved:", file=sys.stderr)
        for issue in report.unresolved:
            print(f"  {issue}", file=sys.stderr)
        return 1
    print("Plan is complete. No files were written.")
    return 0


def prepare(
    *,
    edition: str,
    workspace: Path,
    output: Path,
    selection: SelectionMode,
    workers: int,
    repo_root: Path | None,
    low_storage: bool = False,
) -> int:
    """Prepare and publish the complete data release."""

    result = prepare_edition(
        edition,
        selection=selection,
        workspace=workspace,
        output=output,
        repo_root=repo_root,
        workers=workers,
        low_storage=low_storage,
        progress=print,
    )
    print(f"Completed run {result.run_id}")
    print(f"Data release: {result.release}")
    print(f"Kaldi views: {result.release / 'kaldi'}")
    for split, path in sorted(result.manifests.items()):
        count_key = {"train": "training", "open-evaluation": "open_evaluation"}.get(split, split)
        print(f"  {split}: {path} ({result.counts[count_key]} records)")
    if result.resumed_datasets:
        print("Resumed inventories: " + ", ".join(result.resumed_datasets))
    return 0


def status(*, workspace: Path) -> int:
    """Show completed and interrupted preparation runs."""

    reports = workspace_status(workspace)
    if not reports:
        print("No preparation runs found.")
        return 0
    for report in reports:
        print(f"{report['run_id']}: {'complete' if report['complete'] else 'incomplete'}")
        for stage, stage_status in sorted(report["stages"].items()):
            print(f"  {stage}: {stage_status}")
    return 0


def validate(*, release: Path, workers: int) -> int:
    """Validate a data release independently of its build workspace."""

    report = validate_release(release, workers=workers, progress=print)
    for error in report.errors:
        print(f"  {error}", file=sys.stderr)
    if not report.ok:
        return 1
    print(f"Validated data release: {sum(report.counts.values())} manifest records")
    return 0


def export_kaldi_command(
    *,
    manifest: Path,
    output: Path,
    workspace: Path | None,
) -> int:
    """Export a canonical JSONL manifest to Kaldi-style sidecars."""

    counts = export_kaldi(manifest, output, workspace=workspace)
    print(json.dumps(counts, sort_keys=True))
    return 0


def _directory_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise argparse.ArgumentTypeError(f"not a directory: {value}")
    return path


def _existing_file(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {value}")
    return path


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"not an integer: {value}") from error
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _selection(value: str) -> SelectionMode:
    try:
        return SelectionMode(value.lower())
    except ValueError as error:
        choices = ", ".join(mode.value for mode in SelectionMode)
        raise argparse.ArgumentTypeError(f"choose one of: {choices}") from error


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lrac-data",
        description="Prepare one reproducible LRAC challenge data release.",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    plan_parser = commands.add_parser(
        "plan",
        help="Resolve a complete edition without downloading or writing anything.",
        description="Resolve a complete edition without downloading or writing anything.",
        allow_abbrev=False,
    )
    plan_parser.add_argument("--edition", required=True, help="Edition name or YAML path")
    plan_parser.add_argument(
        "--selection",
        type=_selection,
        default=SelectionMode.CURATED,
        metavar="{curated,uncurated}",
        help="Training selection (default: curated)",
    )
    plan_parser.add_argument(
        "--check-remote",
        action="store_true",
        help="Perform header-only source URL checks",
    )
    plan_parser.add_argument("--repo-root", type=_directory_path, help=argparse.SUPPRESS)

    prepare_parser = commands.add_parser(
        "prepare",
        help="Prepare and publish the complete data release.",
        description="Prepare and publish the complete data release.",
        allow_abbrev=False,
    )
    prepare_parser.add_argument("--edition", required=True, help="Edition name or YAML path")
    prepare_parser.add_argument("--workspace", required=True, type=_directory_path)
    prepare_parser.add_argument("--output", required=True, type=_directory_path)
    prepare_parser.add_argument(
        "--selection",
        type=_selection,
        default=SelectionMode.CURATED,
        metavar="{curated,uncurated}",
        help="Training selection (default: curated)",
    )
    prepare_parser.add_argument(
        "--workers",
        type=_positive_int,
        default=8,
        help=(
            "Upper bound for concurrent downloads, extraction, conversion, and hashing (default: 8)"
        ),
    )
    prepare_parser.add_argument(
        "--low-storage",
        action="store_true",
        help="Discard downloaded and extracted source caches as soon as they are no longer needed",
    )
    prepare_parser.add_argument("--repo-root", type=_directory_path, help=argparse.SUPPRESS)

    status_parser = commands.add_parser(
        "status",
        help="Show completed and interrupted preparation runs.",
        description="Show completed and interrupted preparation runs.",
        allow_abbrev=False,
    )
    status_parser.add_argument("--workspace", required=True, type=_directory_path)

    validate_parser = commands.add_parser(
        "validate",
        help="Validate a data release independently of its build workspace.",
        description="Validate a data release independently of its build workspace.",
        allow_abbrev=False,
    )
    validate_parser.add_argument("--release", required=True, type=_directory_path)
    validate_parser.add_argument(
        "--workers",
        type=_positive_int,
        default=8,
        help="Concurrent audio checksum workers (default: 8)",
    )

    export_parser = commands.add_parser(
        "export-kaldi",
        help="Export a canonical JSONL manifest to Kaldi-style sidecars.",
        description="Export a canonical JSONL manifest to Kaldi-style sidecars.",
        allow_abbrev=False,
    )
    export_parser.add_argument("--manifest", required=True, type=_existing_file)
    export_parser.add_argument("--output", required=True, type=_directory_path)
    export_parser.add_argument("--workspace", type=_directory_path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line arguments and run one LRAC command."""

    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command is None:
        parser.print_help()
        return 2
    if arguments.command == "plan":
        return plan(
            edition=arguments.edition,
            selection=arguments.selection,
            check_remote=arguments.check_remote,
            repo_root=arguments.repo_root,
        )
    if arguments.command == "prepare":
        return prepare(
            edition=arguments.edition,
            workspace=arguments.workspace,
            output=arguments.output,
            selection=arguments.selection,
            workers=arguments.workers,
            low_storage=arguments.low_storage,
            repo_root=arguments.repo_root,
        )
    if arguments.command == "status":
        return status(workspace=arguments.workspace)
    if arguments.command == "validate":
        return validate(release=arguments.release, workers=arguments.workers)
    if arguments.command == "export-kaldi":
        return export_kaldi_command(
            manifest=arguments.manifest,
            output=arguments.output,
            workspace=arguments.workspace,
        )
    raise AssertionError(f"unhandled command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
