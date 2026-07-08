"""Command-line interface for LRAC data planning and preparation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from .exporters import export_kaldi
from .models import SelectionMode
from .pipeline import prepare_edition, workspace_status
from .planner import build_plan
from .validation import validate_manifests, validate_published_run

app = typer.Typer(
    name="lrac-data",
    help="Plan, prepare, validate, and export LRAC challenge data.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


@app.command()
def plan(
    edition: Annotated[str, typer.Option("--edition", help="Edition name or YAML path")],
    selection: Annotated[
        SelectionMode,
        typer.Option("--selection", case_sensitive=False),
    ] = SelectionMode.CURATED,
    check_remote: Annotated[
        bool,
        typer.Option("--check-remote", help="Perform header-only source URL checks"),
    ] = False,
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", file_okay=False, resolve_path=True, hidden=True),
    ] = None,
) -> None:
    """Resolve a complete edition without downloading or writing anything."""

    report = build_plan(
        edition,
        selection=selection,
        repo_root=repo_root,
        check_remote=check_remote,
    )
    typer.echo(f"Edition: {report.edition}")
    typer.echo(f"Selection: {report.selection} ({report.policy})")
    typer.echo(f"Configuration: {report.config_path}")
    audio = report.target_audio
    typer.echo(
        "Target audio: "
        f"{audio['sample_rate_hz']} Hz, {audio['channels']} channel, "
        f"{audio['sample_format']} {audio['container']}"
    )
    typer.echo("Datasets:")
    for dataset in report.datasets:
        kinds = ",".join(dataset.media_kinds)
        typer.echo(
            f"  {dataset.id}: {dataset.adapter} {dataset.release} [{kinds}], "
            f"sources={dataset.sources}, curation={dataset.curation_targets}, "
            f"exclusions={dataset.exclusion_targets}"
        )
    if report.public_evaluation is not None:
        typer.echo(
            "Public evaluation: "
            f"{report.public_evaluation['id']} at "
            f"{report.public_evaluation['revision']}"
        )
    typer.echo("Stages:")
    for index, stage in enumerate(report.stages, start=1):
        typer.echo(f"  {index}. {stage}")
    typer.echo("Requirements:")
    for name, available in sorted(report.requirements.items()):
        typer.echo(f"  {name}: {'available' if available else 'missing'}")
    if report.remote_checks:
        typer.echo("Remote checks:")
        for check in report.remote_checks:
            status = check.status if check.status is not None else "error"
            typer.echo(f"  {check.dataset}/{check.source}: {status} {check.url}")
    if report.integrity_warnings:
        typer.echo("Integrity notes:")
        for warning in report.integrity_warnings:
            typer.echo(f"  {warning}")
    if report.unresolved:
        typer.echo("Unresolved:", err=True)
        for issue in report.unresolved:
            typer.echo(f"  {issue}", err=True)
        raise typer.Exit(code=1)
    typer.echo("Plan is complete. No files were written.")


@app.command()
def prepare(
    edition: Annotated[str, typer.Option("--edition", help="Edition name or YAML path")],
    workspace: Annotated[
        Path,
        typer.Option("--workspace", file_okay=False, resolve_path=True),
    ],
    selection: Annotated[
        SelectionMode,
        typer.Option("--selection", case_sensitive=False),
    ] = SelectionMode.CURATED,
    workers: Annotated[
        int,
        typer.Option(
            "--workers",
            min=1,
            help="Upper bound for concurrent downloads, hashing, extraction, and conversion",
        ),
    ] = 4,
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", file_okay=False, resolve_path=True, hidden=True),
    ] = None,
) -> None:
    """Prepare every source and publish a complete edition manifest set."""

    result = prepare_edition(
        edition,
        selection=selection,
        workspace=workspace,
        repo_root=repo_root,
        workers=workers,
        progress=typer.echo,
    )
    typer.echo(f"Completed run {result.run_id}")
    for split, path in sorted(result.manifests.items()):
        count_key = {
            "train": "training",
            "open-evaluation": "open_evaluation",
        }.get(split, split)
        typer.echo(f"  {split}: {path} ({result.counts[count_key]} records)")
    if result.resumed_datasets:
        typer.echo("Resumed inventories: " + ", ".join(result.resumed_datasets))


@app.command()
def status(
    workspace: Annotated[
        Path,
        typer.Option("--workspace", file_okay=False, resolve_path=True),
    ],
) -> None:
    """Show completed and interrupted preparation runs."""

    reports = workspace_status(workspace)
    if not reports:
        typer.echo("No preparation runs found.")
        return
    for report in reports:
        typer.echo(f"{report['run_id']}: {'complete' if report['complete'] else 'incomplete'}")
        for stage, stage_status in sorted(report["stages"].items()):
            typer.echo(f"  {stage}: {stage_status}")


@app.command("validate")
def validate_command(
    workspace: Annotated[
        Path,
        typer.Option("--workspace", file_okay=False, resolve_path=True),
    ],
    verify_checksums: Annotated[
        bool,
        typer.Option("--verify-checksums/--skip-checksums"),
    ] = True,
    workers: Annotated[
        int,
        typer.Option("--workers", min=1, help="Concurrent audio validation workers"),
    ] = 4,
) -> None:
    """Validate each published edition/selection independently."""

    workspace_root = workspace.expanduser().resolve()
    root = workspace_root / "manifests"
    groups = sorted(
        path
        for path in root.glob("*/*")
        if path.is_dir() and not path.name.startswith(".") and not path.parent.name.startswith(".")
    )
    if not groups:
        typer.echo(f"No published manifests found under {root}", err=True)
        raise typer.Exit(code=1)
    failed = False
    for group in groups:
        publication = validate_published_run(group, workspace=workspace_root)
        for error in publication.errors:
            typer.echo(f"  {error}", err=True)
        report = validate_manifests(
            list(publication.manifests),
            workspace=workspace_root,
            verify_checksums=verify_checksums,
            workers=workers,
        )
        typer.echo(
            f"{group.relative_to(root)}: {report.records} records, {report.audio_files} audio files"
        )
        for error in report.errors:
            typer.echo(f"  {error}", err=True)
        failed = failed or not publication.ok or not report.ok
    if failed:
        raise typer.Exit(code=1)


@app.command("export-kaldi")
def export_kaldi_command(
    manifest: Annotated[
        Path,
        typer.Option("--manifest", exists=True, dir_okay=False, resolve_path=True),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", file_okay=False, resolve_path=True),
    ],
    workspace: Annotated[
        Path | None,
        typer.Option("--workspace", file_okay=False, resolve_path=True),
    ] = None,
) -> None:
    """Export a canonical JSONL manifest to Kaldi-compatible sidecars."""

    counts = export_kaldi(manifest, output, workspace=workspace)
    typer.echo(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    app()
