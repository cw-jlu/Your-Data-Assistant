from pathlib import Path
from time import perf_counter

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import load_app_config
from data_agent_baseline.inspect.summarize import format_diff, format_text, summarize_run
from data_agent_baseline.inspect.trace_view import render_trace_from_path
from data_agent_baseline.run.runner import (
    TaskRunArtifacts,
    create_run_output_dir,
    run_benchmark_with_passes,
    run_single_task,
)
from data_agent_baseline.tools.filesystem import list_context_tree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_RUNS_DIR = ARTIFACTS_DIR / "runs"

app = typer.Typer(add_completion=False, no_args_is_help=False)
console = Console()


def _status_value(path: Path) -> str:
    return "present" if path.exists() else "missing"


def _format_compact_rate(completed_count: int, elapsed_seconds: float) -> str:
    if completed_count <= 0 or elapsed_seconds <= 0:
        return "rate=0.0 task/min"
    return f"rate={(completed_count / elapsed_seconds) * 60:.1f} task/min"


def _format_last_task(artifact: TaskRunArtifacts | None) -> str:
    if artifact is None:
        return "last=-"
    status = "ok" if artifact.succeeded else "fail"
    return f"last={artifact.task_id} ({status})"


def _build_compact_progress_fields(
    *,
    completed_count: int,
    succeeded_count: int,
    failed_count: int,
    task_total: int,
    max_workers: int,
    elapsed_seconds: float,
    last_artifact: TaskRunArtifacts | None,
) -> dict[str, str]:
    remaining_count = max(task_total - completed_count, 0)
    running_count = min(max_workers, remaining_count)
    queued_count = max(remaining_count - running_count, 0)
    return {
        "ok": str(succeeded_count),
        "fail": str(failed_count),
        "run": str(running_count),
        "queue": str(queued_count),
        "speed": _format_compact_rate(completed_count, elapsed_seconds),
        "last": _format_last_task(last_artifact),
    }


@app.callback()
def cli() -> None:
    """Utilities for working with the local DABench baseline project."""


@app.command()
def status(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Show the local project layout and public dataset presence."""
    app_config = load_app_config(config)
    config_path = config.resolve()
    public_dataset = DABenchPublicDataset(app_config.dataset.root_path)

    table = Table(title="DABench Baseline Status")
    table.add_column("Item")
    table.add_column("Path")
    table.add_column("State")

    table.add_row("project_root", str(PROJECT_ROOT), "ready")
    table.add_row("data_dir", str(DATA_DIR), _status_value(DATA_DIR))
    table.add_row("configs_dir", str(CONFIGS_DIR), _status_value(CONFIGS_DIR))
    table.add_row("artifacts_dir", str(ARTIFACTS_DIR), _status_value(ARTIFACTS_DIR))
    table.add_row("runs_dir", str(ARTIFACT_RUNS_DIR), _status_value(ARTIFACT_RUNS_DIR))
    table.add_row("dataset_root", str(app_config.dataset.root_path), _status_value(app_config.dataset.root_path))
    table.add_row("config_path", str(config_path), _status_value(config_path))

    console.print(table)

    if public_dataset.exists:
        console.print(f"Public tasks: {len(public_dataset.list_task_ids())}")
        counts = public_dataset.task_counts()
        if counts:
            rendered_counts = ", ".join(
                f"{difficulty}={count}" for difficulty, count in sorted(counts.items())
            )
            console.print(f"Public task counts: {rendered_counts}")


@app.command("inspect-task")
def inspect_task(
    task_id: str,
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Show task metadata and available context files."""
    app_config = load_app_config(config)
    dataset = DABenchPublicDataset(app_config.dataset.root_path)
    task = dataset.get_task(task_id)
    console.print(f"Task: {task.task_id}")
    console.print(f"Difficulty: {task.difficulty}")
    console.print(f"Question: {task.question}")
    context_listing = list_context_tree(task)
    table = Table(title=f"Context Files for {task.task_id}")
    table.add_column("Path")
    table.add_column("Kind")
    table.add_column("Size")
    for entry in context_listing["entries"]:
        table.add_row(str(entry["path"]), str(entry["kind"]), str(entry["size"] or ""))
    console.print(table)


@app.command("run-task")
def run_task_command(
    task_id: str,
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Run the ReAct baseline on one task."""
    app_config = load_app_config(config)
    try:
        _, run_output_dir = create_run_output_dir(
            app_config.run.output_dir,
            run_id=app_config.run.run_id,
            flat=app_config.run.flat_output_dir,
        )
    except (ValueError, FileExistsError) as exc:
        raise typer.BadParameter(str(exc), param_hint="run.run_id") from exc
    artifacts = run_single_task(task_id=task_id, config=app_config, run_output_dir=run_output_dir)

    console.print(f"Run output: {run_output_dir}")
    console.print(f"Task output: {artifacts.task_output_dir}")
    if artifacts.prediction_csv_path is not None:
        console.print(f"Prediction CSV: {artifacts.prediction_csv_path}")
    else:
        console.print("Prediction CSV: not generated")
    if artifacts.failure_reason is not None:
        console.print(f"Failure: {artifacts.failure_reason}")


@app.command("run-benchmark")
def run_benchmark_command(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
    limit: int | None = typer.Option(None, min=1, help="Maximum number of tasks to run."),
    task_set: Path | None = typer.Option(
        None,
        "--task-set",
        exists=True,
        dir_okay=False,
        help="File listing one task_id per line. Restricts the run to those ids.",
    ),
) -> None:
    """Run the ReAct baseline on multiple tasks from the config selection."""
    app_config = load_app_config(config)
    dataset = DABenchPublicDataset(app_config.dataset.root_path)
    task_filter: list[str] | None = None
    if task_set is not None:
        task_filter = [line.strip() for line in task_set.read_text().splitlines() if line.strip()]
    candidate_tasks = dataset.iter_tasks()
    if task_filter is not None:
        wanted = set(task_filter)
        candidate_tasks = [t for t in candidate_tasks if t.task_id in wanted]
    task_total = len(candidate_tasks)
    if limit is not None:
        task_total = min(task_total, limit)
    effective_workers = app_config.run.max_workers

    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[dim]|[/dim]"),
        TextColumn("[green]ok={task.fields[ok]}[/green]"),
        TextColumn("[red]fail={task.fields[fail]}[/red]"),
        TextColumn("[cyan]run={task.fields[run]}[/cyan]"),
        TextColumn("[yellow]queue={task.fields[queue]}[/yellow]"),
        TextColumn("[dim]|[/dim]"),
        TextColumn("{task.fields[speed]}"),
        TextColumn("[dim]| elapsed[/dim]"),
        TimeElapsedColumn(),
        TextColumn("[dim]| eta[/dim]"),
        TimeRemainingColumn(),
        TextColumn("[dim]|[/dim]"),
        TextColumn("{task.fields[last]}"),
    ]
    with Progress(*progress_columns, console=console) as progress:
        progress_task_id = progress.add_task(
            "Benchmark",
            total=task_total,
            completed=0,
            **_build_compact_progress_fields(
                completed_count=0,
                succeeded_count=0,
                failed_count=0,
                task_total=task_total,
                max_workers=effective_workers,
                elapsed_seconds=0.0,
                last_artifact=None,
            ),
        )

        completion_count = 0
        succeeded_count = 0
        failed_count = 0
        start_time = perf_counter()

        def on_task_complete(artifact) -> None:
            nonlocal completion_count, succeeded_count, failed_count
            completion_count += 1
            if artifact.succeeded:
                succeeded_count += 1
            else:
                failed_count += 1
            progress.update(
                progress_task_id,
                completed=completion_count,
                description="Benchmark",
                refresh=True,
                **_build_compact_progress_fields(
                    completed_count=completion_count,
                    succeeded_count=succeeded_count,
                    failed_count=failed_count,
                    task_total=task_total,
                    max_workers=effective_workers,
                    elapsed_seconds=perf_counter() - start_time,
                    last_artifact=artifact,
                ),
            )

        try:
            run_output_dir, artifacts = run_benchmark_with_passes(
                config=app_config,
                limit=limit,
                progress_callback=on_task_complete,
                task_filter=task_filter,
            )
        except (ValueError, FileExistsError) as exc:
            raise typer.BadParameter(str(exc), param_hint="run.run_id") from exc
        progress.update(
            progress_task_id,
            completed=task_total,
            description="Benchmark",
            refresh=True,
            **_build_compact_progress_fields(
                completed_count=task_total,
                succeeded_count=succeeded_count,
                failed_count=failed_count,
                task_total=task_total,
                max_workers=effective_workers,
                elapsed_seconds=perf_counter() - start_time,
                last_artifact=artifacts[-1] if artifacts else None,
            ),
        )
    console.print(f"Run output: {run_output_dir}")
    console.print(f"Tasks attempted: {len(artifacts)}")
    console.print(f"Succeeded tasks: {sum(1 for item in artifacts if item.succeeded)}")


@app.command("inspect-trace")
def inspect_trace_command(
    task_id: str,
    predictions_root: Path = typer.Option(
        ...,
        "--predictions-root",
        exists=True,
        file_okay=False,
        help="Directory containing task_<id>/trace.json subfolders.",
    ),
    full: bool = typer.Option(
        False,
        "--full/--compact",
        help="Show raw_response, full code blocks, and full observation payloads.",
    ),
    input_root: Path | None = typer.Option(
        None,
        "--input-root",
        exists=True,
        file_okay=False,
        help="Optional dataset input root for task.json metadata (difficulty, question).",
    ),
    gold_root: Path | None = typer.Option(
        None,
        "--gold-root",
        exists=True,
        file_okay=False,
        help="Optional gold root for displaying gold.csv alongside the answer.",
    ),
    no_color: bool = typer.Option(False, "--no-color", help="Disable ANSI color."),
) -> None:
    """Render a single task's trace.json step-by-step."""
    trace_path = predictions_root / task_id / "trace.json"
    if not trace_path.is_file():
        raise typer.BadParameter(f"trace.json not found at {trace_path}")
    target_console = Console(no_color=no_color, force_terminal=not no_color)
    render_trace_from_path(
        trace_path,
        console=target_console,
        full=full,
        input_root=input_root,
        gold_root=gold_root,
    )


@app.command("summarize-traces")
def summarize_traces_command(
    predictions_root: Path = typer.Option(
        ...,
        "--predictions-root",
        exists=True,
        file_okay=False,
        help="Directory containing task_<id>/trace.json subfolders.",
    ),
    gold_root: Path | None = typer.Option(
        None,
        "--gold-root",
        exists=True,
        file_okay=False,
        help="Optional gold root — when supplied, scoring breakdown is included.",
    ),
    input_root: Path | None = typer.Option(
        None,
        "--input-root",
        exists=True,
        file_okay=False,
        help="Optional dataset input root for difficulty mapping.",
    ),
    diff: Path | None = typer.Option(
        None,
        "--diff",
        exists=True,
        file_okay=False,
        help="Predictions root to diff against (left side = --diff, right = --predictions-root).",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Walk trace.json files and report failure / soft-reject / step / score buckets."""
    summary = summarize_run(
        predictions_root,
        input_root=input_root,
        gold_root=gold_root,
    )
    if diff is not None:
        other = summarize_run(diff, input_root=input_root, gold_root=gold_root)
        if json_output:
            import json as _json

            console.print_json(_json.dumps({"left": other.to_dict(), "right": summary.to_dict()}))
            return
        # Plain print — rich markup would mangle "[task_22, task_194]" task lists.
        print(format_text(other))
        print()
        print(format_text(summary))
        print()
        print("=== diff ===")
        print(format_diff(other, summary))
        return

    if json_output:
        import json as _json

        console.print_json(_json.dumps(summary.to_dict()))
        return
    print(format_text(summary))


@app.command("update-learnings")
def update_learnings_command(
    predictions_root: Path = typer.Option(
        ...,
        "--predictions-root",
        exists=True,
        file_okay=False,
        help="Directory containing task_<id>/{prediction.csv, trace.json}.",
    ),
    gold_root: Path = typer.Option(
        ...,
        "--gold-root",
        exists=True,
        file_okay=False,
        help="Public-set gold root (data/public/output).",
    ),
    input_root: Path = typer.Option(
        ...,
        "--input-root",
        exists=True,
        file_okay=False,
        help="Public-set dataset root (data/public/input).",
    ),
    apply_low_risk: bool = typer.Option(
        False,
        "--apply-low-risk",
        help="Apply low-risk suggestions in place to learnings.json (NOT high-risk ones).",
    ),
    output_md: Path | None = typer.Option(
        None,
        "--report",
        help="Optional path to write the markdown report.",
    ),
) -> None:
    """v7 host-side recorder — analyse a benchmark run and propose
    `learnings.json` updates without leaderboard feedback. Each public-set
    benchmark run is a learning iteration: this command turns its trace +
    score data into per-shape failure clusters and prints a markdown
    report. ``--apply-low-risk`` writes back numeric multiplier bumps
    automatically; prompt-cue suggestions always require human review.
    """
    from data_agent_baseline.memory.learnings import save_learnings
    from data_agent_baseline.memory.recorder import (
        DEFAULT_ERROR_PATTERNS_PATH,
        DEFAULT_LEARNINGS_PATH,
        aggregate_error_patterns,
        apply_low_risk_suggestions,
        build_report,
        merge_error_patterns,
        render_markdown,
        save_error_patterns,
    )

    report = build_report(
        predictions_root=predictions_root,
        gold_root=gold_root,
        dataset_root=input_root,
    )
    rendered = render_markdown(report)
    # v7.1 N-1: aggregate step-level errors and surface them in the report.
    error_patterns = aggregate_error_patterns(list(report.outcomes))
    if error_patterns:
        rendered += "\n## Error patterns observed (cross-task)\n\n"
        rendered += "| match | action | error_signature | freq | evidence |\n"
        rendered += "|---|---|---|---|---|\n"
        for pat in error_patterns:
            match_str = ", ".join(f"{k}={v}" for k, v in pat["match"].items())
            ev_str = ", ".join(pat["evidence_tasks"][:4])
            rendered += (
                f"| {match_str} | {pat['action']} | {pat['error_signature']} |"
                f" {pat['frequency']} | {ev_str} |\n"
            )
    print(rendered)
    if output_md is not None:
        output_md.write_text(rendered, encoding="utf-8")
        console.print(f"[green]wrote {output_md}[/green]")

    if apply_low_risk:
        new_payload = apply_low_risk_suggestions(report.suggestions)
        save_learnings(new_payload)
        applied = sum(1 for s in report.suggestions if s.risk == "low")
        console.print(
            f"[green]applied {applied} low-risk suggestion(s) to {DEFAULT_LEARNINGS_PATH}[/green]"
        )
        # v7.1 N-1: error patterns are inherently low-risk (they only add
        # advisory hints, never change budgets). Always merge them on
        # --apply-low-risk so the next build picks them up.
        if error_patterns:
            merged = merge_error_patterns(error_patterns)
            save_error_patterns(merged)
            console.print(
                f"[green]merged {len(error_patterns)} error pattern(s)"
                f" into {DEFAULT_ERROR_PATTERNS_PATH}[/green]"
            )


def main() -> None:
    app()
