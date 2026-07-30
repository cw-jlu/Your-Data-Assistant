"""Entry point for experiment exp_059_dual_eval.

Usage:
    python -m experiments.exp_059_dual_eval.run [COMMAND] [OPTIONS]

Commands:
    run-task        Run the agent on a single task
    run-benchmark   Run the agent on all tasks with optional evaluation
"""
from __future__ import annotations

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

from kobushi_core.benchmark.dataset import DABenchPublicDataset
from kobushi_core.eval import EvaluationOptions, evaluate_run, write_evaluation_outputs

from experiments.exp_059_dual_eval.config import load_app_config
from experiments.exp_059_dual_eval.runner import (
    TaskRunArtifacts,
    create_run_output_dir,
    run_benchmark,
    run_single_task,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
DEFAULT_GOLD_ROOT = PROJECT_ROOT / "data" / "public" / "output"

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


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


def _evaluate_run_and_print(
    *,
    run_dir: Path,
    gold_root: Path,
    unordered: bool,
    preview_rows: int,
) -> None:
    options = EvaluationOptions(preview_rows=preview_rows)
    evaluations = evaluate_run(run_dir=run_dir, gold_root=gold_root, options=options)
    outputs = write_evaluation_outputs(run_dir, evaluations)

    console.print(f"Run evaluation: {run_dir}")
    console.print(f"Tasks evaluated: {len(evaluations)}")
    metric_name = "numeric_tolerant_unordered_match" if unordered else "numeric_tolerant_match"
    metric_count = sum(1 for item in evaluations if getattr(item, metric_name))
    console.print(f"{metric_name}: {metric_count}")
    mean_score = (
        sum(item.official_score_lambda_0_5 for item in evaluations) / len(evaluations)
        if evaluations
        else 0.0
    )
    console.print(f"official_score_lambda_0_5_mean: {mean_score:.6f}")
    console.print(f"Evaluation JSON: {outputs.json_path}")
    console.print(f"Evaluation CSV: {outputs.csv_path}")


@app.command("run-task")
def run_task_command(
    task_id: str,
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False, help="YAML config."),
) -> None:
    """Run the ReAct baseline on one task."""
    app_config = load_app_config(config)
    try:
        _, run_output_dir = create_run_output_dir(
            app_config.run.output_dir, run_id=app_config.run.run_id
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
    config: Path = typer.Option(DEFAULT_CONFIG, exists=True, dir_okay=False, help="YAML config."),
    limit: int | None = typer.Option(None, min=1, help="Maximum number of tasks to run."),
    evaluate: bool = typer.Option(
        False,
        "--evaluate/--no-evaluate",
        help="Evaluate generated prediction.csv files after the benchmark run.",
    ),
    gold_root: Path = typer.Option(
        DEFAULT_GOLD_ROOT, file_okay=False, help="Gold CSV root used when --evaluate is set."
    ),
    unordered: bool = typer.Option(True, help="Report official-style column signature metrics."),
    preview_rows: int = typer.Option(5, min=0, help="Rows included in JSON evaluation previews."),
) -> None:
    """Run the ReAct baseline on multiple tasks from the config selection."""
    app_config = load_app_config(config)
    dataset = DABenchPublicDataset(app_config.dataset.root_path)
    task_total = len(dataset.iter_tasks())
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
            run_output_dir, artifacts = run_benchmark(
                config=app_config,
                limit=limit,
                progress_callback=on_task_complete,
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
    if evaluate:
        _evaluate_run_and_print(
            run_dir=run_output_dir,
            gold_root=gold_root,
            unordered=unordered,
            preview_rows=preview_rows,
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
