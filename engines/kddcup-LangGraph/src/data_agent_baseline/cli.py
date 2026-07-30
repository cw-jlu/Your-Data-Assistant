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

from data_agent_baseline.asr_eval.pipeline import generate_azure_gold, run_candidate_asr
from data_agent_baseline.asr_eval.scoring import score_asr_run
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import load_app_config
from data_agent_baseline.run.runner import (
    TaskRunArtifacts,
    build_asr_prompt_model,
    create_benchmark_output_dirs,
    run_benchmark,
    run_single_task,
    run_task_repeatedly,
)
from data_agent_baseline.scoring import compare_run_scores, resolve_score_run_dir, score_run_outputs
from data_agent_baseline.tools.filesystem import list_context_tree

# 约定好的项目目录入口，CLI 会基于这些路径展示状态和写出产物。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_RUNS_DIR = ARTIFACTS_DIR / "runs"
PUBLIC_GOLD_DIR = DATA_DIR / "output"
ASR_EVALUATION_DIR = PROJECT_ROOT / "evaluation" / "asr"
ASR_ARTIFACTS_DIR = ARTIFACTS_DIR / "asr"

# Typer 应用入口和统一的 rich 控制台输出对象。
app = typer.Typer(add_completion=False, no_args_is_help=False)
console = Console()


# 将路径存在性压缩成简短的状态文案。
def _status_value(path: Path) -> str:
    return "present" if path.exists() else "missing"


# 以 task/min 的格式展示当前吞吐速率，便于进度条紧凑展示。
def _format_compact_rate(completed_count: int, elapsed_seconds: float) -> str:
    if completed_count <= 0 or elapsed_seconds <= 0:
        return "rate=0.0 task/min"
    return f"rate={(completed_count / elapsed_seconds) * 60:.1f} task/min"


# 在进度条右侧显示最近完成任务的编号和状态。
def _format_last_task(artifact: TaskRunArtifacts | None) -> str:
    if artifact is None:
        return "last=-"
    status = "ok" if artifact.succeeded else "fail"
    return f"last={artifact.task_id} ({status})"


# 为 rich Progress 统一构造紧凑字段，避免在多处重复拼接展示逻辑。
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


# CLI 根命令，仅用于挂载子命令。
@app.callback()
def cli() -> None:
    """Utilities for working with the local DABench baseline project."""


# 查看本地项目路径、配置路径和公开数据集是否就绪。
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


# 查看单个任务的元信息以及 context/ 下可访问的文件。
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


# 运行单个任务，并打印输出目录、预测文件和失败原因。
@app.command("run-task")
def run_task_command(
    task_id: str,
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Run the LangGraph baseline on one task."""
    app_config = load_app_config(config)
    try:
        output_dirs = create_benchmark_output_dirs(app_config)
    except (ValueError, FileExistsError) as exc:
        raise typer.BadParameter(str(exc), param_hint="run.run_id") from exc
    artifacts = run_single_task(
        task_id=task_id,
        config=app_config,
        run_output_dir=output_dirs.run_output_dir,
        prediction_output_root=output_dirs.prediction_output_root,
    )

    console.print(f"Run output: {output_dirs.run_output_dir}")
    console.print(f"Task output: {artifacts.task_output_dir}")
    if artifacts.prediction_csv_path is not None:
        console.print(f"Prediction CSV: {artifacts.prediction_csv_path}")
    else:
        console.print("Prediction CSV: not generated")
    if artifacts.failure_reason is not None:
        console.print(f"Failure: {artifacts.failure_reason}")


@app.command("repeat-task")
def repeat_task_command(
    task_id: str,
    runs: int = typer.Option(10, "--runs", "-n", min=1, help="Number of times to run the task."),
    workers: int = typer.Option(8, "--workers", "-w", min=1, help="Number of concurrent workers."),
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Run a single task N times and evaluate accuracy."""
    app_config = load_app_config(config)
    total_workers = min(workers, runs)

    console.print(f"Task: {task_id}")
    console.print(f"Runs: {runs}  |  Workers: {total_workers}")

    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[dim]|[/dim]"),
        TextColumn("[green]ok={task.fields[ok]}[/green]"),
        TextColumn("[red]fail={task.fields[fail]}[/red]"),
        TextColumn("[dim]|[/dim]"),
        TextColumn("{task.fields[speed]}"),
        TextColumn("[dim]| elapsed[/dim]"),
        TimeElapsedColumn(),
        TextColumn("[dim]| eta[/dim]"),
        TimeRemainingColumn(),
        TextColumn("[dim]|[/dim]"),
        TextColumn("{task.fields[last]}"),
    ]

    completion_count = 0
    succeeded_count = 0
    failed_count = 0
    start_time = perf_counter()

    with Progress(*progress_columns, console=console) as progress:
        progress_task_id = progress.add_task(
            f"Repeating {task_id}",
            total=runs,
            completed=0,
            **_build_compact_progress_fields(
                completed_count=0,
                succeeded_count=0,
                failed_count=0,
                task_total=runs,
                max_workers=total_workers,
                elapsed_seconds=0.0,
                last_artifact=None,
            ),
        )

        def on_run_complete(artifact: TaskRunArtifacts) -> None:
            nonlocal completion_count, succeeded_count, failed_count
            completion_count += 1
            if artifact.succeeded:
                succeeded_count += 1
            else:
                failed_count += 1
            progress.update(
                progress_task_id,
                completed=completion_count,
                description=f"Repeating {task_id}",
                refresh=True,
                **_build_compact_progress_fields(
                    completed_count=completion_count,
                    succeeded_count=succeeded_count,
                    failed_count=failed_count,
                    task_total=runs,
                    max_workers=total_workers,
                    elapsed_seconds=perf_counter() - start_time,
                    last_artifact=artifact,
                ),
            )

        try:
            result = run_task_repeatedly(
                task_id=task_id,
                config=app_config,
                runs=runs,
                workers=workers,
                progress_callback=on_run_complete,
            )
        except (ValueError, FileExistsError) as exc:
            raise typer.BadParameter(str(exc), param_hint="run.run_id") from exc

    console.print(f"Run output: {result.run_output_dir}")

    summary_table = Table(title=f"Repeat-Task Results for {task_id} ({runs} runs)")
    summary_table.add_column("Metric")
    summary_table.add_column("Value")

    accuracy_pct = (result.full_cover_count / result.runs) * 100 if result.runs > 0 else 0.0
    success_pct = (result.success_count / result.runs) * 100 if result.runs > 0 else 0.0

    summary_table.add_row("Accuracy (full cover)", f"{result.full_cover_count}/{result.runs} = {accuracy_pct:.1f}%")
    summary_table.add_row("Success rate", f"{result.success_count}/{result.runs} = {success_pct:.1f}%")
    summary_table.add_row("Primary Proxy Score", f"{result.primary_proxy_score:.4f}")
    summary_table.add_row("Mean Recall", f"{result.mean_recall:.4f}")
    summary_table.add_row("Mean Redundancy", f"{result.mean_redundancy:.4f}")
    summary_table.add_row("Total elapsed", f"{result.total_elapsed_seconds:.1f}s")
    console.print(summary_table)

    per_run_table = Table(title="Per-Run Breakdown")
    per_run_table.add_column("Run")
    per_run_table.add_column("Succeeded")
    per_run_table.add_column("Full Cover")
    per_run_table.add_column("Recall")
    per_run_table.add_column("Redundancy")
    per_run_table.add_column("Proxy Score")
    per_run_table.add_column("Failure")
    for i, (artifact, score) in enumerate(
        zip(result.artifacts, result.task_scores), start=1
    ):
        per_run_table.add_row(
            f"run_{i:02d}",
            "[green]yes[/green]" if artifact.succeeded else "[red]no[/red]",
            "[green]yes[/green]" if score.full_cover else "[red]no[/red]",
            f"{score.recall:.4f}",
            f"{score.redundancy_rate:.4f}",
            f"{score.primary_proxy_score:.4f}",
            artifact.failure_reason or "-",
        )
    console.print(per_run_table)


# 批量运行 benchmark，并用 rich 进度条显示完成情况和吞吐。
@app.command("run-benchmark")
def run_benchmark_command(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
    limit: int | None = typer.Option(None, min=1, help="Maximum number of tasks to run."),
    skip_completed: bool = typer.Option(
        False,
        "--skip-completed",
        help="Skip tasks that already have prediction.csv under the configured output root.",
    ),
) -> None:
    """Run the LangGraph baseline on multiple tasks from the config selection."""
    app_config = load_app_config(config)
    dataset = DABenchPublicDataset(app_config.dataset.root_path)
    selected_task_ids = list(app_config.run.task_ids or ())
    selected_tasks = dataset.iter_tasks(task_ids=selected_task_ids or None)
    if skip_completed and app_config.run.output_layout == "flat":
        selected_tasks = [
            task
            for task in selected_tasks
            if not (app_config.run.output_dir / task.task_id / "prediction.csv").is_file()
        ]
    task_total = len(selected_tasks)
    if limit is not None:
        task_total = min(task_total, limit)
    effective_workers = app_config.run.max_workers

    # 进度条字段刻意做得紧凑，方便在终端中同时展示成功/失败、速度和最近任务。
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

        # 每完成一个任务就刷新一次统计信息和进度条展示。
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
                skip_completed=skip_completed,
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

    try:
        score_summary = score_run_outputs(
            run_output_dir=run_output_dir,
            gold_root=PUBLIC_GOLD_DIR,
        )
        console.print(f"Primary proxy score: {score_summary.primary_proxy_score:.4f}")
        console.print(f"Score report: {score_summary.score_report_path}")
    except Exception as exc:
        console.print(f"[yellow]Scoring skipped: {exc}[/yellow]")


# 对某次 run 的 prediction.csv 按官方列匹配规则打分。
@app.command("score-run")
def score_run_command(
    run_id: str | None = typer.Argument(
        None,
        help=(
            "Optional run directory name under artifacts/runs. Latest run is used when omitted. "
            "The run must include summary.json because score-run only evaluates task ids recorded there."
        ),
    ),
    lambda_values: list[float] | None = typer.Option(
        None,
        "--lambda",
        help="Optional proxy lambda values. Repeat the flag to evaluate multiple lambda settings.",
    ),
) -> None:
    """Score one run directory using only the task ids recorded in summary.json."""
    try:
        effective_run_id, run_output_dir = resolve_score_run_dir(ARTIFACT_RUNS_DIR, run_id=run_id)
        summary = score_run_outputs(
            run_output_dir=run_output_dir,
            gold_root=PUBLIC_GOLD_DIR,
            lambda_values=lambda_values,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc), param_hint="run_id") from exc

    summary_table = Table(title=f"Score Summary for {effective_run_id}")
    summary_table.add_column("Item")
    summary_table.add_column("Value")
    summary_table.add_row("run_id", effective_run_id)
    summary_table.add_row("run_output", str(run_output_dir))
    summary_table.add_row("public_gold_dir", str(PUBLIC_GOLD_DIR))
    summary_table.add_row("task_source", summary.task_source)
    summary_table.add_row("task_count", str(summary.task_count))
    summary_table.add_row("prediction_task_count", str(summary.prediction_task_count))
    summary_table.add_row("primary_lambda", f"{summary.primary_lambda:g}")
    summary_table.add_row("primary_proxy_score", f"{summary.primary_proxy_score:.4f}")
    summary_table.add_row("mean_recall", f"{summary.mean_recall:.4f}")
    summary_table.add_row("mean_redundancy_rate", f"{summary.mean_redundancy_rate:.4f}")
    summary_table.add_row("score_json", str(summary.score_path))
    summary_table.add_row("score_report", str(summary.score_report_path))
    console.print(summary_table)

    proxy_table = Table(title="Proxy Scores (Recall - λ * Redundancy)")
    proxy_table.add_column("λ")
    proxy_table.add_column("score")
    for label, score in summary.proxy_scores.items():
        proxy_table.add_row(label, f"{score:.4f}")
    console.print(proxy_table)

    if summary.failure_breakdown:
        failure_table = Table(title="Failure Breakdown")
        failure_table.add_column("Failure")
        failure_table.add_column("Count")
        for reason, count in summary.failure_breakdown.items():
            failure_table.add_row(reason, str(count))
        console.print(failure_table)


def _format_comparison_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _render_plain_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(header), *(len(row[index]) for row in rows)) if rows else len(header)
        for index, header in enumerate(headers)
    ]
    rendered_rows = [
        " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "-+-".join("-" * width for width in widths),
    ]
    rendered_rows.extend(
        " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )
    return "\n".join(rendered_rows)


@app.command("compare-runs")
def compare_runs_command(
    run_refs: list[str] = typer.Argument(
        ...,
        help=(
            "Run ids under artifacts/runs, or paths containing score.json "
            "such as artifacts/standard/baseline."
        ),
    ),
) -> None:
    """Compare already-scored runs by reading each run's score.json."""
    try:
        rows = compare_run_scores(
            runs_root=ARTIFACT_RUNS_DIR,
            run_refs=run_refs,
            base_dir=PROJECT_ROOT,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="run_refs") from exc

    headers = [
        "run_id",
        "task_count",
        "prediction_task_count",
        "primary_proxy_score",
        "mean_recall",
        "mean_redundancy_rate",
        "mean_model_step_count",
        "max_model_step_count",
        "p95_e2e_elapsed_seconds",
        "top_failure",
    ]
    rendered_rows = [
        [
            row.run_id,
            str(row.task_count),
            str(row.prediction_task_count),
            _format_comparison_float(row.primary_proxy_score),
            _format_comparison_float(row.mean_recall),
            _format_comparison_float(row.mean_redundancy_rate),
            _format_comparison_float(row.mean_model_step_count, digits=2),
            str(row.max_model_step_count),
            _format_comparison_float(row.p95_e2e_elapsed_seconds, digits=3),
            row.top_failure or "-",
        ]
        for row in rows
    ]

    console.print("Run Comparison")
    console.print(_render_plain_table(headers, rendered_rows), soft_wrap=True)


@app.command("asr-generate-gold")
def asr_generate_gold_command(
    gold_id: str = typer.Option(..., help="Immutable identifier for this Azure gold set."),
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
    task_ids: list[str] | None = typer.Option(
        None,
        "--task-id",
        help="Optional task id. Repeat this option to select multiple video tasks.",
    ),
) -> None:
    """Generate or safely resume frozen Azure Fast Transcription references."""
    app_config = load_app_config(config)
    selected_task_ids = task_ids or list(app_config.run.task_ids or ())
    try:
        manifest_path, manifest = generate_azure_gold(
            project_root=PROJECT_ROOT,
            dataset_root=app_config.dataset.root_path,
            gold_id=gold_id,
            task_ids=selected_task_ids or None,
            evaluation_root=ASR_EVALUATION_DIR,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="asr-generate-gold") from exc

    samples = list(manifest.get("samples") or [])
    succeeded = sum(1 for item in samples if item.get("status") == "ok")
    console.print(f"Gold manifest: {manifest_path}")
    console.print(f"Azure references: {succeeded}/{len(samples)}")
    if not manifest.get("completed"):
        console.print("[red]Gold generation is incomplete. Inspect failed sample entries.[/red]")
        raise typer.Exit(code=1)


@app.command("asr-run")
def asr_run_command(
    run_id: str = typer.Option(..., help="Immutable identifier for this ASR candidate run."),
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
    task_ids: list[str] | None = typer.Option(
        None,
        "--task-id",
        help="Optional task id. Repeat this option to select multiple video tasks.",
    ),
    stage: str | None = typer.Option(
        None,
        "--stage",
        help=(
            "ASR ablation stage: medium-baseline, tiny-route, dynamic-prompt, "
            "or dynamic-plus-ui. Defaults to video_preprocessing.asr_stage."
        ),
    ),
) -> None:
    """Run or safely resume the configured faster-whisper ASR baseline."""
    app_config = load_app_config(config)
    video = app_config.video_preprocessing
    effective_stage = stage or video.asr_stage
    selected_task_ids = task_ids or list(app_config.run.task_ids or ())
    prompt_model = None
    prompt_model_error = None
    if effective_stage in {"dynamic-prompt", "dynamic-plus-ui"}:
        try:
            prompt_model = build_asr_prompt_model(app_config)
        except Exception as exc:  # noqa: BLE001
            prompt_model_error = f"{type(exc).__name__}: {exc}"
            console.print(
                "[yellow]Qwen prompt model is unavailable; ASR will use the configured "
                "safe fallback.[/yellow]"
            )
    try:
        manifest_path, manifest = run_candidate_asr(
            project_root=PROJECT_ROOT,
            dataset_root=app_config.dataset.root_path,
            run_id=run_id,
            model_name=video.asr_model,
            device=video.asr_device,
            compute_type=video.asr_compute_type,
            stage=effective_stage,
            detector_model_name=video.asr_language_detector_model,
            cpu_threads=video.asr_cpu_threads,
            num_workers=video.asr_num_workers,
            max_workers=min(video.asr_num_workers, 8),
            language_threshold=video.asr_language_threshold,
            prompt_min_terms=video.asr_prompt_min_terms,
            prompt_max_terms=video.asr_prompt_max_terms,
            ui_terms=video.asr_ui_terms,
            prompt_model=prompt_model,
            prompt_model_name=app_config.agent.model,
            prompt_model_base=app_config.agent.api_base,
            task_ids=selected_task_ids or None,
            evaluation_root=ASR_EVALUATION_DIR,
            artifacts_root=ASR_ARTIFACTS_DIR,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="asr-run") from exc

    samples = list(manifest.get("samples") or [])
    succeeded = sum(1 for item in samples if item.get("status") == "ok")
    console.print(f"ASR run manifest: {manifest_path}")
    console.print(
        "Candidate: "
        f"{video.asr_model} / {video.asr_device} / {video.asr_compute_type} / "
        f"{effective_stage}"
    )
    if prompt_model_error:
        console.print(f"[yellow]Prompt model setup fallback: {prompt_model_error}[/yellow]")
    console.print(f"Candidate transcripts: {succeeded}/{len(samples)}")
    if not manifest.get("completed"):
        console.print("[red]Candidate run is incomplete. Inspect failed sample entries.[/red]")
        raise typer.Exit(code=1)


@app.command("asr-score")
def asr_score_command(
    gold_id: str = typer.Option(..., help="Frozen Azure gold identifier."),
    run_id: str = typer.Option(..., help="ASR candidate run identifier."),
) -> None:
    """Score one candidate run offline against one frozen Azure gold set."""
    try:
        score_path, score = score_asr_run(
            project_root=PROJECT_ROOT,
            gold_id=gold_id,
            run_id=run_id,
            evaluation_root=ASR_EVALUATION_DIR,
            artifacts_root=ASR_ARTIFACTS_DIR,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="asr-score") from exc

    overview = score["overview"]
    console.print(f"ASR score: {score_path}")
    console.print(
        f"Candidate success: {overview['successful_sample_count']}/"
        f"{overview['sample_count']} ({overview['success_rate']:.2%})"
    )
    for language in ("zh", "en"):
        summary = score["language_summaries"].get(language)
        if summary:
            console.print(
                f"{language} {summary['metric']}: "
                f"corpus={summary['corpus_error_rate']:.4%}, "
                f"macro={summary['macro_error_rate']:.4%}"
            )
    console.print(
        "[yellow]Metrics are relative to Azure Fast Transcription output, "
        "not absolute transcription accuracy.[/yellow]"
    )


# 供 pyproject 或脚本入口直接调用的主函数。
def main() -> None:
    app()
