"""`dabench` CLI 入口模块。

暴露 5 个子命令：
- `status`: 打印项目目录结构与数据集状态
- `inspect-task <id>`: 查看单任务的元信息与 context 文件树
- `run-task <id>`: 对单个任务跑一次 ReAct agent
- `run-benchmark`: 对数据集里全部（或 `--limit N` / `--range S-E`）任务批量跑
- `dashboard`: 启动 tracing dashboard（FastAPI）

打包发布时入口由 `pyproject.toml` 的 `[project.scripts]` 暴露为 `dabench`，
直接运行 `python -m agents.cli` 在 Typer + 模块运行器下会无输出，
务必通过 `uv run dabench ...` 调用。
"""

from pathlib import Path
from time import perf_counter
from typing import Any, cast

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

from agents.benchmark.dataset import DABenchPublicDataset, task_id_number
from agents.config import load_app_config
from agents.runs.runner import (
    TaskRunArtifacts,
    create_run_output_dir,
    run_benchmark,
    run_single_task,
)
from agents.tools import list_context_tree

# 关键路径常量：全部以 PROJECT_ROOT 为基，方便 status 报告和默认配置回落
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_RUNS_DIR = ARTIFACTS_DIR / "runs"

# Typer 应用实例：禁用自动补全并允许无参数调用（会显示 help）
app = typer.Typer(add_completion=False, no_args_is_help=False)
# Rich 控制台，供所有子命令统一输出彩色表格 / 进度条
console = Console()


def _status_value(path: Path) -> str:
    """把路径是否存在渲染成 `status` 表格里的 present/missing 文案。"""
    return "present" if path.exists() else "missing"


def _coerce_task_number(token: str) -> int:
    """`--range` 每端的解析：纯数字直接 int，`task_<n>` 借公共 task_id_number 走前缀校验。"""
    cleaned = token.strip()
    if cleaned.startswith("task_"):
        return task_id_number(cleaned)
    return int(cleaned)


def _parse_task_range(spec: str) -> tuple[int, int]:
    """把 `--range` 字符串解析成 (start, end)，两端含。

    支持 `5-10` / `task_5-task_10` / 混合形式。错误一律转成 `typer.BadParameter`，
    让 CLI 出干净的错误提示而不是 traceback。
    """
    cleaned = spec.strip()
    left, sep, right = cleaned.partition("-")
    if not sep or not left.strip() or not right.strip():
        raise typer.BadParameter(
            f"--range 需要 START-END 形式（例如 '5-10' 或 'task_5-task_10'），收到 '{spec}'.",
            param_hint="--range",
        )
    try:
        start = _coerce_task_number(left)
        end = _coerce_task_number(right)
    except ValueError as exc:
        raise typer.BadParameter(
            f"--range 解析失败 ('{spec}'): {exc}", param_hint="--range"
        ) from exc
    if start < 1:
        raise typer.BadParameter(
            f"--range 起始任务号必须 >= 1，收到 {start}.", param_hint="--range"
        )
    if start > end:
        raise typer.BadParameter(
            f"--range 起始 ({start}) 不能大于结束 ({end}).", param_hint="--range"
        )
    return start, end


def _format_compact_rate(completed_count: int, elapsed_seconds: float) -> str:
    """把 "已完成任务数/累计秒数" 换算为 "task/min" 文案用于进度条展示。"""
    # 防御 0 除：任务刚开始或刚失败时，不展示误导性的速率
    if completed_count <= 0 or elapsed_seconds <= 0:
        return "rate=0.0 task/min"
    return f"rate={(completed_count / elapsed_seconds) * 60:.1f} task/min"


def _format_last_task(artifact: TaskRunArtifacts | None) -> str:
    """在进度条右侧展示最近完成的任务号及其 ok/fail 状态。"""
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
) -> dict[str, Any]:
    """构造 rich.progress 自定义字段的 dict（`task.fields[<key>]` 引用）。

    `run` 字段估算当前正在跑的任务数：剩余里最多 `max_workers` 个被 worker 占用，
    其余计入 `queue`。这只是展示上的近似——真正的并发由线程池决定。
    """
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
    """Utilities for working with the local DABench agent project."""
    # Typer 要求 callback 存在才能展示整体 help；不需要额外逻辑


@app.command()
def status(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Show the local project layout and public dataset presence."""
    # 读取配置，拿到数据集根路径（可能被 YAML 覆盖成自定义位置）
    app_config = load_app_config(config)
    config_path = config.resolve()
    public_dataset = DABenchPublicDataset(app_config.dataset.root_path)

    # 组装状态表：列出关键目录的物理位置和存在性
    table = Table(title="DABench Agent Status")
    table.add_column("Item")
    table.add_column("Path")
    table.add_column("State")

    table.add_row("project_root", str(PROJECT_ROOT), "ready")
    table.add_row("data_dir", str(DATA_DIR), _status_value(DATA_DIR))
    table.add_row("configs_dir", str(CONFIGS_DIR), _status_value(CONFIGS_DIR))
    table.add_row("artifacts_dir", str(ARTIFACTS_DIR), _status_value(ARTIFACTS_DIR))
    table.add_row("runs_dir", str(ARTIFACT_RUNS_DIR), _status_value(ARTIFACT_RUNS_DIR))
    table.add_row(
        "dataset_root",
        str(app_config.dataset.root_path),
        _status_value(app_config.dataset.root_path),
    )
    table.add_row("config_path", str(config_path), _status_value(config_path))

    console.print(table)

    # 数据集存在时，补报任务总数与按 difficulty 的计数，帮助快速核对数据版本
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

    # 打印任务头部元信息：ID / difficulty / 自然语言问题
    console.print(f"Task: {task.task_id}")
    console.print(f"Difficulty: {task.difficulty}")
    console.print(f"Question: {task.question}")

    # 再把 context/ 目录下的文件列表以表格渲染出来，供手工调试参考
    context_listing = list_context_tree(task)
    table = Table(title=f"Context Files for {task.task_id}")
    table.add_column("Path")
    table.add_column("Kind")
    table.add_column("Size")
    entries = cast(list[dict[str, Any]], context_listing["entries"])
    for entry in entries:
        # 目录 entry 的 size 为 None，这里统一渲染为空字符串避免显示 "None"
        table.add_row(str(entry["path"]), str(entry["kind"]), str(entry["size"] or ""))
    console.print(table)


@app.command("run-task")
def run_task_command(
    task_id: str,
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Run the ReAct agent on one task."""
    app_config = load_app_config(config)
    # 即便只跑一个任务，也复用 run-benchmark 的目录结构：
    # <output_dir>/<run_id>/<task_id>/prediction.csv
    try:
        _, run_output_dir = create_run_output_dir(
            app_config.run.output_dir, run_id=app_config.run.run_id
        )
    except (ValueError, FileExistsError) as exc:
        # run_id 非法或已存在时转化为 Typer 友好的错误提示
        raise typer.BadParameter(str(exc), param_hint="run.run_id") from exc
    artifacts = run_single_task(task_id=task_id, config=app_config, run_output_dir=run_output_dir)

    # 汇报产物落盘位置，方便用户直接打开查看
    console.print(f"Run output: {run_output_dir}")
    console.print(f"Task output: {artifacts.task_output_dir}")
    if artifacts.prediction_csv_path is not None:
        console.print(f"Prediction CSV: {artifacts.prediction_csv_path}")
    else:
        # Agent 未触发 answer 工具时就没有 CSV 产物
        console.print("Prediction CSV: not generated")
    if artifacts.failure_reason is not None:
        console.print(f"Failure: {artifacts.failure_reason}")


@app.command("run-benchmark")
def run_benchmark_command(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
    limit: int | None = typer.Option(None, min=1, help="Maximum number of tasks to run."),
    range_spec: str | None = typer.Option(
        None,
        "--range",
        help=(
            "Inclusive task range, e.g. '5-10' or 'task_5-task_10'. "
            "Combines with --limit (range filters first, then limit truncates)."
        ),
    ),
) -> None:
    """Run the ReAct agent on multiple tasks from the config selection."""
    app_config = load_app_config(config)
    dataset = DABenchPublicDataset(app_config.dataset.root_path)
    task_range = _parse_task_range(range_spec) if range_spec else None
    # 用 task_dirs()（仅 listdir + sort，不读 task.json）算总数，跟 runner 的过滤逻辑
    # 等价：先按 range 过滤，再用 limit 截断。这样进度条 total 和真正跑的任务数一致。
    selected_dirs = dataset.task_dirs()
    if task_range is not None:
        range_start, range_end = task_range
        selected_dirs = [
            path for path in selected_dirs if range_start <= task_id_number(path.name) <= range_end
        ]
    if limit is not None:
        selected_dirs = selected_dirs[:limit]
    task_total = len(selected_dirs)
    if task_total == 0:
        raise typer.BadParameter(
            "Selection (range/limit) produced 0 tasks; nothing to run.",
            param_hint="--range" if task_range is not None else "--limit",
        )
    effective_workers = app_config.run.max_workers

    # ---- 进度条列定义 ----
    # 所有自定义字段（ok/fail/run/queue/speed/last）通过 `task.fields[...]` 引用
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
        # 初始化进度条行，所有字段给零值占位
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

        # 三个滚动计数器 + 开始时间戳，用于速率估算
        completion_count = 0
        succeeded_count = 0
        failed_count = 0
        start_time = perf_counter()

        def on_task_complete(artifact: TaskRunArtifacts) -> None:
            """单任务完成回调：由 runner 在每个任务产出后触发。

            runner 在并行模式下通过 `as_completed` 顺序调用，保证对 nonlocal
            计数器的写入是串行的（进度条 UI 不需要额外加锁）。
            """
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

        # 实际运行发生在 runner.run_benchmark；progress_callback 串联进度刷新
        try:
            run_output_dir, artifacts = run_benchmark(
                config=app_config,
                limit=limit,
                task_range=task_range,
                progress_callback=on_task_complete,
            )
        except (ValueError, FileExistsError) as exc:
            # 同 run-task：非法 run_id 转化为 Typer 错误
            raise typer.BadParameter(str(exc), param_hint="run.run_id") from exc

        # 收尾刷新：确保即便最后一个回调晚到，进度条也以 100% 状态结束
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
    # 终端汇总：run 目录、任务总数、成功数。注意 "succeeded" 指 Agent 调用 answer 成功，
    # 并不等于答案正确——评分需要另外比对 prediction.csv vs gold.csv。
    console.print(f"Run output: {run_output_dir}")
    console.print(f"Tasks attempted: {len(artifacts)}")
    console.print(f"Succeeded tasks: {sum(1 for item in artifacts if item.succeeded)}")


def _ensure_dashboard_built() -> Path:
    """Build the dashboard frontend if the static assets are missing."""
    import shutil
    import subprocess

    dashboard_pkg = Path(__file__).parent / "dashboard"
    static_dir = dashboard_pkg / "static"
    frontend_dir = PROJECT_ROOT / "dashboard" / "frontend"

    if (static_dir / "index.html").is_file():
        return static_dir

    if not (frontend_dir / "package.json").is_file():
        raise typer.BadParameter(
            f"Dashboard frontend source not found at {frontend_dir}. "
            "Run from a source checkout, or pre-build with: "
            "cd dashboard/frontend && npm ci && npm run build"
        )

    npm = shutil.which("npm")
    if npm is None:
        raise typer.BadParameter(
            "npm not found. Install Node.js to build the dashboard frontend, "
            "or pre-build with: cd dashboard/frontend && npm ci && npm run build"
        )

    console.print("[dim]Installing frontend dependencies…[/dim]")
    subprocess.run([npm, "ci"], cwd=str(frontend_dir), check=True)
    console.print("[dim]Building dashboard frontend…[/dim]")
    subprocess.run([npm, "run", "build"], cwd=str(frontend_dir), check=True)

    if not (static_dir / "index.html").is_file():
        raise typer.BadParameter("Frontend build succeeded but static/index.html not found")

    return static_dir


@app.command("dashboard")
def dashboard_command(
    port: int = typer.Option(8765, "--port", help="Dashboard server port."),
    db_path: Path = typer.Option(
        ARTIFACTS_DIR / "traces.db",
        "--db",
        help="Path to traces.db (default: artifacts/traces.db).",
    ),
) -> None:
    """Launch the tracing dashboard (FastAPI)."""
    if not db_path.exists():
        console.print(
            f"[yellow]Warning:[/yellow] {db_path} not found; starting with empty database."
        )
    try:
        import uvicorn
    except ImportError as exc:
        raise typer.BadParameter(
            "uvicorn not found. Install the dev extra first: uv sync --extra dev"
        ) from exc

    static_dir = _ensure_dashboard_built()

    from agents.dashboard.app import create_app

    dash_app = create_app(db_path, static_dir=static_dir)
    console.print(f"Dashboard: http://127.0.0.1:{port}")
    uvicorn.run(dash_app, host="127.0.0.1", port=port, log_level="info")


def main() -> None:
    """`pyproject.toml` 中 `[project.scripts] dabench = "..:main"` 的入口函数。"""
    app()
