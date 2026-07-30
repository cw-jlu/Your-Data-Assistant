from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from kobushi_core.benchmark.dataset import DABenchPublicDataset
from kobushi_core.eval import EvaluationOptions, evaluate_run, write_evaluation_outputs
from kobushi_core.utils import list_context_tree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_RUNS_DIR = ARTIFACTS_DIR / "runs"
CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_GOLD_ROOT = DATA_DIR / "public" / "output"

app = typer.Typer(add_completion=False, no_args_is_help=False)
console = Console()


def _status_value(path: Path) -> str:
    return "present" if path.exists() else "missing"


def _evaluate_run_and_print(
    *,
    run_dir: Path,
    gold_root: Path,
    rtol: float,
    atol: float,
    unordered: bool,
    preview_rows: int,
) -> None:
    options = EvaluationOptions(
        rtol=rtol,
        atol=atol,
        preview_rows=preview_rows,
    )
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


@app.callback()
def cli() -> None:
    """Utilities for working with the kobushi DABench project."""


@app.command()
def status(
    dataset_root: Path = typer.Option(
        None, help="Dataset root path. Defaults to data/public/input."
    ),
) -> None:
    """Show the local project layout and public dataset presence."""
    effective_root = dataset_root or (DATA_DIR / "public" / "input")
    public_dataset = DABenchPublicDataset(effective_root)

    table = Table(title="Kobushi DABench Status")
    table.add_column("Item")
    table.add_column("Path")
    table.add_column("State")

    table.add_row("project_root", str(PROJECT_ROOT), "ready")
    table.add_row("data_dir", str(DATA_DIR), _status_value(DATA_DIR))
    table.add_row("configs_dir", str(CONFIGS_DIR), _status_value(CONFIGS_DIR))
    table.add_row("artifacts_dir", str(ARTIFACTS_DIR), _status_value(ARTIFACTS_DIR))
    table.add_row("runs_dir", str(ARTIFACT_RUNS_DIR), _status_value(ARTIFACT_RUNS_DIR))
    table.add_row("dataset_root", str(effective_root), _status_value(effective_root))

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
    dataset_root: Path = typer.Option(
        None, help="Dataset root path. Defaults to data/public/input."
    ),
) -> None:
    """Show task metadata and available context files."""
    effective_root = dataset_root or (DATA_DIR / "public" / "input")
    dataset = DABenchPublicDataset(effective_root)
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


@app.command("evaluate-run")
def evaluate_run_command(
    run_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, help="Run artifact directory."
    ),
    gold_root: Path = typer.Option(
        DEFAULT_GOLD_ROOT, file_okay=False, help="Gold CSV root directory."
    ),
    rtol: float = typer.Option(
        1e-6,
        help="Deprecated compatibility option. Official-style evaluation uses Decimal rounding.",
    ),
    atol: float = typer.Option(
        1e-6,
        help="Deprecated compatibility option. Official-style evaluation uses Decimal rounding.",
    ),
    unordered: bool = typer.Option(True, help="Report official-style column signature metrics."),
    preview_rows: int = typer.Option(5, min=0, help="Rows included in JSON previews."),
) -> None:
    """Evaluate prediction.csv files against gold.csv files for a completed run."""
    _evaluate_run_and_print(
        run_dir=run_dir,
        gold_root=gold_root,
        rtol=rtol,
        atol=atol,
        unordered=unordered,
        preview_rows=preview_rows,
    )


def main() -> None:
    app()
