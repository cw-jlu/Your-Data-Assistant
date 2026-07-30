"""Run task_38 N times in parallel and compute accuracy statistics.

Usage:
    uv run python scripts/benchmark_task_38.py
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from rich.console import Console
from rich.table import Table

from data_agent_baseline.config import load_app_config
from data_agent_baseline.run.runner import TaskRunArtifacts, run_single_task, _write_json
from data_agent_baseline.scoring import score_run_outputs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLD_ROOT = PROJECT_ROOT / "data" / "output"
CONFIG_PATH = PROJECT_ROOT / "configs" / "easy.yaml"

console = Console()


def run_trial(
    task_id: str,
    config,
    bench_dir: Path,
    trial_index: int,
) -> dict:
    trial_dir = bench_dir / f"{trial_index:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    started_at = perf_counter()
    try:
        artifact = run_single_task(
            task_id=task_id,
            config=config,
            run_output_dir=trial_dir,
            prediction_output_root=trial_dir,
        )
        elapsed = round(perf_counter() - started_at, 3)
    except BaseException as exc:
        import traceback

        elapsed = round(perf_counter() - started_at, 3)
        return {
            "trial_index": trial_index,
            "ok": False,
            "error": f"{exc}\n{traceback.format_exc()}",
            "elapsed_seconds": elapsed,
        }

    # Write a summary.json so the scorer can process this trial directory.
    summary = {
        "run_id": trial_dir.name,
        "output_layout": "run_dir",
        "run_output_dir": str(trial_dir),
        "prediction_output_root": str(trial_dir),
        "log_dir": None,
        "task_count": 1,
        "skipped_task_count": 0,
        "skipped_task_ids": [],
        "succeeded_task_count": 1 if artifact.succeeded else 0,
        "total_elapsed_seconds": elapsed,
        "max_workers": 1,
        "task_timeout_seconds": config.run.task_timeout_seconds,
        "max_steps": config.agent.max_steps,
        "temperature": config.agent.temperature,
        "enable_data_inspector": config.agent.enable_data_inspector,
        "enable_answer_validator": config.agent.enable_answer_validator,
        "data_inspector": {
        },
        "tasks": [artifact.to_dict()],
    }
    _write_json(trial_dir / "summary.json", summary)

    return {
        "trial_index": trial_index,
        "ok": True,
        "artifact": artifact,
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    config = load_app_config(CONFIG_PATH)

    N_TRIALS = 10
    MAX_WORKERS = 8
    TASK_ID = "task_38"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bench_dir = PROJECT_ROOT / "artifacts" / "runs" / f"task_38_bench_{timestamp}"
    bench_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Benchmark: {TASK_ID} x {N_TRIALS}[/bold]")
    console.print(f"Config: {CONFIG_PATH}")
    console.print(f"Output: {bench_dir}")
    console.print(f"Workers: {MAX_WORKERS}\n")

    # --- Phase 1: run all trials in parallel ---
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(run_trial, TASK_ID, config, bench_dir, i): i
            for i in range(N_TRIALS)
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            idx = result["trial_index"]
            status = "OK" if result["ok"] else "FAIL"
            elapsed = result.get("elapsed_seconds", 0)
            succeeded = result.get("artifact") and result["artifact"].succeeded
            detail = "succeeded" if succeeded else "no_prediction"
            console.print(f"  Trial {idx:03d}: {status} | {elapsed:.1f}s | {detail}")

    results.sort(key=lambda r: r["trial_index"])

    # --- Phase 2: score each trial ---
    console.print("\n[bold]Scoring each trial...[/bold]")

    scores: list[dict] = []
    for r in results:
        trial_dir = bench_dir / f"{r['trial_index']:03d}"
        try:
            summary = score_run_outputs(
                run_output_dir=trial_dir,
                gold_root=GOLD_ROOT,
            )
            ts = summary.tasks[0] if summary.tasks else None
            scores.append({
                "trial_index": r["trial_index"],
                "ok": bool(r["ok"]),
                "recall": ts.recall if ts else 0.0,
                "redundancy_rate": ts.redundancy_rate if ts else 0.0,
                "proxy_score": ts.primary_proxy_score if ts else 0.0,
                "full_cover": bool(ts.full_cover) if ts else False,
                "succeeded": bool(ts.succeeded) if ts else False,
                "covered_gold_columns": ts.covered_gold_columns if ts else 0,
                "gold_column_count": ts.gold_column_count if ts else 0,
                "prediction_column_count": ts.prediction_column_count if ts else 0,
            })
        except Exception as exc:
            scores.append({
                "trial_index": r["trial_index"],
                "ok": False,
                "recall": 0.0,
                "redundancy_rate": 0.0,
                "proxy_score": 0.0,
                "full_cover": False,
                "succeeded": False,
                "covered_gold_columns": 0,
                "gold_column_count": 0,
                "prediction_column_count": 0,
                "score_error": str(exc),
            })

    # --- Phase 3: aggregate & display ---
    console.print("\n[bold]Results Summary[/bold]")

    recall_values = [s["recall"] for s in scores]
    proxy_values = [s["proxy_score"] for s in scores]
    redundancy_values = [s["redundancy_rate"] for s in scores]
    full_cover_count = sum(1 for s in scores if s["full_cover"])
    success_count = sum(1 for s in scores if s["succeeded"])

    summary_table = Table(title=f"{TASK_ID} Benchmark (N={N_TRIALS})")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    n = len(scores)
    summary_table.add_row("Mean Recall", f"{sum(recall_values) / n:.4f}")
    summary_table.add_row("Mean Proxy Score (λ=0.1)", f"{sum(proxy_values) / n:.4f}")
    summary_table.add_row("Mean Redundancy Rate", f"{sum(redundancy_values) / n:.4f}")
    summary_table.add_row(
        "Full Cover Rate",
        f"{full_cover_count}/{N_TRIALS} ({full_cover_count / N_TRIALS * 100:.1f}%)",
    )
    summary_table.add_row(
        "Success Rate",
        f"{success_count}/{N_TRIALS} ({success_count / N_TRIALS * 100:.1f}%)",
    )
    summary_table.add_row("Min Recall", f"{min(recall_values):.4f}")
    summary_table.add_row("Max Recall", f"{max(recall_values):.4f}")
    summary_table.add_row("Recall Std Dev", f"{(_stdev(recall_values)):.4f}")

    console.print(summary_table)

    # Per-trial detail table
    detail_table = Table(title="Per-Trial Details")
    detail_table.add_column("Trial", style="dim")
    detail_table.add_column("Recall")
    detail_table.add_column("Redundancy")
    detail_table.add_column("Proxy")
    detail_table.add_column("Cols", justify="right")
    detail_table.add_column("FullCover")
    detail_table.add_column("OK")

    for s in scores:
        cols_info = f"{s['covered_gold_columns']}/{s['gold_column_count']}"
        detail_table.add_row(
            f"{s['trial_index']:03d}",
            f"{s['recall']:.4f}",
            f"{s['redundancy_rate']:.4f}",
            f"{s['proxy_score']:.4f}",
            cols_info,
            "Y" if s["full_cover"] else "N",
            "Y" if s["succeeded"] else "N",
        )

    console.print(detail_table)

    # Save aggregate results
    aggregate = {
        "task_id": TASK_ID,
        "config": str(CONFIG_PATH),
        "n_trials": N_TRIALS,
        "max_workers": MAX_WORKERS,
        "mean_recall": sum(recall_values) / n,
        "mean_proxy_score": sum(proxy_values) / n,
        "mean_redundancy_rate": sum(redundancy_values) / n,
        "full_cover_count": full_cover_count,
        "success_count": success_count,
        "min_recall": min(recall_values),
        "max_recall": max(recall_values),
        "recall_std": _stdev(recall_values),
        "recall_values": recall_values,
        "proxy_scores": proxy_values,
        "redundancy_values": redundancy_values,
        "elapsed_seconds_per_trial": [r.get("elapsed_seconds", 0) for r in results],
        "per_trial": scores,
    }
    aggregate_path = bench_dir / "aggregate.json"
    aggregate_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    console.print(f"\n[green]Aggregate results saved to: {aggregate_path}[/green]")
    console.print(f"[green]Trial outputs in: {bench_dir}[/green]")


def _stdev(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return variance**0.5


if __name__ == "__main__":
    main()
