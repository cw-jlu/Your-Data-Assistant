"""Benchmark Global Data Profile generation time with concurrent execution.

Picks 3 easy, 3 medium, 3 hard, and 1 extreme task (seed=42),
runs ONLY the global data profiling step in parallel (8 workers),
and reports timing breakdowns.

Usage:
    uv run python scripts/benchmark_global_profile.py
"""

import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "easy.yaml"
MAX_WORKERS = 8

console = Console()
_print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with _print_lock:
        console.print(*args, **kwargs)


def discover_tasks(input_root: Path) -> dict[str, list[str]]:
    by_difficulty: dict[str, list[str]] = {}
    for task_dir in sorted(input_root.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith("task_"):
            continue
        task_json = task_dir / "task.json"
        if not task_json.exists():
            continue
        data = json.loads(task_json.read_text(encoding="utf-8"))
        difficulty = data.get("difficulty", "unknown")
        by_difficulty.setdefault(difficulty, []).append(task_dir.name)
    return by_difficulty


def pick_tasks(by_difficulty: dict[str, list[str]]) -> dict[str, list[str]]:
    picks: dict[str, list[str]] = {}
    rng = random.Random(42)
    for difficulty, count in [("easy", 3), ("medium", 3), ("hard", 3), ("extreme", 1)]:
        pool = sorted(by_difficulty.get(difficulty, []))
        picks[difficulty] = pool if len(pool) <= count else rng.sample(pool, count)
    return picks


def run_one_task(
    task_id: str,
    difficulty: str,
    context_dir: Path,
    config: Any,
) -> dict[str, Any]:
    """Run global data profiling for one task. Each thread creates its own model."""
    from data_agent_baseline.agents.model import create_chat_model
    from data_agent_baseline.inspectors.data_understanding_agent import DataUnderstandingAgent
    from data_agent_baseline.inspectors.semantic_catalog import build_semantic_catalog
    from data_agent_baseline.benchmark.schema import PublicTask

    # Each worker gets its own model + agent instance
    model = create_chat_model(
        model=config.agent.model,
        api_base=config.agent.api_base,
        api_key=config.agent.api_key,
        api_key_env=config.agent.api_key_env,
        temperature=config.agent.temperature,
    )
    data_agent = DataUnderstandingAgent(model=model, config=config.data_inspector)

    # Phase A: catalog
    t0 = perf_counter()
    catalog = build_semantic_catalog(
        PublicTask(
            record=type("TaskRecord", (), {"task_id": task_id, "difficulty": "", "question": ""})(),
            assets=type("TaskAssets", (), {"task_dir": context_dir.parent, "context_dir": context_dir})(),
        ),
        budget=data_agent.config.sample_budget,
    )
    catalog_time = round(perf_counter() - t0, 3)

    schemas = catalog.get("schemas", [])
    table_count = sum(1 for s in schemas if s.get("kind") in ("table", "csv", "json", "sqlite"))
    doc_count = sum(1 for s in schemas if s.get("kind") == "document")
    total_fields = sum(len(s.get("fields", [])) for s in schemas)

    # Phase B: LLM profiling call
    llm_time: float | None = None
    method: str = "llm"
    profile_length: int = 0
    error: str | None = None

    t1 = perf_counter()
    try:
        from data_agent_baseline.token_utils import count_tokens

        full_profile, _catalog = data_agent.explore_data_globally(
            context_dir=context_dir,
            task_id=task_id,
        )
        total_time = round(perf_counter() - t1, 3)
        profile_length = count_tokens(full_profile)
        if not full_profile.strip():
            method = "empty"
    except Exception as exc:
        total_time = round(perf_counter() - t1, 3)
        method = "failed"
        error = str(exc)
        profile_length = 0

    # Estimate LLM time: total minus (catalog was already built above, but
    # explore_data_globally rebuilds catalog internally too, so we report
    # the total and the catalog cost separately as measured)
    return {
        "task_id": task_id,
        "difficulty": difficulty,
        "method": method,
        "catalog_time_s": catalog_time,
        "total_time_s": total_time,
        "profile_tokens": profile_length,
        "table_count": table_count,
        "doc_count": doc_count,
        "total_fields": total_fields,
        "error": error,
    }


def main() -> None:
    from data_agent_baseline.config import load_app_config

    config = load_app_config(CONFIG_PATH)

    console.print("[bold]Global Data Profile Benchmark (Concurrent)[/bold]")
    console.print(f"Config: {CONFIG_PATH}")
    console.print(f"Model: {config.agent.model}")
    console.print(f"Workers: {MAX_WORKERS}\n")

    input_root = config.dataset.root_path
    by_difficulty = discover_tasks(input_root)

    console.print("Available tasks by difficulty:")
    for diff, tasks in sorted(by_difficulty.items()):
        console.print(f"  {diff}: {len(tasks)} tasks")
    console.print()

    picks = pick_tasks(by_difficulty)
    ordered_tasks: list[tuple[str, str]] = []
    for difficulty in ["easy", "medium", "hard", "extreme"]:
        for tid in picks.get(difficulty, []):
            ordered_tasks.append((tid, difficulty))

    console.print("Selected tasks:")
    for tid, diff in ordered_tasks:
        console.print(f"  {tid} ({diff})")
    console.print()

    # --- Run all tasks concurrently ---
    results: list[dict[str, Any]] = []
    overall_start = perf_counter()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                run_one_task,
                task_id,
                difficulty,
                input_root / task_id / "context",
                config,
            ): (task_id, difficulty)
            for task_id, difficulty in ordered_tasks
        }

        for future in as_completed(futures):
            task_id, difficulty = futures[future]
            try:
                result = future.result()
                results.append(result)
                status = "FAIL" if result.get("error") else "OK"
                safe_print(
                    f"  [{status}] {result['task_id']} ({result['difficulty']}) | "
                    f"catalog={result['catalog_time_s']:.1f}s | "
                    f"total={result['total_time_s']:.1f}s | "
                    f"profile={result['profile_tokens']} tokens"
                )
            except Exception as exc:
                import traceback
                safe_print(f"  [FAIL] {task_id} ({difficulty}): {exc}")
                results.append({
                    "task_id": task_id,
                    "difficulty": difficulty,
                    "error": str(exc),
                })

    wall_clock = round(perf_counter() - overall_start, 1)

    # Sort by difficulty then task_id for report
    diff_order = {"easy": 0, "medium": 1, "hard": 2, "extreme": 3}
    results.sort(key=lambda r: (diff_order.get(r.get("difficulty", ""), 99), r.get("task_id", "")))

    # --- Report ---
    console.print()
    console.print("[bold]" + "=" * 80)
    console.print("  GLOBAL DATA PROFILE BENCHMARK REPORT (CONCURRENT)")
    console.print("=" * 80)

    summary_table = Table(title="Per-Task Results")
    summary_table.add_column("Task", style="cyan")
    summary_table.add_column("Difficulty", style="magenta")
    summary_table.add_column("Method")
    summary_table.add_column("Catalog (s)", justify="right")
    summary_table.add_column("Total (s)", justify="right")
    summary_table.add_column("Profile (tokens)", justify="right")
    summary_table.add_column("Tables", justify="right")
    summary_table.add_column("Docs", justify="right")
    summary_table.add_column("Fields", justify="right")

    for r in results:
        if r.get("error"):
            summary_table.add_row(r["task_id"], r["difficulty"], "FAILED", "-", "-", "-", "-", "-", "-")
        else:
            summary_table.add_row(
                r["task_id"],
                r["difficulty"],
                r["method"],
                f"{r['catalog_time_s']:.2f}",
                f"{r['total_time_s']:.2f}",
                str(r["profile_tokens"]),
                str(r["table_count"]),
                str(r["doc_count"]),
                str(r["total_fields"]),
            )

    console.print(summary_table)

    # Aggregate
    console.print()
    agg_table = Table(title="Aggregate by Difficulty")
    agg_table.add_column("Difficulty", style="magenta")
    agg_table.add_column("Tasks")
    agg_table.add_column("Avg Catalog (s)", justify="right")
    agg_table.add_column("Avg Total (s)", justify="right")
    agg_table.add_column("Min Total (s)", justify="right")
    agg_table.add_column("Max Total (s)", justify="right")

    by_diff: dict[str, list[dict]] = {}
    for r in results:
        if "error" not in r:
            by_diff.setdefault(r["difficulty"], []).append(r)

    for difficulty in ["easy", "medium", "hard", "extreme"]:
        group = by_diff.get(difficulty, [])
        if not group:
            continue
        n = len(group)
        avg_cat = sum(r["catalog_time_s"] for r in group) / n
        avg_total = sum(r["total_time_s"] for r in group) / n
        agg_table.add_row(
            difficulty,
            str(n),
            f"{avg_cat:.2f}",
            f"{avg_total:.2f}",
            f"{min(r['total_time_s'] for r in group):.2f}",
            f"{max(r['total_time_s'] for r in group):.2f}",
        )

    console.print(agg_table)

    # Overall
    valid = [r for r in results if "error" not in r]
    if valid:
        totals = [r["total_time_s"] for r in valid]
        catalogs = [r["catalog_time_s"] for r in valid]
        console.print()
        console.print(f"[bold]Overall (n={len(valid)}):[/bold]")
        console.print(f"  Avg catalog time: {sum(catalogs) / len(catalogs):.2f}s")
        console.print(f"  Avg total time (per task): {sum(totals) / len(totals):.2f}s")
        console.print(f"  Min total: {min(totals):.2f}s")
        console.print(f"  Max total: {max(totals):.2f}s")
        console.print(f"  [bold]Wall-clock (8 concurrent): {wall_clock:.1f}s[/bold]")

    # Save
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = PROJECT_ROOT / "artifacts" / "runs" / f"profile_bench_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": timestamp,
        "config": str(CONFIG_PATH),
        "model": config.agent.model,
        "max_workers": MAX_WORKERS,
        "wall_clock_s": wall_clock,
        "picked_tasks": picks,
        "results": results,
    }
    report_path = report_dir / "benchmark_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    console.print(f"\n[green]Report saved to: {report_path}[/green]")


if __name__ == "__main__":
    main()
