#!/usr/bin/env python3
"""Run a Phase 2 partial benchmark by skipping stable-perfect tasks.

This is a lightweight wrapper around the existing phase2_expXXX_ablation.py
scripts. It does not change experiment code; it only supplies a task subset via
--tasks and annotates the resulting summary.json with a full-60 projection.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
RUNS_ROOT = REPO / "artifacts" / "runs"
DATA_ROOT = REPO / "data" / "phase2_demo" / "demo_samples_phase2" / "input"

# Conservative skip set: perfect across recent major full-60 samples, including
# degraded exp160 runs. Use this as the default fast bench skip set.
STABLE11_SKIP = (
    "task_4",
    "task_9",
    "task_13",
    "task_16",
    "task_22",
    "task_28",
    "task_29",
    "task_30",
    "task_36",
    "task_42",
    "task_54",
)

# Legacy aggressive skip set for faster iteration. Kept for comparison only:
# several tasks in this set are not strictly perfect across recent samples.
PARTIAL19_SKIP = (
    "task_4",
    "task_9",
    "task_10",
    "task_11",
    "task_12",
    "task_13",
    "task_16",
    "task_19",
    "task_20",
    "task_21",
    "task_22",
    "task_23",
    "task_28",
    "task_29",
    "task_30",
    "task_42",
    "task_54",
    "task_56",
    "task_60",
)

SKIP_SETS = {
    "stable11": STABLE11_SKIP,
    "partial19": PARTIAL19_SKIP,
}


def _task_key(task_id: str) -> int:
    return int(task_id.split("_", 1)[1])


def _all_task_ids() -> list[str]:
    if DATA_ROOT.exists():
        return sorted(
            [p.name for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("task_")],
            key=_task_key,
        )
    return [f"task_{i}" for i in range(1, 61)]


def _csv(items: list[str] | tuple[str, ...]) -> str:
    return ",".join(items)


def _script_for_experiment(exp: str) -> Path:
    exp = exp.lower().removeprefix("exp")
    path = REPO / "scripts" / f"phase2_exp{int(exp)}_ablation.py"
    if not path.exists():
        raise SystemExit(f"experiment script not found: {path}")
    return path


def _summaries_for_exp(exp: str) -> set[Path]:
    return set(RUNS_ROOT.glob(f"exp_{int(exp)}_*/summary.json"))


def _annotate_summary(
    path: Path,
    *,
    run_tasks: list[str],
    assumed_skip_tasks: list[str],
    reference_skip_tasks: list[str],
    skip_set_name: str,
    mode: str,
) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results") or []
    partial_sum = sum(float(r.get("score", 0.0)) for r in results)
    skipped_assumed_sum = float(len(assumed_skip_tasks))
    projected_mean = None
    if assumed_skip_tasks:
        projected_mean = (partial_sum + skipped_assumed_sum) / (
            len(run_tasks) + len(assumed_skip_tasks)
        )
    data["partial_bench"] = {
        "name": f"{skip_set_name}_{mode}",
        "mode": mode,
        "skip_set": skip_set_name,
        "run_task_count": len(run_tasks),
        "assumed_skipped_task_count": len(assumed_skip_tasks),
        "reference_skip_task_count": len(reference_skip_tasks),
        "run_tasks": run_tasks,
        "skipped_tasks_assumed_1": assumed_skip_tasks,
        "reference_skip_tasks": reference_skip_tasks,
        "partial_mean": data.get("mean_score"),
        "projected_full60_mean_if_skipped_are_1": projected_mean,
        "note": (
            "Fast-iteration metric only. In partial mode, the projected full-60 "
            "score assumes all skipped stable-perfect tasks would score 1.0 in "
            "this run. In run_skipped_only mode, use partial_mean as a regression "
            "check for the skipped controls."
        ),
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="159", help="experiment number, e.g. 159 or 160")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n-attempts", type=int, default=1)
    parser.add_argument("--skip-set", choices=sorted(SKIP_SETS), default="stable11")
    parser.add_argument(
        "--run-skipped-only",
        action="store_true",
        help="run only the selected skip set as a regression check",
    )
    parser.add_argument("--dry-run", action="store_true", help="print command and task lists only")
    args, passthrough = parser.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    script = _script_for_experiment(args.experiment)
    all_tasks = _all_task_ids()
    reference_skip_tasks = sorted(SKIP_SETS[args.skip_set], key=_task_key)
    skip_set = set(reference_skip_tasks)
    if args.run_skipped_only:
        run_tasks = reference_skip_tasks
        assumed_skip_tasks: list[str] = []
        mode = "run_skipped_only"
    else:
        run_tasks = [tid for tid in all_tasks if tid not in skip_set]
        assumed_skip_tasks = reference_skip_tasks
        mode = "partial"
    if len(all_tasks) != 60:
        raise SystemExit(f"expected 60 tasks, found {len(all_tasks)}")
    if len(reference_skip_tasks) != len(SKIP_SETS[args.skip_set]):
        raise SystemExit(f"skip-set invariant broken: {args.skip_set}")
    expected_run = len(reference_skip_tasks) if args.run_skipped_only else 60 - len(reference_skip_tasks)
    if len(run_tasks) != expected_run:
        raise SystemExit(
            f"{args.skip_set} invariant broken: skip={len(reference_skip_tasks)} "
            f"run={len(run_tasks)} expected_run={expected_run}"
        )

    cmd = [
        sys.executable,
        str(script),
        "--workers",
        str(args.workers),
        "--n-attempts",
        str(args.n_attempts),
        "--tasks",
        _csv(run_tasks),
        *passthrough,
    ]

    print("=== Phase 2 partial bench ===", flush=True)
    print(f"experiment=exp{int(args.experiment)} script={script.relative_to(REPO)}", flush=True)
    print(
        f"skip_set={args.skip_set} mode={mode} skipped={len(assumed_skip_tasks)} "
        f"reference_skip={len(reference_skip_tasks)} run={len(run_tasks)}",
        flush=True,
    )
    print("reference_skip_tasks=" + _csv(reference_skip_tasks), flush=True)
    print("skipped_tasks_assumed_1=" + _csv(assumed_skip_tasks), flush=True)
    print("run_tasks=" + _csv(run_tasks), flush=True)
    print("command=" + " ".join(cmd), flush=True)

    if args.dry_run:
        return 0

    before = _summaries_for_exp(args.experiment)
    rc = subprocess.call(cmd, cwd=REPO, env=os.environ.copy())
    after = _summaries_for_exp(args.experiment)
    new = sorted(after - before, key=lambda p: p.stat().st_mtime)
    if new:
        summary_path = new[-1]
        _annotate_summary(
            summary_path,
            run_tasks=run_tasks,
            assumed_skip_tasks=assumed_skip_tasks,
            reference_skip_tasks=reference_skip_tasks,
            skip_set_name=args.skip_set,
            mode=mode,
        )
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        projection = data["partial_bench"]["projected_full60_mean_if_skipped_are_1"]
        print(f"annotated {summary_path}", flush=True)
        if projection is None:
            print(f"regression_mean={data.get('mean_score'):.4f}", flush=True)
        else:
            print(
                f"partial_mean={data.get('mean_score'):.4f} "
                f"projected_full60={projection:.4f}",
                flush=True,
            )
    else:
        print("warning: no new summary.json found to annotate", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
