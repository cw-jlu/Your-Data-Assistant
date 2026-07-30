"""Run exp_145_phase2_best on the Phase 2 demo set and score it.

This is the fixed-profile successor to exp_144 run 008. No EXP144_* flags are
required from the caller; exp_145 fixes the promoted Phase 2 profile in code.

Operational guardrails:
- Full/resume benchmarks must use timeout=6000s. timeout=900s is too short for
  source-router_008 reproduction; several valid tasks took >900s in the best run.
- Do not pass --limit for an unattended full/resume benchmark. --limit intentionally
  processes only that many remaining tasks and then exits normally.
- Cloudflare 530/tunnel failures are API outage artifacts, not score measurements.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv

load_dotenv(REPO / ".env")

from kobushi_core.eval.csv_compare import EvaluationOptions, _evaluate_task

from experiments.exp_145_phase2_best.config import (
    AgentConfig,
    AppConfig,
    DatasetConfig,
    RunConfig,
)
from experiments.exp_145_phase2_best.runner import run_benchmark


DATA_ROOT = REPO / "data" / "phase2_demo" / "demo_samples_phase2"
INPUT_ROOT = DATA_ROOT / "input"
GOLD_ROOT = DATA_ROOT / "output"


def _official_score(pred_csv: Path, gold_csv: Path) -> float:
    if not gold_csv.is_file() or not pred_csv.is_file():
        return 0.0
    try:
        ev = _evaluate_task(
            task_id=pred_csv.parent.name,
            prediction_path=pred_csv,
            gold_path=gold_csv,
            options=EvaluationOptions(),
        )
        return float(ev.official_score_lambda_0_5)
    except Exception as exc:
        print(f"  [score err] {pred_csv.parent.name}: {exc}", flush=True)
        return 0.0


def _make_config(*, run_id: str, workers: int, timeout: int) -> AppConfig:
    headers = {}
    cf_id = os.environ.get("CF_ACCESS_CLIENT_ID", "")
    cf_secret = os.environ.get("CF_ACCESS_CLIENT_SECRET", "")
    if cf_id:
        headers["CF-Access-Client-Id"] = cf_id
    if cf_secret:
        headers["CF-Access-Client-Secret"] = cf_secret

    return AppConfig(
        dataset=DatasetConfig(root_path=INPUT_ROOT),
        agent=AgentConfig(
            model=os.environ.get("MODEL_NAME", "qwen3.5-35b-a3b"),
            api_base=os.environ.get("MODEL_API_URL") or os.environ.get("AGENT_API_BASE", ""),
            api_key=os.environ.get("MODEL_API_KEY") or os.environ.get("AGENT_API_KEY", ""),
            extra_headers=headers,
            max_steps=64,
            temperature=0.6,
        ),
        run=RunConfig(
            output_dir=REPO / "artifacts" / "runs",
            run_id=run_id,
            max_workers=workers,
            task_timeout_seconds=timeout,
        ),
    )


def _score_run(run_dir: Path) -> dict:
    results = []
    for task_dir in sorted(run_dir.glob("task_*"), key=lambda p: int(p.name.split("_")[1])):
        pred = task_dir / "prediction.csv"
        score = _official_score(pred, GOLD_ROOT / task_dir.name / "gold.csv")
        results.append(
            {
                "tid": task_dir.name,
                "score": score,
                "has_prediction": pred.is_file(),
            }
        )
    mean = sum(r["score"] for r in results) / len(results) if results else 0.0
    return {
        "n_tasks": len(results),
        "mean_score": mean,
        "n_perfect": sum(1 for r in results if r["score"] >= 0.999),
        "n_zero": sum(1 for r in results if r["score"] < 0.01),
        "results": results,
    }


def _compare_runs(summaries: list[dict]) -> dict:
    if len(summaries) < 2:
        return {}
    by_tid = []
    left = {r["tid"]: r for r in summaries[0]["results"]}
    right = {r["tid"]: r for r in summaries[1]["results"]}
    for tid in sorted(set(left) & set(right), key=lambda x: int(x.split("_")[1])):
        s1 = left[tid]["score"]
        s2 = right[tid]["score"]
        if abs(s1 - s2) > 1e-9:
            by_tid.append({"tid": tid, "run_1": s1, "run_2": s2, "delta": s2 - s1})
    means = [s["mean_score"] for s in summaries]
    return {
        "mean_scores": means,
        "mean_delta_run2_minus_run1": means[1] - means[0],
        "abs_mean_delta": abs(means[1] - means[0]),
        "n_task_flips": len(by_tid),
        "task_score_changes": by_tid,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=6000)
    ap.add_argument("--run-prefix", default="exp_145_phase2_best")
    ap.add_argument(
        "--resume-run-id",
        default=None,
        help="resume an existing run directory, skipping tasks with prediction.csv",
    )
    ap.add_argument("--tasks", type=str, default=None, help="comma-separated task ids for smoke runs")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    selected_tasks = None
    if args.tasks:
        selected_tasks = frozenset(t.strip() for t in args.tasks.split(",") if t.strip())

    summaries = []
    started = time.time()
    for i in range(1, args.repeats + 1):
        resume_completed: frozenset[str] = frozenset()
        allow_existing_run_dir = False
        run_id = args.resume_run_id or f"{args.run_prefix}_{i:03d}"
        run_dir = REPO / "artifacts" / "runs" / run_id
        if args.resume_run_id:
            allow_existing_run_dir = True
            resume_completed = frozenset(
                task_dir.name
                for task_dir in run_dir.glob("task_*")
                if (task_dir / "prediction.csv").is_file()
            )
        else:
            suffix = i
            while run_dir.exists():
                suffix += 1
                run_id = f"{args.run_prefix}_{suffix:03d}"
                run_dir = REPO / "artifacts" / "runs" / run_id

        print(f"=== exp145 repeat {i}/{args.repeats}: {run_id} ===", flush=True)
        print(f"workers={args.workers}, timeout={args.timeout}s", flush=True)
        if args.resume_run_id:
            print(
                f"resume: skip {len(resume_completed)} completed tasks with prediction.csv",
                flush=True,
            )

        done = len(resume_completed)

        def progress(artifact) -> None:
            nonlocal done
            done += 1
            mark = "ok" if artifact.succeeded else "ng"
            print(f"[{done:3d}] {mark} {artifact.task_id}", flush=True)

        cfg = _make_config(run_id=run_id, workers=args.workers, timeout=args.timeout)
        run_dir, _ = run_benchmark(
            config=cfg,
            limit=args.limit,
            progress_callback=progress,
            exclude_task_ids=resume_completed,
            include_task_ids=selected_tasks,
            allow_existing_run_dir=allow_existing_run_dir,
        )
        summary = _score_run(run_dir)
        summary.update(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "workers": args.workers,
                "timeout": args.timeout,
            }
        )
        (run_dir / "score_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
        print(
            f"=== scored {run_id}: mean={summary['mean_score']:.4f} "
            f"perfect={summary['n_perfect']}/{summary['n_tasks']} "
            f"zero={summary['n_zero']} ===",
            flush=True,
        )
        summaries.append(summary)

    aggregate = {
        "experiment": "exp_145_phase2_best",
        "repeats": args.repeats,
        "elapsed_minutes": (time.time() - started) / 60,
        "runs": [
            {
                "run_id": s["run_id"],
                "run_dir": s["run_dir"],
                "mean_score": s["mean_score"],
                "n_tasks": s["n_tasks"],
                "n_perfect": s["n_perfect"],
                "n_zero": s["n_zero"],
            }
            for s in summaries
        ],
        "comparison": _compare_runs(summaries),
    }
    out = REPO / "artifacts" / "runs" / f"{args.run_prefix}_variance_summary.json"
    out.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n")
    print(f"=== aggregate saved: {out} ===", flush=True)


if __name__ == "__main__":
    main()
