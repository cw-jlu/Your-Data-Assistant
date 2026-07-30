"""exp_139 math advisor bench using SUBMISSION-IDENTICAL pipeline.

Differences from math_advisor_full50.py:
- Delegates to `experiments.exp_139_raw_plus_outline.runner._run_single_task_with_timeout`
  which uses **multiprocessing.spawn × 3 parallel attempts** (= exactly what
  submission/main.py runs in production).
- Each attempt runs in its own Python subprocess with its own DuckDB connection
  → NO competition / NO conflict.
- Math advisor formula injection happens inside each subprocess via
  exp_139_raw_plus_outline/preamble.py (= ad-hoc bench-level injection removed).
- Scoring uses kobushi_core.eval._evaluate_task (= official column-multiset λ=0.5).
- Concurrency: MAX_WORKERS=4 tasks × 3 attempt processes each = 12 vLLM streams.

Usage:
    set -a && source .env && set +a && uv run python -u scripts/math_advisor_bench_parallel.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

from kobushi_core.eval.csv_compare import _evaluate_task, EvaluationOptions
from kobushi_core.benchmark.dataset import DABenchPublicDataset

# Use exp_139 module's runner and config (= same as submission/main.py)
from experiments.exp_139_raw_plus_outline.runner import _run_single_task_with_timeout
from experiments.exp_139_raw_plus_outline.config import load_app_config, AppConfig

OPTS = EvaluationOptions()
N_ATTEMPTS_NOTE = "(= configured in exp_139.runner._ATTEMPT_TEMPS = (0.6, 0.6, 0.6))"
MAX_WORKERS = 4

# Output dir is allocated inside main() to avoid spawn re-execution duplication.
# (Each spawned attempt subprocess re-imports this module; without guarding the
# OUT allocation it'd create N+1 new dirs per task.)
OUT: Path | None = None  # set in main()


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _build_app_config() -> AppConfig:
    """Load AppConfig from yaml (= already max_steps=64 + timeout=900 + max_workers=4).
    AppConfig is frozen; .env supplies api_base/api_key via load_app_config."""
    config_yaml = REPO / "src" / "experiments" / "exp_139_raw_plus_outline" / "config.yaml"
    return load_app_config(config_yaml)


def _score_csv_official(pred_path: Path, gold_path: Path) -> float:
    if not pred_path.exists() or not gold_path.exists():
        return 0.0
    e = _evaluate_task(
        task_id=str(pred_path.parent.name),
        prediction_path=pred_path,
        gold_path=gold_path,
        options=OPTS,
    )
    return float(e.official_score_lambda_0_5)


def _save_prediction(task_dir: Path, answer: dict | None):
    if not answer or not answer.get("rows"):
        return None
    pred = task_dir / "prediction.csv"
    columns = list(answer.get("columns") or [])
    rows = [list(r) for r in (answer.get("rows") or [])]
    with pred.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            w.writerow(r)
    return pred


def run_one(tid: str, config: AppConfig) -> dict:
    t0 = time.time()
    try:
        result = _run_single_task_with_timeout(task_id=tid, config=config)
    except Exception as e:
        return {"tid": tid, "score": 0.0, "elapsed_s": round(time.time()-t0, 1),
                "error": f"{type(e).__name__}: {str(e)[:200]}"}
    task_dir = OUT / tid
    task_dir.mkdir(parents=True, exist_ok=True)
    # Save per-attempt details if present
    (task_dir / "result.json").write_text(json.dumps({
        "tid": tid,
        "succeeded": result.get("succeeded"),
        "failure_reason": result.get("failure_reason"),
        "elapsed": round(time.time() - t0, 1),
        "answer": result.get("answer"),
        "vote_meta": {k: v for k, v in result.items() if k in ("temps", "n_attempts_succeeded", "vote_strategy")},
    }, default=str, indent=2))
    pred = _save_prediction(task_dir, result.get("answer"))
    if not pred:
        return {"tid": tid, "score": 0.0, "elapsed_s": round(time.time()-t0, 1),
                "error": "no answer"}
    gold = REPO / "data" / "public" / "output" / tid / "gold.csv"
    score = _score_csv_official(pred, gold)
    return {"tid": tid, "score": score, "elapsed_s": round(time.time()-t0, 1),
            "n_attempts": result.get("n_attempts_succeeded"),
            "cols": list((result.get("answer") or {}).get("columns") or []),
            "first_row": list((result.get("answer") or {}).get("rows") or [[]])[0] if (result.get("answer") or {}).get("rows") else None}


def main():
    global OUT
    _runs = REPO / "artifacts" / "runs"
    _i = 1
    while (_runs / f"exp_139_raw_plus_outline_{_i:03d}").exists():
        _i += 1
    OUT = _runs / f"exp_139_raw_plus_outline_{_i:03d}"
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[init] output dir: {OUT}", flush=True)

    config = _build_app_config()
    _log(f"=== exp_139 math advisor bench (submission-style: 4 tasks parallel × 3 attempts spawn, max_steps={config.agent.max_steps}) ===")
    ds = DABenchPublicDataset(root_dir=REPO/"data"/"public"/"input")
    tids = sorted(ds.list_task_ids())
    _log(f"  total tasks: {len(tids)}, output: {OUT}")

    results: list[dict] = []
    t_start = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(run_one, tid, config): tid for tid in tids}
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            results.append(r)
            tag = "ERR" if r.get("error") else "OK"
            print(f"  [{done:>2d}/{len(tids)}] {r['tid']}: score={r['score']:.2f} {tag} t={r['elapsed_s']}s", flush=True)

    elapsed = time.time() - t_start
    mean = sum(r["score"] for r in results) / max(1, len(results))
    perfect = sum(1 for r in results if r["score"] >= 0.99)
    zero = sum(1 for r in results if r["score"] < 0.01)
    _log(f"=== DONE: n={len(results)}, mean={mean:.4f}, perfect={perfect}, zero={zero}, elapsed={elapsed/60:.1f}min ===")
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str))
    (OUT / "summary.json").write_text(json.dumps({
        "n_tasks": len(results),
        "mean_score": mean,
        "n_perfect": perfect,
        "n_zero": zero,
        "elapsed_minutes": round(elapsed/60, 2),
        "config": {
            "max_steps": config.agent.max_steps,
            "task_timeout_seconds": config.run.task_timeout_seconds,
            "max_workers": MAX_WORKERS,
            "n_attempts_per_task": 3,
            "scorer": "kobushi_core.eval._evaluate_task (official λ=0.5)",
        }
    }, indent=2))
    _log(f"Saved: {OUT}/summary.json")


if __name__ == "__main__":
    main()
