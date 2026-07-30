"""exp_140 (= BIRD-ICL + cell VALUE HINTS) on gpuhost FP8, 3-attempt vote, workers=4.

Submission-identical pipeline: each task spawns 3 attempt subprocesses (vote)
via exp_140_agentar_icl.runner._run_single_task_with_timeout, and 4 tasks run
concurrently in the outer thread pool.

Cloned from scripts/math_advisor_bench_parallel.py — only changes:
  - exp_137_math_advisor → exp_140_agentar_icl
  - output dir prefix: exp_140_agentar_NNN
"""
from __future__ import annotations
import csv, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

from kobushi_core.eval.csv_compare import _evaluate_task, EvaluationOptions
from kobushi_core.benchmark.dataset import DABenchPublicDataset
from experiments.exp_140_agentar_icl.runner import _run_single_task_with_timeout
from experiments.exp_140_agentar_icl.config import load_app_config, AppConfig

OPTS = EvaluationOptions()
MAX_WORKERS = 2   # = 4→2 to reduce gpuhost concurrent load (= 12→6 streams)
OUT: Path | None = None


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _build_app_config() -> AppConfig:
    config_yaml = REPO / "src" / "experiments" / "exp_140_agentar_icl" / "config.yaml"
    return load_app_config(config_yaml)


def _score(pred_path: Path, gold_path: Path) -> float:
    if not pred_path.exists() or not gold_path.exists():
        return 0.0
    e = _evaluate_task(task_id=str(pred_path.parent.name),
                       prediction_path=pred_path, gold_path=gold_path, options=OPTS)
    return float(e.official_score_lambda_0_5)


def _save_pred(task_dir: Path, answer: dict | None):
    if not answer or not answer.get("rows"):
        return None
    pred = task_dir / "prediction.csv"
    with pred.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(answer.get("columns") or []))
        for r in answer.get("rows") or []:
            w.writerow(list(r))
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
    # Full backbone → trace.json (= per_attempt + steps + majority_3x_metadata
    # + preamble fields). Required for offline vote analysis.
    (task_dir / "trace.json").write_text(json.dumps(result, default=str, indent=2))
    # Slim result.json with key fields surfaced (= for quick diff / scoring).
    (task_dir / "result.json").write_text(json.dumps({
        "tid": tid,
        "succeeded": result.get("succeeded"),
        "failure_reason": result.get("failure_reason"),
        "elapsed": round(time.time() - t0, 1),
        "answer": result.get("answer"),
        "vote_meta": result.get("majority_3x_metadata", {}),
        "per_attempt_summary": [
            {
                "attempt_index": a.get("attempt_index"),
                "succeeded": a.get("succeeded"),
                "n_steps": a.get("n_steps"),
                "terminal_sql": a.get("terminal_sql"),
                "n_cols": len((a.get("answer") or {}).get("columns") or []),
                "n_rows": len((a.get("answer") or {}).get("rows") or []),
            }
            for a in (result.get("per_attempt") or [])
        ],
    }, default=str, indent=2))
    pred = _save_pred(task_dir, result.get("answer"))
    if not pred:
        return {"tid": tid, "score": 0.0, "elapsed_s": round(time.time()-t0, 1),
                "error": "no answer"}
    gold = REPO / "data" / "public" / "output" / tid / "gold.csv"
    score = _score(pred, gold)
    return {"tid": tid, "score": score, "elapsed_s": round(time.time()-t0, 1),
            "n_attempts_succeeded": result.get("majority_3x_metadata", {}).get("n_succeeded")}


def main():
    global OUT
    _runs = REPO / "artifacts" / "runs"
    _i = 1
    while (_runs / f"exp_140_agentar_{_i:03d}").exists():
        _i += 1
    OUT = _runs / f"exp_140_agentar_{_i:03d}"
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[init] output dir: {OUT}", flush=True)

    config = _build_app_config()
    _log(f"=== exp_140 gpuhost bench (workers={MAX_WORKERS} × 3-attempt spawn, max_steps={config.agent.max_steps}) ===")
    ds = DABenchPublicDataset(root_dir=REPO/"data"/"public"/"input")
    tids = sorted(ds.list_task_ids())
    _log(f"tasks: {len(tids)}")

    results = []
    t_all = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(run_one, tid, config): tid for tid in tids}
        for i, fut in enumerate(as_completed(futures), 1):
            tid = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"tid": tid, "score": 0.0,
                     "error": f"outer {type(e).__name__}: {str(e)[:200]}"}
            results.append(r)
            tag = "✓" if r["score"] >= 0.99 else ("✗" if r["score"] < 0.01 else "○")
            running = sum(x["score"] for x in results) / len(results)
            _log(f"  [{i:2d}/{len(tids)}] {tag} {r['tid']}: score={r['score']:.2f} "
                 f"t={r.get('elapsed_s', 0):.0f}s  running={running:.4f}")

    total = time.time() - t_all
    mean = sum(r["score"] for r in results) / len(results)
    perfect = sum(1 for r in results if r["score"] >= 0.99)
    zero = sum(1 for r in results if r["score"] < 0.01)
    _log(f"=== DONE n={len(results)}, mean={mean:.4f}, perfect={perfect}, zero={zero}, elapsed={total/60:.1f}min ===")

    (OUT / "summary.json").write_text(json.dumps({
        "n_tasks": len(results),
        "mean_score": mean,
        "n_perfect": perfect,
        "n_zero": zero,
        "elapsed_minutes": round(total/60, 1),
        "config": {
            "endpoint": "gpuhost FP8 (https://gpu-host.internal/v1)",
            "experiment": "exp_140_agentar_icl",
            "n_attempts": 3,
            "max_workers": MAX_WORKERS,
            "max_steps": config.agent.max_steps,
        },
    }, indent=2))
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
