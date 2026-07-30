"""PoC: ReFoRCE-style 2-phase pipeline (= exp_116) with 3-attempt outer union.

Architecture:
  - Outer: 3 parallel attempts at temps [0.6, 0.6, 0.7] (= matches exp_109 style)
  - Inner: ReFoRCE pipeline = exploration + self_refine loop with self-consistency
  - Final: signature-vote across the 3 attempts' committed answers

5-task subset compared to control + sql_only (= exp_111) baselines.

Usage:
    uv run python scripts/test_reforce_loop_poc.py [--n_attempts 3] [--max_iter 5]
"""
from __future__ import annotations

import argparse
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
load_dotenv()

from kobushi_core.benchmark.dataset import DABenchPublicDataset
from kobushi_core.benchmark.schema import AnswerTable
from kobushi_core.eval.csv_compare import _evaluate_task, EvaluationOptions
from kobushi_core.model import OpenAIModelAdapter
from experiments.exp_116_reforce_loop.pipeline import run_pipeline


TASKS = ["task_25", "task_163", "task_169", "task_180", "task_38"]


def make_model(temperature: float) -> OpenAIModelAdapter:
    return OpenAIModelAdapter(
        model="qwen3.5-35b-a3b",
        api_base=os.environ["AGENT_API_BASE"],
        api_key=os.environ["AGENT_API_KEY"],
        temperature=temperature,
        extra_headers={
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
        presence_penalty=1.0,
    )


def _norm_value(v):
    if v is None: return None
    s = str(v).strip()
    try:
        return round(float(s), 3)
    except Exception:
        return s.lower()


def _signature(answer: AnswerTable) -> tuple:
    if not answer or not answer.columns:
        return ("__empty__",)
    n_cols = len(answer.columns)
    rows = [tuple(_norm_value(v) for v in r) for r in (answer.rows or []) if len(r) == n_cols]
    rows.sort()
    return (n_cols, len(rows), tuple(rows))


def _vote_answer(answers: list[AnswerTable]) -> AnswerTable | None:
    if not answers:
        return None
    sig_to_indices: dict[tuple, list[int]] = {}
    for i, a in enumerate(answers):
        sig = _signature(a)
        sig_to_indices.setdefault(sig, []).append(i)
    best = max(sig_to_indices, key=lambda s: len(sig_to_indices[s]))
    return answers[sig_to_indices[best][0]]


def run_one_attempt(*, attempt_idx: int, task, temp: float, max_iter: int):
    model = make_model(temp)
    return run_pipeline(task=task, model=model, max_iter=max_iter)


def run_one(tid: str, n_attempts: int, max_iter: int) -> dict:
    print(f"\n{'='*60}", flush=True)
    print(f"## {tid} [n_attempts={n_attempts}, max_iter={max_iter}]", flush=True)
    print(f"{'='*60}", flush=True)
    ds = DABenchPublicDataset(root_dir=REPO / "data" / "public" / "input")
    task = ds.get_task(tid)
    print(f"Q: {task.question}", flush=True)

    temps = [0.6, 0.6, 0.7][:n_attempts]
    t0 = time.time()
    pipeline_results = []
    with ThreadPoolExecutor(max_workers=n_attempts) as ex:
        futures = {
            ex.submit(run_one_attempt, attempt_idx=i, task=task, temp=temps[i], max_iter=max_iter): i
            for i in range(n_attempts)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                pipeline_results.append((i, fut.result()))
            except Exception as exc:
                import traceback; traceback.print_exc()
                pipeline_results.append((i, None))
    total_dt = time.time() - t0
    pipeline_results.sort()

    # Log per-attempt summary
    for idx, pr in pipeline_results:
        if pr is None:
            print(f"  attempt {idx}: EXCEPTION", flush=True)
            continue
        sr = pr.self_refine
        commit = sr.commit_reason if sr else "?"
        n_iter = sr.n_iterations if sr else 0
        exp_meta = pr.exploration_meta
        print(
            f"  attempt {idx} [temp={temps[idx]}]: "
            f"explore={exp_meta['successful']}/{exp_meta['queries_attempted']}, "
            f"self_refine iter={n_iter}, commit={commit}, "
            f"answer={'yes' if pr.ok else 'no'}, {pr.elapsed_seconds:.0f}s",
            flush=True,
        )

    answers = [pr.answer for _, pr in pipeline_results if pr and pr.ok and pr.answer]
    if not answers:
        return {"task": tid, "score": 0.0, "fail": "all attempts produced no answer", "total_latency_s": int(total_dt)}

    final = _vote_answer(answers)
    if final is None:
        return {"task": tid, "score": 0.0, "fail": "vote produced no answer"}

    pred_dir = REPO / "artifacts" / "reforce_loop_poc" / tid
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_p = pred_dir / "prediction.csv"
    with open(pred_p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(final.columns)
        for row in final.rows:
            w.writerow(row)
    # Save attempt traces for inspection
    for idx, pr in pipeline_results:
        if pr is None: continue
        trace_p = pred_dir / f"attempt_{idx}.json"
        trace_p.write_text(json.dumps({
            "exploration_meta": {k: v for k, v in pr.exploration_meta.items() if k != "raw_response"},
            "self_refine_history": pr.self_refine.history if pr.self_refine else None,
            "commit_reason": pr.self_refine.commit_reason if pr.self_refine else None,
            "n_iterations": pr.self_refine.n_iterations if pr.self_refine else None,
            "winning_sql": pr.self_refine.winning_sql if pr.self_refine else None,
            "elapsed_s": pr.elapsed_seconds,
        }, indent=2, default=str))

    e = _evaluate_task(
        task_id=tid, prediction_path=pred_p,
        gold_path=REPO / "data" / "public" / "output" / tid / "gold.csv",
        options=EvaluationOptions(),
    )
    print(f"  vote winner: cols={final.columns} n_rows={len(final.rows)}", flush=True)
    print(f"  score: λ0.5={e.official_score_lambda_0_5:.4f}  recall={e.recall:.2f}", flush=True)
    if final.rows[:2]:
        print(f"  rows[0:2]: {final.rows[:2]}", flush=True)
    return {
        "task": tid,
        "score": e.official_score_lambda_0_5,
        "pred_shape": [e.rows_pred, len(final.columns)],
        "n_attempts_succ": len(answers),
        "total_latency_s": int(total_dt),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_attempts", type=int, default=3)
    ap.add_argument("--max_iter", type=int, default=5)
    args = ap.parse_args()

    CONTROL = {
        "task_25": 0.583, "task_163": 0.000, "task_169": 0.000,
        "task_180": 0.000, "task_38": 0.367,
    }
    SQL_ONLY = {
        "task_25": 0.667, "task_163": 0.000, "task_169": 0.000,
        "task_180": 0.000, "task_38": 0.567,
    }

    results = []
    for tid in TASKS:
        try:
            results.append(run_one(tid, n_attempts=args.n_attempts, max_iter=args.max_iter))
        except Exception as exc:
            import traceback; traceback.print_exc()
            results.append({"task": tid, "score": 0.0, "exception": repr(exc)})

    print("\n" + "=" * 80, flush=True)
    print(f"REFORCE-LOOP POC SUMMARY (n_attempts={args.n_attempts}, max_iter={args.max_iter})", flush=True)
    print("=" * 80, flush=True)
    print(f"{'task':<10} {'score':>8} {'control':>8} {'sql_only':>10} {'vs_ctrl':>8} {'vs_sql':>8} {'lat_s':>6}", flush=True)
    summary = {}
    sum_d_ctrl = 0.0; sum_d_sql = 0.0
    for r in results:
        tid = r["task"]; s = r.get("score", 0.0)
        ctrl = CONTROL[tid]; sql_only = SQL_ONLY[tid]
        d_ctrl = s - ctrl; d_sql = s - sql_only
        sum_d_ctrl += d_ctrl; sum_d_sql += d_sql
        lat = r.get("total_latency_s", 0)
        summary[tid] = {"score": s, "control": ctrl, "sql_only": sql_only, "d_ctrl": d_ctrl, "d_sql": d_sql, "latency_s": lat}
        print(f"  {tid:<10} {s:>8.3f} {ctrl:>8.3f} {sql_only:>10.3f} {d_ctrl:>+8.3f} {d_sql:>+8.3f} {lat:>6}", flush=True)
    print(f"\nsum delta vs control:  {sum_d_ctrl:+.3f}  | avg: {sum_d_ctrl/len(TASKS):+.3f}", flush=True)
    print(f"sum delta vs sql_only: {sum_d_sql:+.3f}  | avg: {sum_d_sql/len(TASKS):+.3f}", flush=True)

    out = REPO / "artifacts" / "reforce_loop_poc" / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"raw": results, "summary": summary, "args": vars(args)}, indent=2, ensure_ascii=False))
    print(f"\nresults: {out}", flush=True)


if __name__ == "__main__":
    main()
