"""PoC: ReFoRCE-stacked SQL-only agent (= exp_114).

Stacks three techniques on top of exp_111's SQL-only base:
  (b) Self-refinement on suspicious results (= empty / wrong shape)
  (d) Column exploration pre-phase (= LLM-generated DISTINCT/COUNT queries)
  ANSWER_FROM_SQL enforcement (= no manual `answer` tool)

Same 5 tasks × N runs single-attempt, compared to control + sql_only baselines.

Usage:
    uv run python scripts/test_reforce_stack_poc.py [--n 1]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv
load_dotenv()

from kobushi_core.benchmark.dataset import DABenchPublicDataset
from kobushi_core.eval.csv_compare import _evaluate_task, EvaluationOptions
from kobushi_core.model import OpenAIModelAdapter
from experiments.exp_114_reforce_stack.preamble import build_preamble
from experiments.exp_114_reforce_stack.tools.registry import create_default_tool_registry
from experiments.exp_114_reforce_stack.exploration import run_exploration_phase
from experiments.exp_114_reforce_stack.refinement import run_with_refinement


TASKS = ["task_25", "task_163", "task_169", "task_180", "task_38"]


def make_model() -> OpenAIModelAdapter:
    return OpenAIModelAdapter(
        model="qwen3.5-35b-a3b",
        api_base=os.environ["AGENT_API_BASE"],
        api_key=os.environ["AGENT_API_KEY"],
        temperature=0.6,
        extra_headers={
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
        presence_penalty=1.0,
    )


def run_one(tid: str, run_idx: int, model, tools) -> dict:
    print(f"\n{'='*60}", flush=True)
    print(f"## {tid} [run {run_idx}]", flush=True)
    print(f"{'='*60}", flush=True)
    ds = DABenchPublicDataset(root_dir=REPO / "data" / "public" / "input")
    task = ds.get_task(tid)
    print(f"Q: {task.question}", flush=True)

    # Phase 1: column exploration
    t0 = time.time()
    exp_result = run_exploration_phase(task=task, model=model, max_queries=8)
    exp_dt = time.time() - t0
    print(
        f"exploration: {exp_dt:.0f}s, "
        f"{exp_result['successful']} ok / {exp_result['failed']} fail "
        f"({exp_result['queries_attempted']} attempted)",
        flush=True,
    )

    # Phase 2: build preamble + main agent + refinement
    base_preamble = build_preamble(task)
    preamble_text = base_preamble.text
    if exp_result["block"]:
        preamble_text = preamble_text + "\n\n---\n\n" + exp_result["block"]
    print(f"preamble: {len(preamble_text)} chars (= base {base_preamble.char_count} + exploration)", flush=True)

    t0 = time.time()
    result = run_with_refinement(
        task=task,
        model=model,
        tools=tools,
        base_preamble=preamble_text,
        max_steps=14,
        min_steps=3,
        max_refines=1,
    )
    main_dt = time.time() - t0
    print(f"main agent (+refinement): {main_dt:.0f}s, steps: {len(result.steps)}, succeeded: {result.succeeded}", flush=True)

    if not result.succeeded or not result.answer:
        print(f"FAIL: {result.failure_reason}", flush=True)
        return {"task": tid, "run": run_idx, "score": 0.0, "fail": result.failure_reason}

    pred_dir = REPO / "artifacts" / "reforce_stack_poc" / f"run_{run_idx}" / tid
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_p = pred_dir / "prediction.csv"
    with open(pred_p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(result.answer.columns)
        for row in result.answer.rows:
            w.writerow(row)
    # Save exploration block for inspection
    (pred_dir / "exploration.txt").write_text(exp_result["block"])

    e = _evaluate_task(
        task_id=tid,
        prediction_path=pred_p,
        gold_path=REPO / "data" / "public" / "output" / tid / "gold.csv",
        options=EvaluationOptions(),
    )
    print(f"score: λ0.5={e.official_score_lambda_0_5:.4f}  recall={e.recall:.2f}", flush=True)
    if result.answer.rows[:2]:
        print(f"  rows[0:2]: {result.answer.rows[:2]}", flush=True)
    return {
        "task": tid,
        "run": run_idx,
        "score": e.official_score_lambda_0_5,
        "pred_shape": [e.rows_pred, len(result.answer.columns)],
        "n_steps": len(result.steps),
        "exp_latency_s": int(exp_dt),
        "main_latency_s": int(main_dt),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1)
    args = ap.parse_args()

    model = make_model()
    tools = create_default_tool_registry()

    # Reference baselines
    CONTROL = {
        "task_25": 0.583, "task_163": 0.000, "task_169": 0.000,
        "task_180": 0.000, "task_38": 0.367,
    }
    SQL_ONLY = {
        "task_25": 0.667, "task_163": 0.000, "task_169": 0.000,
        "task_180": 0.000, "task_38": 0.567,
    }

    results = []
    for run_idx in range(args.n):
        for tid in TASKS:
            try:
                results.append(run_one(tid, run_idx, model, tools))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                results.append({"task": tid, "run": run_idx, "score": 0.0, "exception": repr(exc)})

    print("\n" + "=" * 80, flush=True)
    print(f"REFORCE-STACK POC SUMMARY (single-attempt × N={args.n})", flush=True)
    print("=" * 80, flush=True)
    headers = ['task'] + [f'run{i}' for i in range(args.n)] + ['mean', 'control', 'sql_only', 'vs_ctrl', 'vs_sql']
    print(' '.join(f'{h:>10}' for h in headers), flush=True)
    summary = {}
    sum_d_ctrl = 0.0
    sum_d_sql = 0.0
    for tid in TASKS:
        scores = [r["score"] for r in results if r["task"] == tid]
        while len(scores) < args.n: scores.append(0.0)
        mean = sum(scores) / len(scores)
        ctrl = CONTROL[tid]
        sql_only = SQL_ONLY[tid]
        d_ctrl = mean - ctrl
        d_sql = mean - sql_only
        sum_d_ctrl += d_ctrl
        sum_d_sql += d_sql
        summary[tid] = {"runs": scores, "mean": mean, "control": ctrl, "sql_only": sql_only,
                        "d_ctrl": d_ctrl, "d_sql": d_sql}
        row = [tid] + [f'{s:.3f}' for s in scores] + [f'{mean:.3f}', f'{ctrl:.3f}', f'{sql_only:.3f}',
                                                       f'{d_ctrl:+.3f}', f'{d_sql:+.3f}']
        print(' '.join(f'{x:>10}' for x in row), flush=True)
    print(f"\nsum delta vs control:  {sum_d_ctrl:+.3f}", flush=True)
    print(f"avg delta vs control:  {sum_d_ctrl/len(TASKS):+.3f}", flush=True)
    print(f"sum delta vs sql_only: {sum_d_sql:+.3f}", flush=True)
    print(f"avg delta vs sql_only: {sum_d_sql/len(TASKS):+.3f}", flush=True)

    out = REPO / "artifacts" / "reforce_stack_poc" / "results.json"
    out.write_text(json.dumps({"raw": results, "summary": summary}, indent=2, ensure_ascii=False))
    print(f"\nresults: {out}", flush=True)


if __name__ == "__main__":
    main()
