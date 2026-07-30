"""PoC: Orchestrator + full-loop SQL sub-agent (= exp_113).

Same 5 tasks × N runs single-attempt. Compare to control + exp_111 single-attempt.

Usage:
    uv run python scripts/test_subagent_poc.py [--n 3]
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
from experiments.exp_113_subagent_orchestrator.agent import ReActAgent, ReActAgentConfig
from experiments.exp_113_subagent_orchestrator.preamble import build_preamble
from experiments.exp_113_subagent_orchestrator.tools.registry import (
    create_default_tool_registry,
)


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

    base_preamble = build_preamble(task)
    print(f"preamble: {base_preamble.char_count} chars", flush=True)

    agent = ReActAgent(
        model=model,
        tools=tools,
        config=ReActAgentConfig(max_steps=14, min_steps=3),
        preamble=base_preamble.text,
    )
    t0 = time.time()
    result = agent.run(task)
    dt = time.time() - t0
    print(f"orchestrator latency: {dt:.0f}s  steps: {len(result.steps)}  succeeded: {result.succeeded}", flush=True)

    if not result.succeeded or not result.answer:
        print(f"FAIL: {result.failure_reason}", flush=True)
        return {"task": tid, "run": run_idx, "score": 0.0, "fail": result.failure_reason}

    pred_dir = REPO / "artifacts" / "subagent_poc" / f"run_{run_idx}" / tid
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_p = pred_dir / "prediction.csv"
    with open(pred_p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(result.answer.columns)
        for row in result.answer.rows:
            w.writerow(row)

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
        "latency_s": int(dt),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1)
    args = ap.parse_args()

    model = make_model()
    tools = create_default_tool_registry(model=model)

    # Reference baselines from past runs
    CONTROL = {  # qr_brief_control: exp_109 single-attempt no-brief mean
        "task_25": 0.583, "task_163": 0.000, "task_169": 0.000,
        "task_180": 0.000, "task_38": 0.367,
    }
    SQL_ONLY = {  # exp_111 5-task PoC mean
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

    print("\n" + "=" * 70, flush=True)
    print(f"SUBAGENT-ORCHESTRATOR POC SUMMARY (single-attempt × N={args.n})", flush=True)
    print("=" * 70, flush=True)
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
        summary[tid] = {"runs": scores, "mean": mean, "control": ctrl, "sql_only": sql_only, "d_ctrl": d_ctrl, "d_sql": d_sql}
        row = [tid] + [f'{s:.3f}' for s in scores] + [f'{mean:.3f}', f'{ctrl:.3f}', f'{sql_only:.3f}', f'{d_ctrl:+.3f}', f'{d_sql:+.3f}']
        print(' '.join(f'{x:>10}' for x in row), flush=True)
    print(f"\nsum delta vs control:  {sum_d_ctrl:+.3f}", flush=True)
    print(f"avg delta vs control:  {sum_d_ctrl/len(TASKS):+.3f}", flush=True)
    print(f"sum delta vs sql_only: {sum_d_sql:+.3f}", flush=True)
    print(f"avg delta vs sql_only: {sum_d_sql/len(TASKS):+.3f}", flush=True)

    out = REPO / "artifacts" / "subagent_poc" / "results.json"
    out.write_text(json.dumps({"raw": results, "summary": summary}, indent=2, ensure_ascii=False))
    print(f"\nresults: {out}", flush=True)


if __name__ == "__main__":
    main()
