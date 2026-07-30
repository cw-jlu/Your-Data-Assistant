"""Control: same 5 tasks, single-attempt, NO brief — for fair comparison vs POC.

The qr_brief POC ran single-attempt with the brief. To attribute any
delta to the brief itself (and not to attempts-class confounding), this
script runs the SAME tasks at single-attempt with the SAME exp_109
baseline preamble — no brief.

Re-runs N times to estimate variance.

Usage:
    uv run python scripts/test_qr_brief_control.py [--n 3]
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
from experiments.exp_109_plan_first_strengthen.agent import ReActAgent, ReActAgentConfig
from experiments.exp_109_plan_first_strengthen.preamble import build_preamble
from experiments.exp_109_plan_first_strengthen.tools.registry import (
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
    print(f"\n{'='*60}")
    print(f"## {tid} [run {run_idx}]")
    print(f"{'='*60}")
    ds = DABenchPublicDataset(root_dir=REPO / "data" / "public" / "input")
    task = ds.get_task(tid)
    print(f"Q: {task.question}")

    base_preamble = build_preamble(task)
    print(f"preamble: {base_preamble.char_count} chars")

    agent = ReActAgent(
        model=model,
        tools=tools,
        config=ReActAgentConfig(max_steps=24, min_steps=4),
        preamble=base_preamble.text,
    )
    t0 = time.time()
    result = agent.run(task)
    dt = time.time() - t0
    print(f"agent latency: {dt:.0f}s  steps: {len(result.steps)}  succeeded: {result.succeeded}")

    if not result.succeeded or not result.answer:
        print(f"FAIL: {result.failure_reason}")
        return {"task": tid, "run": run_idx, "score": 0.0, "fail": result.failure_reason}

    pred_dir = REPO / "artifacts" / "qr_brief_control" / f"run_{run_idx}" / tid
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
    print(f"score: λ0.5={e.official_score_lambda_0_5:.4f}  recall={e.recall:.2f}")
    if result.answer.rows[:2]:
        print(f"  rows[0:2]: {result.answer.rows[:2]}")
    return {
        "task": tid,
        "run": run_idx,
        "score": e.official_score_lambda_0_5,
        "pred_shape": [e.rows_pred, len(result.answer.columns)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="repetitions per task")
    args = ap.parse_args()

    model = make_model()
    tools = create_default_tool_registry()
    results = []
    for run_idx in range(args.n):
        for tid in TASKS:
            try:
                results.append(run_one(tid, run_idx, model, tools))
            except Exception as exc:
                print(f"  EXCEPTION on {tid} run {run_idx}: {exc!r}")
                results.append({"task": tid, "run": run_idx, "score": 0.0, "exception": repr(exc)})

    # Summary
    print("\n" + "=" * 60)
    print("CONTROL SUMMARY (single-attempt, NO brief)")
    print("=" * 60)
    print(f"{'task':<10} " + " ".join(f"{'run'+str(i):>8}" for i in range(args.n)) + f" {'mean':>8} {'max':>8} {'min':>8}")
    summary = {}
    for tid in TASKS:
        row = [r["score"] for r in results if r["task"] == tid]
        while len(row) < args.n: row.append(0.0)
        mn, mx, mu = min(row), max(row), sum(row)/len(row)
        summary[tid] = {"runs": row, "mean": mu, "max": mx, "min": mn}
        print(f"  {tid:<10} " + " ".join(f"{s:>8.3f}" for s in row) + f" {mu:>8.3f} {mx:>8.3f} {mn:>8.3f}")

    out = REPO / "artifacts" / "qr_brief_control" / "results.json"
    out.write_text(json.dumps({"raw": results, "summary": summary}, indent=2, ensure_ascii=False))
    print(f"\nresults: {out}")


if __name__ == "__main__":
    main()
