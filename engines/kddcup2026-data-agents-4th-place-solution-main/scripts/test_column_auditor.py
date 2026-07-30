"""Test exp_122 (= exp_120 + post-execution column auditor) on the partial-credit tasks.

Targets the 8 partial + 2 wrong (= 10) tasks where exp_120 had column extras.

Compare per-task to exp_120's prior score.

Usage:
    uv run python scripts/test_column_auditor.py
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
load_dotenv()

from kobushi_core.benchmark.dataset import DABenchPublicDataset
from kobushi_core.eval.csv_compare import _evaluate_task, EvaluationOptions
from kobushi_core.model import OpenAIModelAdapter
from experiments.exp_122_column_auditor.agent import ReActAgent, ReActAgentConfig
from experiments.exp_122_column_auditor.preamble import build_preamble
from experiments.exp_122_column_auditor.tools.registry import create_default_tool_registry


# 8 partial + 2 wrong tasks where exp_120 had column extras
TARGET_TASKS = [
    # Partial-credit tasks
    ("task_27",  0.88),
    ("task_38",  0.56),
    ("task_196", 0.75),
    ("task_259", 0.67),
    ("task_330", 0.25),
    ("task_352", 0.75),
    ("task_379", 0.75),
    ("task_418", 0.75),
    # Wrong tasks where extras may have contributed
    ("task_180", 0.00),
    ("task_86",  0.00),
]

PRIOR_120 = dict(TARGET_TASKS)


def make_model(temp):
    return OpenAIModelAdapter(
        model="qwen3.5-35b-a3b",
        api_base=os.environ["AGENT_API_BASE"],
        api_key=os.environ["AGENT_API_KEY"],
        temperature=temp,
        extra_headers={
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
        presence_penalty=1.0,
    )


from experiments.exp_122_column_auditor.adaptive_vote import adaptive_vote as _vote


def run_attempt(task, temp):
    agent_model = make_model(temp)
    auditor_model = make_model(0.0)  # deterministic auditor
    tools = create_default_tool_registry(
        auditor_model=auditor_model,
        question_provider=lambda: task.question,
    )
    pre = build_preamble(task)
    agent = ReActAgent(
        model=agent_model, tools=tools,
        config=ReActAgentConfig(max_steps=24, min_steps=4),
        preamble=pre.text,
    )
    return agent.run(task)


def run_one(task, n_attempts=3):
    print(f"\n## {task.task_id}", flush=True)
    print(f"  Q: {task.question}", flush=True)
    temps = [0.6, 0.6, 0.7][:n_attempts]
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=n_attempts) as ex:
        futs = {ex.submit(run_attempt, task, temps[i]): i for i in range(n_attempts)}
        for fut in as_completed(futs):
            i = futs[fut]
            try: results.append((i, fut.result()))
            except Exception as exc:
                print(f"  attempt {i}: EXC {repr(exc)[:120]}", flush=True)
                results.append((i, None))
    answers = []
    pred_dir = REPO / "artifacts" / "column_auditor" / task.task_id
    pred_dir.mkdir(parents=True, exist_ok=True)
    for i, r in results:
        if r is None:
            (pred_dir / f"attempt_{i}.json").write_text(json.dumps({"failed": True}))
            continue
        ans_shape = (len(r.answer.rows), len(r.answer.columns)) if r.answer else None
        ans_preview = r.answer.rows[:3] if r.answer else None
        print(f"  attempt {i} [t={temps[i]}]: ok={r.succeeded} ans={ans_shape} preview={ans_preview}", flush=True)
        # Extract the agent's terminal SQL (= for offline LLM-router with full context)
        agent_sql = None
        for s in reversed(r.steps):
            if s.action == "answer_from_sql":
                ai = s.action_input or {}
                if isinstance(ai, dict) and "sql" in ai:
                    agent_sql = ai["sql"]
                    break
        # Save per-attempt audited answer + SQL
        attempt_data = {
            "attempt": i, "temp": temps[i], "succeeded": r.succeeded,
            "agent_sql": agent_sql,
            "answer_columns": list(r.answer.columns) if r.answer else None,
            "answer_rows": [list(row) for row in r.answer.rows] if r.answer else None,
        }
        (pred_dir / f"attempt_{i}.json").write_text(json.dumps(attempt_data, default=str))
        if r.succeeded and r.answer:
            answers.append(r.answer)
    if not answers:
        return {"task": task.task_id, "score": 0.0, "fail": "all attempts failed", "latency_s": int(time.time()-t0)}
    final = _vote(answers)
    pred_p = pred_dir / "prediction.csv"
    with open(pred_p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(final.columns)
        for r in final.rows: w.writerow(r)
    e = _evaluate_task(
        task_id=task.task_id, prediction_path=pred_p,
        gold_path=REPO / "data" / "public" / "output" / task.task_id / "gold.csv",
        options=EvaluationOptions(),
    )
    print(f"  vote: cols={final.columns} n_rows={len(final.rows)}", flush=True)
    print(f"  score: λ0.5={e.official_score_lambda_0_5:.4f}", flush=True)
    return {"task": task.task_id, "score": e.official_score_lambda_0_5,
            "shape": [e.rows_pred, len(final.columns)],
            "rows_preview": final.rows[:3],
            "latency_s": int(time.time()-t0)}


def main():
    ds = DABenchPublicDataset(root_dir=REPO / "data" / "public" / "input")
    results = []
    for tid, _ in TARGET_TASKS:
        try:
            results.append(run_one(ds.get_task(tid)))
        except Exception as exc:
            import traceback; traceback.print_exc()
            results.append({"task": tid, "score": 0.0, "exception": repr(exc)})

    print("\n" + "=" * 80, flush=True)
    print("COLUMN AUDITOR PoC SUMMARY", flush=True)
    print("=" * 80, flush=True)
    print(f"{'task':<10} {'exp_122':>8} {'exp_120':>8} {'delta':>8}", flush=True)
    sum_d = 0.0
    for r in results:
        tid = r["task"]; s = r.get("score", 0.0)
        prior = PRIOR_120[tid]
        d = s - prior
        sum_d += d
        print(f"  {tid:<10} {s:>8.3f} {prior:>8.3f} {d:>+8.3f}", flush=True)
    print(f"\nsum delta vs exp_120 (10 tasks): {sum_d:+.3f}", flush=True)
    print(f"avg delta: {sum_d/len(results):+.3f}", flush=True)
    # extrapolate to 50-task mean
    proj = 0.7469 + sum_d / 50
    print(f"projected 50-task mean (= exp_120 0.7469 + sum_delta/50): {proj:.4f}", flush=True)

    out = REPO / "artifacts" / "column_auditor" / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
