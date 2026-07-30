"""PoC: voted_answer_from_sql with K=5 candidates @ temp=1.0 + 3-attempt union.

Architecture (= ReFoRCE pillars (b)+(c)):
  - Each task runs 3 attempts in parallel (= outer 3-attempt union, like exp_109)
  - Each attempt's terminal action is voted_answer_from_sql
  - Inside voted_answer_from_sql: K=5 candidate SQLs at temp=1.0, executed in
    parallel, majority-vote on result rows
  - Outer 3-attempt union via signature-merge

Same 5 tasks vs control + sql_only (= exp_111 single-attempt) baselines.

Usage:
    uv run python scripts/test_voted_sql_poc.py [--n_attempts 3] [--k 5]
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
from experiments.exp_115_voted_sql.agent import ReActAgent, ReActAgentConfig
from experiments.exp_115_voted_sql.preamble import build_preamble
from experiments.exp_115_voted_sql.tools.registry import create_default_tool_registry


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
    """Pick the most-voted answer among 3-attempt outputs."""
    if not answers:
        return None
    sig_to_indices: dict[tuple, list[int]] = {}
    for i, a in enumerate(answers):
        sig = _signature(a)
        sig_to_indices.setdefault(sig, []).append(i)
    best = max(sig_to_indices, key=lambda s: len(sig_to_indices[s]))
    return answers[sig_to_indices[best][0]]


def run_one_attempt(
    *, attempt_idx: int, task, agent_temp: float, voter_temp: float,
    base_preamble_text: str, k: int = 5,
):
    agent_model = make_model(agent_temp)
    voter_model = make_model(voter_temp)
    # Tie-break model: deterministic (low temp) for stable arbitration
    tie_break_model = make_model(0.2)
    tools = create_default_tool_registry(
        voter_model=voter_model, tie_break_model=tie_break_model, k=k,
    )
    agent = ReActAgent(
        model=agent_model,
        tools=tools,
        config=ReActAgentConfig(max_steps=14, min_steps=3),
        preamble=base_preamble_text,
    )
    t0 = time.time()
    result = agent.run(task)
    dt = time.time() - t0
    return {
        "attempt": attempt_idx,
        "succeeded": result.succeeded,
        "answer": result.answer,
        "n_steps": len(result.steps),
        "latency_s": dt,
    }


def run_one(tid: str, run_idx: int, n_attempts: int, k: int) -> dict:
    print(f"\n{'='*60}", flush=True)
    print(f"## {tid} [run {run_idx}, n_attempts={n_attempts}, k={k}]", flush=True)
    print(f"{'='*60}", flush=True)
    ds = DABenchPublicDataset(root_dir=REPO / "data" / "public" / "input")
    task = ds.get_task(tid)
    print(f"Q: {task.question}", flush=True)

    base_preamble = build_preamble(task)
    print(f"preamble: {base_preamble.char_count} chars", flush=True)

    # Outer 3-attempt with two different agent temperatures (= matches exp_109 style)
    agent_temps = [0.6, 0.6, 0.7][:n_attempts]
    t0 = time.time()
    attempts: list = []
    with ThreadPoolExecutor(max_workers=n_attempts) as ex:
        futures = [
            ex.submit(
                run_one_attempt,
                attempt_idx=i, task=task,
                agent_temp=agent_temps[i], voter_temp=1.0,
                base_preamble_text=base_preamble.text, k=k,
            )
            for i in range(n_attempts)
        ]
        for fut in as_completed(futures):
            attempts.append(fut.result())
    total_dt = time.time() - t0
    attempts.sort(key=lambda a: a["attempt"])

    succeeded_answers = [a["answer"] for a in attempts if a["succeeded"] and a["answer"]]
    per_att = ", ".join(f"{a['latency_s']:.0f}s" for a in attempts)
    print(
        f"3-attempt: {len(succeeded_answers)}/{n_attempts} succeeded, "
        f"total latency: {total_dt:.0f}s (per-attempt: [{per_att}])",
        flush=True,
    )

    if not succeeded_answers:
        print("FAIL: all attempts failed", flush=True)
        return {"task": tid, "run": run_idx, "score": 0.0, "fail": "all attempts failed"}

    final_answer = _vote_answer(succeeded_answers)
    if final_answer is None:
        return {"task": tid, "run": run_idx, "score": 0.0, "fail": "vote produced no answer"}

    pred_dir = REPO / "artifacts" / "voted_sql_poc" / f"run_{run_idx}" / tid
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_p = pred_dir / "prediction.csv"
    with open(pred_p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(final_answer.columns)
        for row in final_answer.rows:
            w.writerow(row)

    e = _evaluate_task(
        task_id=tid,
        prediction_path=pred_p,
        gold_path=REPO / "data" / "public" / "output" / tid / "gold.csv",
        options=EvaluationOptions(),
    )
    print(f"score: λ0.5={e.official_score_lambda_0_5:.4f}  recall={e.recall:.2f}", flush=True)
    if final_answer.rows[:2]:
        print(f"  rows[0:2]: {final_answer.rows[:2]}", flush=True)
    return {
        "task": tid,
        "run": run_idx,
        "score": e.official_score_lambda_0_5,
        "pred_shape": [e.rows_pred, len(final_answer.columns)],
        "n_attempts_succ": len(succeeded_answers),
        "total_latency_s": int(total_dt),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_attempts", type=int, default=3)
    ap.add_argument("--k", type=int, default=5)
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
            results.append(run_one(tid, 0, n_attempts=args.n_attempts, k=args.k))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            results.append({"task": tid, "run": 0, "score": 0.0, "exception": repr(exc)})

    print("\n" + "=" * 80, flush=True)
    print(f"VOTED-SQL POC SUMMARY (n_attempts={args.n_attempts}, k={args.k})", flush=True)
    print("=" * 80, flush=True)
    print(f"{'task':<10} {'score':>8} {'control':>8} {'sql_only':>10} {'vs_ctrl':>8} {'vs_sql':>8}", flush=True)
    summary = {}
    sum_d_ctrl = 0.0; sum_d_sql = 0.0
    for r in results:
        tid = r["task"]
        s = r.get("score", 0.0)
        ctrl = CONTROL[tid]; sql_only = SQL_ONLY[tid]
        d_ctrl = s - ctrl; d_sql = s - sql_only
        sum_d_ctrl += d_ctrl; sum_d_sql += d_sql
        summary[tid] = {"score": s, "control": ctrl, "sql_only": sql_only, "d_ctrl": d_ctrl, "d_sql": d_sql}
        print(f"  {tid:<10} {s:>8.3f} {ctrl:>8.3f} {sql_only:>10.3f} {d_ctrl:>+8.3f} {d_sql:>+8.3f}", flush=True)
    print(f"\nsum delta vs control:  {sum_d_ctrl:+.3f}  | avg: {sum_d_ctrl/len(TASKS):+.3f}", flush=True)
    print(f"sum delta vs sql_only: {sum_d_sql:+.3f}  | avg: {sum_d_sql/len(TASKS):+.3f}", flush=True)

    out = REPO / "artifacts" / "voted_sql_poc" / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"raw": results, "summary": summary, "args": vars(args)}, indent=2, ensure_ascii=False))
    print(f"\nresults: {out}", flush=True)


if __name__ == "__main__":
    main()
