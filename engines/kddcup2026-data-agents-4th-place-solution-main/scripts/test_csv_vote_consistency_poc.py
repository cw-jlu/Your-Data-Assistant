"""PoC: exp_111 sql_only + ReFoRCE 3 mechanisms (= csv header + vote + consistency).

Architecture (= minimal diff from sql_only):
  Pre-step: format inference (= 1 LLM call at temp=0.0) → CSV header hint.
            Header is injected into the agent's preamble.
  ReAct loop: unchanged from exp_111.
  Terminal: answer_from_sql replaced with voted-terminal handler. The agent's
            SQL becomes candidate 0; K-1 alternatives are generated at temp=1.0
            from a focused specialist prompt; all execute against DuckDB; vote
            on result row-sets; commit only on UNIQUE max-vote winner.
            Tied votes return a non-terminal error so the agent retries.

Run on 5 task subset, 3-attempt outer × single-attempt-equivalent inner.

Usage:
    uv run python scripts/test_csv_vote_consistency_poc.py [--n_attempts 3] [--k 5]
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
from kobushi_core.benchmark.schema import AnswerTable, PublicTask
from kobushi_core.eval.csv_compare import _evaluate_task, EvaluationOptions
from kobushi_core.model import OpenAIModelAdapter
from experiments.exp_119_csv_vote_consistency.agent import ReActAgent, ReActAgentConfig
from experiments.exp_119_csv_vote_consistency.preamble import build_preamble
from experiments.exp_119_csv_vote_consistency.tools.registry import create_default_tool_registry
from experiments.exp_119_csv_vote_consistency.format_inference import run_format_inference


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
    if v is None: return (0, "")
    s = str(v).strip()
    try: return (1, round(float(s), 3))
    except: return (2, s.lower())


def _signature(answer: AnswerTable) -> tuple:
    if not answer or not answer.columns: return ("__empty__",)
    n_cols = len(answer.columns)
    rows = [tuple(_norm_value(v) for v in r) for r in (answer.rows or []) if len(r) == n_cols]
    rows.sort()
    return (n_cols, len(rows), tuple(rows))


def _vote_answer(answers: list[AnswerTable]) -> AnswerTable | None:
    if not answers: return None
    s2i: dict[tuple, list[int]] = {}
    for i, a in enumerate(answers):
        s2i.setdefault(_signature(a), []).append(i)
    best = max(s2i, key=lambda s: len(s2i[s]))
    return answers[s2i[best][0]]


def run_one_attempt(*, task: PublicTask, agent_temp: float, voter_temp: float, k: int):
    agent_model = make_model(agent_temp)
    voter_model = make_model(voter_temp)
    fmt_model = make_model(0.0)

    # Phase 0: format inference (1 LLM call at deterministic temp)
    format_csv = run_format_inference(task=task, model=fmt_model)

    # Build registry with closures so the voted handler sees question + format
    tools = create_default_tool_registry(
        voter_model=voter_model,
        sub_question_provider=lambda: task.question,
        format_csv_provider=lambda: format_csv,
        k=k,
    )

    # Inject format header into the preamble as a hint
    base_preamble = build_preamble(task)
    fmt_hint = (
        f"\n\n# Expected answer format (= inferred CSV header)\n"
        f"```csv\n{format_csv}\n```\n"
        f"Your final answer SQL should produce these columns. Use `AS` to "
        f"rename source columns to match the expected header. Match the "
        f"column count exactly — extras subtract points.\n"
        if format_csv else ""
    )
    preamble_text = base_preamble.text + fmt_hint

    agent = ReActAgent(
        model=agent_model, tools=tools,
        config=ReActAgentConfig(max_steps=24, min_steps=4),
        preamble=preamble_text,
    )
    t0 = time.time()
    result = agent.run(task)
    dt = time.time() - t0
    return {
        "format_csv": format_csv,
        "result": result,
        "elapsed_s": dt,
    }


def run_one(tid: str, n_attempts: int, k: int) -> dict:
    print(f"\n{'='*60}", flush=True)
    print(f"## {tid} [n_attempts={n_attempts}, k={k}]", flush=True)
    print(f"{'='*60}", flush=True)
    ds = DABenchPublicDataset(root_dir=REPO / "data" / "public" / "input")
    task = ds.get_task(tid)
    print(f"Q: {task.question}", flush=True)

    temps = [0.6, 0.6, 0.7][:n_attempts]
    t0 = time.time()
    attempt_meta = []
    with ThreadPoolExecutor(max_workers=n_attempts) as ex:
        futures = {
            ex.submit(run_one_attempt, task=task, agent_temp=temps[i], voter_temp=1.0, k=k): i
            for i in range(n_attempts)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                attempt_meta.append((i, fut.result()))
            except Exception as exc:
                import traceback; traceback.print_exc()
                attempt_meta.append((i, None))
    total_dt = time.time() - t0
    attempt_meta.sort()

    for idx, m in attempt_meta:
        if m is None:
            print(f"  attempt {idx}: EXCEPTION", flush=True); continue
        r = m["result"]
        ans = r.answer
        n_steps = len(r.steps)
        print(
            f"  attempt {idx} [temp={temps[idx]}]: format='{m['format_csv']}', "
            f"steps={n_steps}, succeeded={r.succeeded}, "
            f"answer_shape={(len(ans.rows), len(ans.columns)) if ans else None}, "
            f"{m['elapsed_s']:.0f}s",
            flush=True,
        )

    answers = [m["result"].answer for _, m in attempt_meta if m and m["result"].succeeded and m["result"].answer]
    if not answers:
        return {"task": tid, "score": 0.0, "fail": "all attempts failed",
                "total_latency_s": int(total_dt)}

    final = _vote_answer(answers)
    if final is None:
        return {"task": tid, "score": 0.0, "fail": "vote produced no answer"}

    pred_dir = REPO / "artifacts" / "csv_vote_consistency_poc" / tid
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_p = pred_dir / "prediction.csv"
    with open(pred_p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(final.columns)
        for row in final.rows:
            w.writerow(row)

    e = _evaluate_task(
        task_id=tid, prediction_path=pred_p,
        gold_path=REPO / "data" / "public" / "output" / tid / "gold.csv",
        options=EvaluationOptions(),
    )
    print(f"  vote winner: cols={final.columns} n_rows={len(final.rows)}", flush=True)
    print(f"  score: λ0.5={e.official_score_lambda_0_5:.4f} recall={e.recall:.2f}", flush=True)
    if final.rows[:2]:
        print(f"  rows[0:2]: {final.rows[:2]}", flush=True)
    return {
        "task": tid, "score": e.official_score_lambda_0_5,
        "pred_shape": [e.rows_pred, len(final.columns)],
        "n_attempts_succ": len(answers),
        "total_latency_s": int(total_dt),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_attempts", type=int, default=3)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    SQL_ONLY = {"task_25": 0.667, "task_163": 0.000, "task_169": 0.000,
                "task_180": 0.000, "task_38": 0.567}

    results = []
    for tid in TASKS:
        try:
            results.append(run_one(tid, n_attempts=args.n_attempts, k=args.k))
        except Exception as exc:
            import traceback; traceback.print_exc()
            results.append({"task": tid, "score": 0.0, "exception": repr(exc)})

    print("\n" + "=" * 70, flush=True)
    print(f"CSV-VOTE-CONSISTENCY POC SUMMARY (n_attempts={args.n_attempts}, k={args.k})", flush=True)
    print("=" * 70, flush=True)
    print(f"{'task':<10} {'score':>8} {'sql_only':>10} {'delta':>8} {'lat_s':>6}", flush=True)
    summary = {}
    sum_d = 0.0
    for r in results:
        tid = r["task"]; s = r.get("score", 0.0)
        base = SQL_ONLY[tid]; d = s - base
        sum_d += d
        lat = r.get("total_latency_s", 0)
        summary[tid] = {"score": s, "sql_only": base, "delta": d, "latency_s": lat}
        print(f"  {tid:<10} {s:>8.3f} {base:>10.3f} {d:>+8.3f} {lat:>6}", flush=True)
    print(f"\nsum delta vs sql_only: {sum_d:+.3f}  | avg: {sum_d/len(TASKS):+.3f}", flush=True)

    out = REPO / "artifacts" / "csv_vote_consistency_poc" / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"raw": results, "summary": summary, "args": vars(args)}, indent=2, ensure_ascii=False))
    print(f"\nresults: {out}", flush=True)


if __name__ == "__main__":
    main()
