"""PoC: exp_120 (= sql_only + searchable read_doc + grep) on 11 mixed-doc tasks.

Architecture (= minimal diff from exp_111 sql_only):
  - read_doc adds `search` and `offset` parameters (= grep paragraphs + paginate)
  - new tool `grep` (= regex × multi-file × line-numbers × context, Claude-Code style)
  - all other tools/prompts identical to exp_111

Per task: 3 attempts in parallel (= matches exp_109's 3-attempt union pattern),
signature-vote across attempts. Compare to exp_109 historic best per task.

Usage:
    uv run python scripts/test_searchable_doc_poc.py [--n_attempts 3]
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
from experiments.exp_120_searchable_doc.agent import ReActAgent, ReActAgentConfig
from experiments.exp_120_searchable_doc.preamble import build_preamble
from experiments.exp_120_searchable_doc.tools.registry import create_default_tool_registry


# All 11 tasks with non-knowledge.md docs
MIXED_DOC_TASKS = [
    "task_330", "task_344", "task_349", "task_350", "task_352",
    "task_355", "task_379", "task_396", "task_408", "task_415", "task_420",
]


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


def run_attempt(*, task, agent_temp: float):
    model = make_model(agent_temp)
    tools = create_default_tool_registry()
    base_preamble = build_preamble(task)
    agent = ReActAgent(
        model=model, tools=tools,
        config=ReActAgentConfig(max_steps=24, min_steps=4),
        preamble=base_preamble.text,
    )
    t0 = time.time()
    result = agent.run(task)
    return {"result": result, "elapsed_s": time.time() - t0}


def run_one(tid: str, n_attempts: int) -> dict:
    print(f"\n## {tid}", flush=True)
    ds = DABenchPublicDataset(root_dir=REPO / "data" / "public" / "input")
    task = ds.get_task(tid)
    print(f"  Q: {task.question}", flush=True)

    temps = [0.6, 0.6, 0.7][:n_attempts]
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=n_attempts) as ex:
        futures = {ex.submit(run_attempt, task=task, agent_temp=temps[i]): i for i in range(n_attempts)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results.append((i, fut.result()))
            except Exception as exc:
                import traceback; traceback.print_exc()
                results.append((i, None))
    total_dt = time.time() - t0
    results.sort()

    for idx, m in results:
        if m is None:
            print(f"  attempt {idx}: EXCEPTION", flush=True); continue
        r = m["result"]
        ans = r.answer
        n_steps = len(r.steps)
        print(
            f"  attempt {idx} [temp={temps[idx]}]: steps={n_steps}, "
            f"succeeded={r.succeeded}, "
            f"answer_shape={(len(ans.rows), len(ans.columns)) if ans else None}, "
            f"{m['elapsed_s']:.0f}s",
            flush=True,
        )

    answers = [m["result"].answer for _, m in results if m and m["result"].succeeded and m["result"].answer]
    if not answers:
        return {"task": tid, "score": 0.0, "fail": "all attempts failed", "total_latency_s": int(total_dt)}

    final = _vote_answer(answers)
    if final is None:
        return {"task": tid, "score": 0.0, "fail": "vote produced no answer"}

    pred_dir = REPO / "artifacts" / "searchable_doc_poc" / tid
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
    return {
        "task": tid, "score": e.official_score_lambda_0_5,
        "pred_shape": [e.rows_pred, len(final.columns)],
        "n_attempts_succ": len(answers),
        "total_latency_s": int(total_dt),
    }


def get_exp_109_baselines() -> dict[str, dict[str, float]]:
    """Return per-task best/mean from exp_109 historic 3-attempt union runs."""
    rec: dict[str, list[float]] = {}
    for run in ("001", "002", "003"):
        p = REPO / f"artifacts/runs/exp_109_plan_first_strengthen_{run}/evaluation.csv"
        if not p.exists(): continue
        for row in csv.DictReader(open(p)):
            try:
                s = float(row.get("official_score_lambda_0_5") or 0)
            except Exception:
                continue
            rec.setdefault(row["task_id"], []).append(s)
    out = {}
    for tid, scores in rec.items():
        if scores:
            out[tid] = {"best": max(scores), "mean": sum(scores) / len(scores)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_attempts", type=int, default=3)
    args = ap.parse_args()

    e109 = get_exp_109_baselines()
    results = []
    for tid in MIXED_DOC_TASKS:
        try:
            results.append(run_one(tid, n_attempts=args.n_attempts))
        except Exception as exc:
            import traceback; traceback.print_exc()
            results.append({"task": tid, "score": 0.0, "exception": repr(exc)})

    print("\n" + "=" * 80, flush=True)
    print(f"SEARCHABLE-DOC POC SUMMARY (n_attempts={args.n_attempts}, 11 mixed-doc tasks)", flush=True)
    print("=" * 80, flush=True)
    print(f"{'task':<10} {'score':>8} {'e109_best':>10} {'e109_mean':>10} {'vs_best':>8} {'vs_mean':>8}", flush=True)
    sum_d_best = 0.0; sum_d_mean = 0.0
    summary = {}
    n = 0
    for r in results:
        tid = r["task"]; s = r.get("score", 0.0)
        baseline = e109.get(tid, {"best": 0.0, "mean": 0.0})
        b = baseline["best"]; m = baseline["mean"]
        d_best = s - b; d_mean = s - m
        sum_d_best += d_best; sum_d_mean += d_mean
        n += 1
        summary[tid] = {"score": s, "e109_best": b, "e109_mean": m, "d_best": d_best, "d_mean": d_mean}
        print(f"  {tid:<10} {s:>8.3f} {b:>10.3f} {m:>10.3f} {d_best:>+8.3f} {d_mean:>+8.3f}", flush=True)
    if n > 0:
        print(f"\nsum vs e109_best:  {sum_d_best:+.3f} | avg: {sum_d_best/n:+.3f}", flush=True)
        print(f"sum vs e109_mean:  {sum_d_mean:+.3f} | avg: {sum_d_mean/n:+.3f}", flush=True)

    out = REPO / "artifacts" / "searchable_doc_poc" / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"raw": results, "summary": summary, "args": vars(args)}, indent=2, ensure_ascii=False))
    print(f"\nresults: {out}", flush=True)


if __name__ == "__main__":
    main()
