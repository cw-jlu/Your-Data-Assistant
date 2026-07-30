"""Retry the 3 failed tasks (task_173, 249, 250) with exp_120."""
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
from kobushi_core.benchmark.schema import AnswerTable
from kobushi_core.eval.csv_compare import _evaluate_task, EvaluationOptions
from kobushi_core.model import OpenAIModelAdapter
from experiments.exp_120_searchable_doc.agent import ReActAgent, ReActAgentConfig
from experiments.exp_120_searchable_doc.preamble import build_preamble
from experiments.exp_120_searchable_doc.tools.registry import create_default_tool_registry


RETRY = ["task_173", "task_249", "task_250"]


def make_model(temp: float):
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


def _norm(v):
    if v is None: return (0, "")
    s = str(v).strip()
    try: return (1, round(float(s), 3))
    except: return (2, s.lower())


def _sig(a):
    if not a or not a.columns: return ("__empty__",)
    rows = [tuple(_norm(v) for v in r) for r in (a.rows or []) if len(r) == len(a.columns)]
    rows.sort()
    return (len(a.columns), len(rows), tuple(rows))


def _vote(answers):
    if not answers: return None
    s2i = {}
    for i, a in enumerate(answers):
        s2i.setdefault(_sig(a), []).append(i)
    best = max(s2i, key=lambda k: len(s2i[k]))
    return answers[s2i[best][0]]


def run_attempt(task, temp):
    model = make_model(temp)
    tools = create_default_tool_registry()
    pre = build_preamble(task)
    agent = ReActAgent(
        model=model, tools=tools,
        config=ReActAgentConfig(max_steps=24, min_steps=4),
        preamble=pre.text,
    )
    return agent.run(task)


def run_one(tid):
    print(f"\n## {tid}", flush=True)
    ds = DABenchPublicDataset(root_dir=REPO / "data" / "public" / "input")
    task = ds.get_task(tid)
    print(f"  Q: {task.question}", flush=True)
    temps = [0.6, 0.6, 0.7]
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(run_attempt, task, temps[i]): i for i in range(3)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results.append((i, fut.result()))
            except Exception as exc:
                print(f"  attempt {i}: EXCEPTION: {repr(exc)[:200]}", flush=True)
                results.append((i, None))
    total_dt = time.time() - t0

    answers = []
    for i, r in results:
        if r is None:
            print(f"  attempt {i}: EXCEPTION", flush=True); continue
        ans_shape = (len(r.answer.rows), len(r.answer.columns)) if r.answer else None
        print(f"  attempt {i}: succeeded={r.succeeded} steps={len(r.steps)} answer={ans_shape}", flush=True)
        if r.succeeded and r.answer:
            answers.append(r.answer)

    if not answers:
        return {"task": tid, "score": 0.0, "fail": "all attempts failed"}
    final = _vote(answers)

    pred_dir = REPO / "artifacts" / "exp_120_retry" / tid
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_p = pred_dir / "prediction.csv"
    with open(pred_p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(final.columns)
        for r in final.rows: w.writerow(r)

    e = _evaluate_task(
        task_id=tid, prediction_path=pred_p,
        gold_path=REPO / "data" / "public" / "output" / tid / "gold.csv",
        options=EvaluationOptions(),
    )
    print(f"  vote winner cols={final.columns} n_rows={len(final.rows)}", flush=True)
    print(f"  score: λ0.5={e.official_score_lambda_0_5:.4f}", flush=True)
    return {"task": tid, "score": e.official_score_lambda_0_5,
            "shape": [e.rows_pred, len(final.columns)], "latency_s": int(total_dt)}


def main():
    results = []
    for tid in RETRY:
        try:
            results.append(run_one(tid))
        except Exception as exc:
            import traceback; traceback.print_exc()
            results.append({"task": tid, "score": 0.0, "exception": repr(exc)})

    print("\n" + "=" * 70, flush=True)
    print("RETRY RESULTS", flush=True)
    print("=" * 70, flush=True)
    for r in results:
        print(f"  {r['task']:<10} → λ0.5={r.get('score',0):.4f}", flush=True)

    out = REPO / "artifacts" / "exp_120_retry" / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nresults: {out}", flush=True)


if __name__ == "__main__":
    main()
