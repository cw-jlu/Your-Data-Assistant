"""Re-run task_418 (= the only doc-only task) 5 times to estimate variance."""
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
from experiments.exp_120_searchable_doc.agent import ReActAgent, ReActAgentConfig
from experiments.exp_120_searchable_doc.preamble import build_preamble
from experiments.exp_120_searchable_doc.tools.registry import create_default_tool_registry


TID = "task_418"
N_RUNS = 3


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


def _serialize_step(s):
    return {
        "step_index": s.step_index,
        "thought": s.thought[:1500],
        "action": s.action,
        "action_input": s.action_input,
        "ok": s.ok,
        "observation": s.observation,
    }


def run_one_3attempt(task, run_idx):
    print(f"\n## run {run_idx}", flush=True)
    temps = [0.6, 0.6, 0.7]
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(run_attempt, task, temps[i]): i for i in range(3)}
        for fut in as_completed(futs):
            i = futs[fut]
            try: results.append((i, fut.result()))
            except Exception as exc:
                print(f"  attempt {i}: EXC {repr(exc)[:120]}", flush=True)
                results.append((i, None))
    answers = []
    pred_dir = REPO / "artifacts" / "task_418_sample" / f"run_{run_idx}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    # Save per-attempt trace
    for i, r in results:
        if r is None:
            (pred_dir / f"attempt_{i}_trace.json").write_text(json.dumps({"failed": True}))
            continue
        ans_shape = (len(r.answer.rows), len(r.answer.columns)) if r.answer else None
        ans_preview = r.answer.rows[:3] if r.answer else None
        print(f"  attempt {i} [temp={temps[i]}]: succeeded={r.succeeded} answer={ans_shape} preview={ans_preview}", flush=True)
        # serialize trace
        trace = {
            "attempt": i,
            "temp": temps[i],
            "succeeded": r.succeeded,
            "n_steps": len(r.steps),
            "failure_reason": r.failure_reason,
            "answer_columns": list(r.answer.columns) if r.answer else None,
            "answer_rows": [list(row) for row in r.answer.rows] if r.answer else None,
            "steps": [_serialize_step(s) for s in r.steps],
        }
        (pred_dir / f"attempt_{i}_trace.json").write_text(json.dumps(trace, indent=2, default=str))
        if r.succeeded and r.answer:
            answers.append(r.answer)
    if not answers:
        return {"run": run_idx, "score": 0.0, "fail": "all attempts failed"}
    final = _vote(answers)
    pred_p = pred_dir / "prediction.csv"
    with open(pred_p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(final.columns)
        for r in final.rows: w.writerow(r)
    e = _evaluate_task(
        task_id=TID, prediction_path=pred_p,
        gold_path=REPO / "data" / "public" / "output" / TID / "gold.csv",
        options=EvaluationOptions(),
    )
    print(f"  vote: cols={final.columns} n_rows={len(final.rows)}", flush=True)
    print(f"  score: λ0.5={e.official_score_lambda_0_5:.4f}", flush=True)
    return {
        "run": run_idx, "score": e.official_score_lambda_0_5,
        "shape": [e.rows_pred, len(final.columns)],
        "rows_preview": final.rows[:5],
        "latency_s": int(time.time() - t0),
    }


def main():
    ds = DABenchPublicDataset(root_dir=REPO / "data" / "public" / "input")
    task = ds.get_task(TID)
    print(f"Q: {task.question}", flush=True)

    results = []
    for i in range(N_RUNS):
        try:
            results.append(run_one_3attempt(task, i))
        except Exception as exc:
            import traceback; traceback.print_exc()
            results.append({"run": i, "score": 0.0, "exception": repr(exc)})

    print("\n" + "=" * 60, flush=True)
    print(f"task_418 × {N_RUNS} runs (= each is 3-attempt union)", flush=True)
    print("=" * 60, flush=True)
    for r in results:
        print(f"  run {r['run']}: λ0.5={r.get('score',0):.4f}, shape={r.get('shape')}, rows={r.get('rows_preview')}", flush=True)
    scores = [r.get("score", 0) for r in results]
    mean = sum(scores) / len(scores)
    print(f"\n  mean: {mean:.4f}, min: {min(scores):.4f}, max: {max(scores):.4f}", flush=True)

    out = REPO / "artifacts" / "task_418_sample" / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
