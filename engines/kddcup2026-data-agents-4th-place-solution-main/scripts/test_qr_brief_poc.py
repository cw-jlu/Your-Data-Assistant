"""POC: Question-Relevant Knowledge Brief (= structured-surfacing of knowledge.md).

Runs a single-attempt agent on 5 representative tasks with the brief
prepended to the preamble, and compares to baseline scores from
exp_109 history.

Tasks chosen:
- task_25  (= cost ambiguity, section 6 has "amount" warning) — HIGH expectation
- task_163 (= "type" interpretation, no section 6 hit) — MEDIUM expectation
- task_169 (= scope ambiguity, formula in knowledge.md) — LOW expectation (Pass 1 POC failed)
- task_180 (= multi-condition filter) — MEDIUM expectation
- task_38  (= existing 0.52 baseline, has SQL examples — control)

Usage:
    uv run python scripts/test_qr_brief_poc.py
"""
from __future__ import annotations

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
BRIEF_CACHE = REPO / "artifacts" / "qr_brief_cache"


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


def baseline_score(tid: str) -> float:
    """Best score across exp_109 runs for the same task."""
    import glob
    best = 0.0
    for run_dir in glob.glob(str(REPO / "artifacts/runs/exp_109_plan_first_strengthen_*")):
        ev = Path(run_dir) / "evaluation.csv"
        if not ev.exists(): continue
        for r in csv.DictReader(open(ev)):
            if r["task_id"] == tid:
                try:
                    s = float(r.get("official_score_lambda_0_5", 0) or 0)
                    best = max(best, s)
                except: pass
    return best


def run_one(tid: str, model, tools) -> dict:
    print(f"\n{'='*60}")
    print(f"## {tid}")
    print(f"{'='*60}")
    ds = DABenchPublicDataset(root_dir=REPO / "data" / "public" / "input")
    task = ds.get_task(tid)
    print(f"Q: {task.question}")
    print(f"baseline (exp_109 best): {baseline_score(tid):.4f}")

    base_preamble = build_preamble(task)
    brief = (BRIEF_CACHE / f"{tid}.md").read_text()
    augmented = brief + "\n\n---\n\n" + base_preamble.text

    print(f"brief: {len(brief)} chars  preamble: {base_preamble.char_count} chars  augmented: {len(augmented)} chars")

    agent = ReActAgent(
        model=model,
        tools=tools,
        config=ReActAgentConfig(max_steps=24, min_steps=4),
        preamble=augmented,
    )
    t0 = time.time()
    result = agent.run(task)
    dt = time.time() - t0
    print(f"agent latency: {dt:.0f}s  steps: {len(result.steps)}  succeeded: {result.succeeded}")

    if not result.succeeded or not result.answer:
        print(f"FAIL: {result.failure_reason}")
        return {"task": tid, "score": 0.0, "fail": result.failure_reason}

    pred_dir = REPO / "artifacts" / "qr_brief_test" / tid
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
    print(f"pred: {e.rows_pred}r × cols={result.answer.columns}")
    if result.answer.rows[:2]:
        print(f"  rows[0:2]: {result.answer.rows[:2]}")
    return {
        "task": tid,
        "score": e.official_score_lambda_0_5,
        "baseline": baseline_score(tid),
        "delta": e.official_score_lambda_0_5 - baseline_score(tid),
        "pred_shape": [e.rows_pred, len(result.answer.columns)],
    }


def main():
    model = make_model()
    tools = create_default_tool_registry()
    results = []
    for tid in TASKS:
        try:
            results.append(run_one(tid, model, tools))
        except Exception as exc:
            print(f"  EXCEPTION on {tid}: {exc!r}")
            results.append({"task": tid, "score": 0.0, "exception": repr(exc)})

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'task':<10} {'baseline':>10} {'POC':>10} {'delta':>10}")
    total_delta = 0.0
    for r in results:
        b = r.get("baseline", 0.0)
        s = r.get("score", 0.0)
        d = r.get("delta", s - b)
        total_delta += d
        print(f"  {r['task']:<10} {b:>10.4f} {s:>10.4f} {d:>+10.4f}")
    print(f"\nsum delta: {total_delta:+.4f}")
    print(f"avg delta: {total_delta/len(results):+.4f}")

    out = REPO / "artifacts" / "qr_brief_test" / "results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nresults: {out}")


if __name__ == "__main__":
    main()
