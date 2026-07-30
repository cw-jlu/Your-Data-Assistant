"""Run exp_122_column_auditor (= with concat guard) on full-name tasks only.

Concurrent with main bench (= use low workers to avoid contention).
"""
from __future__ import annotations

import csv, importlib, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

from kobushi_core.benchmark.dataset import DABenchPublicDataset
from kobushi_core.model import OpenAIModelAdapter
from experiments.exp_122_column_auditor.phased_agent import (
    PhasedReActAgent, PhasedAgentConfig,
)
from experiments.exp_122_column_auditor.preamble import build_preamble
from experiments.exp_122_column_auditor.tools.registry import create_default_tool_registry
from experiments.exp_122_column_auditor.adaptive_vote import adaptive_vote

OUT = REPO / "artifacts" / "concat_guard_test"
OUT.mkdir(parents=True, exist_ok=True)

TASKS = ["task_19", "task_27", "task_74", "task_355"]
N_ATTEMPTS = 3
MAX_WORKERS = 1  # = 3 streams concurrent with main bench (= 20 streams) → 23 total


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
    )


def norm(rows):
    out = set()
    for r in rows:
        nr = tuple(re.sub(r"\.0+$", "", str(x).strip().lower()) for x in r)
        nr = tuple(f"{float(v):.2f}" if re.match(r"^-?\d+\.\d+$", v) else v for v in nr)
        out.add(nr)
    return out


def score_csv(pred, gold):
    p = list(csv.reader(open(pred))); g = list(csv.reader(open(gold)))
    if len(p)<2 or len(g)<2: return 0.0
    P, G = norm(p[1:]), norm(g[1:])
    if not G: return 0.0
    return max(0.0, len(P&G)/len(G) - 0.5*(len(P-G)/max(1,len(P))))


def run_one_task(tid: str) -> dict:
    ds = DABenchPublicDataset(root_dir=REPO/"data"/"public"/"input")
    task = ds.get_task(tid)
    answers = []
    attempt_traces = []
    for i in range(N_ATTEMPTS):
        model = make_model()
        preamble = build_preamble(task)
        tools = create_default_tool_registry(
            auditor_model=model,
            question_provider=lambda: task.question,
            context_dir=task.context_dir,
        )
        agent = PhasedReActAgent(
            model=model, tools=tools,
            config=PhasedAgentConfig(max_steps=32, min_explore_queries=3),
            preamble=preamble.text,
        )
        try:
            r = agent.run(task)
            if r.answer and r.answer.rows:
                answers.append(r.answer)
            # Save action sequence + check if guard fired
            actions = [(s.action, str(s.action_input)[:120], "RULE16" if "Rule 16" in str(s.observation) else "") for s in r.steps]
            attempt_traces.append(actions)
        except Exception as e:
            print(f"  {tid} att{i}: ERROR {str(e)[:100]}")
            attempt_traces.append([("ERROR", str(e)[:100], "")])
    # Save trace summary per task
    task_dir = OUT / tid
    task_dir.mkdir(parents=True, exist_ok=True)
    with open(task_dir / "trace_summary.txt", "w") as f:
        for i, actions in enumerate(attempt_traces):
            f.write(f"=== attempt {i} ===\n")
            for j, (a, ai, flag) in enumerate(actions):
                f.write(f"  step {j}: {a} {flag}\n    input: {ai}\n")
    if not answers:
        return {"task_id": tid, "score": 0.0, "n_ok": 0, "err": "no answers"}
    voted = adaptive_vote(answers)
    if voted is None or not voted.rows:
        return {"task_id": tid, "score": 0.0, "n_ok": len(answers), "err": "voted empty"}
    task_dir = OUT / tid
    task_dir.mkdir(parents=True, exist_ok=True)
    pred = task_dir / "prediction.csv"
    with open(pred, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(voted.columns)
        for row in voted.rows:
            w.writerow(row)
    gold = REPO/"data"/"public"/"output"/tid/"gold.csv"
    s = score_csv(pred, gold)
    return {"task_id": tid, "score": s, "n_ok": len(answers),
            "cols": voted.columns, "first_row": voted.rows[0] if voted.rows else None}


def main():
    print(f"=== concat guard test on {TASKS} ===", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(run_one_task, t): t for t in TASKS}
        for fut in as_completed(futs):
            tid = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"task_id": tid, "score": 0.0, "err": str(e)[:200]}
            results.append(r)
            print(f"  {r['task_id']}: score={r.get('score',0):.2f} cols={r.get('cols','?')} first={r.get('first_row','?')}", flush=True)
    with open(OUT/"results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nmean={sum(r.get('score',0) for r in results)/len(results):.3f}")


if __name__ == "__main__":
    main()
