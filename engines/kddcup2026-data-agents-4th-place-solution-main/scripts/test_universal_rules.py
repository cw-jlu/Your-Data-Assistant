"""Test universal rule guards on target failure tasks + sanity tasks."""
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
from experiments.exp_122_column_auditor.phased_agent import PhasedReActAgent, PhasedAgentConfig
from experiments.exp_122_column_auditor.preamble import build_preamble
from experiments.exp_122_column_auditor.tools.registry import create_default_tool_registry
from experiments.exp_122_column_auditor.adaptive_vote import adaptive_vote

OUT = REPO / "artifacts" / "universal_rules_test"
OUT.mkdir(parents=True, exist_ok=True)

# 3 target failures + 5 sanity
TARGETS = ["task_169", "task_180", "task_396"]
SANITY = ["task_24", "task_67", "task_75", "task_287", "task_305"]
ALL = TARGETS + SANITY
N_ATTEMPTS = 3
MAX_WORKERS = 4  # 4 outer × 3 inner = 12 streams (= reduced after congestion seen at 24)


def make_model():
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


def run_one(tid):
    ds = DABenchPublicDataset(root_dir=REPO/"data"/"public"/"input")
    task = ds.get_task(tid)
    answers = []
    for i in range(N_ATTEMPTS):
        model = make_model()
        preamble = build_preamble(task)
        tools = create_default_tool_registry(
            auditor_model=model, question_provider=lambda: task.question, context_dir=task.context_dir,
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
        except Exception as e:
            pass
    if not answers:
        return {"task_id": tid, "score": 0.0}
    voted = adaptive_vote(answers)
    if voted is None or not voted.rows:
        return {"task_id": tid, "score": 0.0}
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
    return {"task_id": tid, "score": s, "cols": voted.columns, "first_row": voted.rows[0] if voted.rows else None}


def main():
    print(f"=== universal rules test: {len(ALL)} tasks ===")
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(run_one, t): t for t in ALL}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            label = "TARGET" if r["task_id"] in TARGETS else "sanity"
            print(f"  [{label}] {r['task_id']}: score={r.get('score',0):.2f}", flush=True)
    targets_mean = sum(r["score"] for r in results if r["task_id"] in TARGETS) / len(TARGETS)
    sanity_mean = sum(r["score"] for r in results if r["task_id"] in SANITY) / len(SANITY)
    print(f"\ntargets mean = {targets_mean:.3f} (vs main rerun 0.000)")
    print(f"sanity  mean = {sanity_mean:.3f} (vs main rerun: expect ~1.0)")
    with open(OUT/"results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
