"""POC: math expert as advisor (= pseudo-formula injected into agent preamble).

Pipeline:
  1. math_expert(question) → pseudo-formula
  2. Inject formula into agent preamble (= MATH FORMULA HINT section)
  3. Agent runs with formula visible → produces SQL
  4. Score against gold

Test on:
  - target failures: task_169 (avg/12), task_180 (per unit), task_396 (%)
  - sanity: task_24, task_67 (passing without help, ensure no regression)
"""
from __future__ import annotations

import csv, json, os, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from dotenv import load_dotenv
load_dotenv(REPO / ".env")

from kobushi_core.benchmark.dataset import DABenchPublicDataset
from kobushi_core.model import OpenAIModelAdapter, ModelMessage
from experiments.exp_122_column_auditor.phased_agent import PhasedReActAgent, PhasedAgentConfig
from experiments.exp_122_column_auditor.preamble import build_preamble
from experiments.exp_122_column_auditor.tools.registry import create_default_tool_registry
from experiments.exp_122_column_auditor.adaptive_vote import adaptive_vote

OUT = REPO / "artifacts" / "math_advisor_test"
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = ["task_169", "task_180", "task_396"]  # 3 known formula-failure tasks
SANITY = ["task_67", "task_283", "task_243"]    # ratio/avg/percentage that PASS today
ALL = TARGETS + SANITY
N_ATTEMPTS = 3
MAX_WORKERS = 4


_EXPERT_SYS = """You are a math/SQL expert. Read the question and write the EXPECTED
calculation as a pseudo-math formula. Focus on the calculation structure.

Make explicit:
  - aggregation (SUM, COUNT, AVG, MIN, MAX)
  - filters on numerator vs denominator
  - multiplication factors (*100 for %)
  - division (for averages, ratios, per-unit)
  - DISTINCT scope

Output: ONE-LINE pseudo-formula, no JSON wrapping. Be SHORT.

Examples (= unrelated topics):
Q: "What percentage of orders are shipped late?"
A: result = COUNT(orders WHERE status='late') * 100.0 / COUNT(orders)

Q: "What is the average daily traffic in 2020?"
A: result = AVG(visits_per_day) if records are per-day, else SUM(yearly_visits)/365

Q: "How many times did France win compared to Spain?"
A: result = COUNT(DISTINCT match_id WHERE winner='France') / COUNT(DISTINCT match_id WHERE winner='Spain')
   -- use DISTINCT on the unique row identifier when JOINs may duplicate

Q: "How many times was Q4 sales more than Q3 sales?" (= ratio, NOT difference)
A: result = SUM(value WHERE quarter='Q4') / SUM(value WHERE quarter='Q3')

Q: "List the books with the highest rating"
A: result = SELECT books WHERE rating = (SELECT MAX(rating) FROM books)  -- filter-back for ties

Q: "Average yearly rainfall in 1995"
A: result = AVG(rainfall)  -- per-year records, use AVG; if monthly aggregates, use SUM/12

Output the formula line (no JSON, no markdown). Just the formula."""


def make_model():
    return OpenAIModelAdapter(
        model="qwen3.5-35b-a3b",
        api_base=os.environ["AGENT_API_BASE"],
        api_key=os.environ["AGENT_API_KEY"],
        temperature=0.0,
        extra_headers={
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
    )


def get_formula(question, model):
    r = model.complete(
        [ModelMessage(role="system", content=_EXPERT_SYS),
         ModelMessage(role="user", content=f"Q: {question}\n\nA:")],
        enable_thinking=False, max_tokens=300)
    return r.strip()[:400]


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
    model = make_model()
    formula = get_formula(task.question, model)

    answers = []
    for i in range(N_ATTEMPTS):
        m = make_model()
        # use temperature 0.6 for attempts
        m.temperature = 0.6
        preamble = build_preamble(task)
        # Inject math formula hint at the top of preamble
        injected = (
            "# MATH FORMULA HINT (= expert calculation guide)\n"
            "Use this pseudo-math formula as the structural template for your SQL:\n"
            f"  {formula}\n\n"
            "Follow this aggregation/division/filter structure EXACTLY. If the formula\n"
            "says AVG, use AVG (not SUM/N). If it says *100, include *100. If it says\n"
            "filter-back, use (SELECT MIN/MAX) subquery (not LIMIT 1).\n\n"
        ) + preamble.text
        tools = create_default_tool_registry(
            auditor_model=m, question_provider=lambda: task.question, context_dir=task.context_dir,
        )
        agent = PhasedReActAgent(
            model=m, tools=tools,
            config=PhasedAgentConfig(max_steps=32, min_explore_queries=3),
            preamble=injected,
        )
        try:
            r = agent.run(task)
            if r.answer and r.answer.rows:
                answers.append(r.answer)
        except Exception as e:
            pass
    if not answers:
        return {"tid": tid, "formula": formula, "score": 0.0}
    voted = adaptive_vote(answers)
    if voted is None or not voted.rows:
        return {"tid": tid, "formula": formula, "score": 0.0}
    task_dir = OUT / tid
    task_dir.mkdir(parents=True, exist_ok=True)
    pred = task_dir / "prediction.csv"
    with open(pred, "w", newline="") as f:
        w = csv.writer(f); w.writerow(voted.columns); [w.writerow(r) for r in voted.rows]
    gold = REPO/"data"/"public"/"output"/tid/"gold.csv"
    s = score_csv(pred, gold)
    return {"tid": tid, "formula": formula, "score": s,
            "cols": voted.columns, "first_row": voted.rows[0] if voted.rows else None}


def main():
    print(f"=== math advisor agent test: {len(ALL)} tasks ===\n")
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(run_one, t): t for t in ALL}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            label = "TGT" if r["tid"] in TARGETS else "san"
            print(f"  [{label}] {r['tid']}: score={r.get('score',0):.2f}", flush=True)
            print(f"        formula: {r.get('formula','')[:100]}", flush=True)
    tg = sum(r["score"] for r in results if r["tid"] in TARGETS) / len(TARGETS)
    sn = sum(r["score"] for r in results if r["tid"] in SANITY) / len(SANITY)
    print(f"\ntargets mean: {tg:.3f} (baseline was 0.00 for failures)")
    print(f"sanity  mean: {sn:.3f} (baseline ~1.0)")
    with open(OUT/"results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
