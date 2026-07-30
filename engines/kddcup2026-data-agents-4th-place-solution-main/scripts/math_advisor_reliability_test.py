"""Reliability test:
  - task_180 × 3 outer runs (= confirm rescue is reproducible, not sampling luck)
  - sanity tasks × 1 outer run each (= confirm advisor doesn't regress passing tasks)

Each outer run = 3 inner attempts → adaptive_vote → 1 prediction.
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

OUT = REPO / "artifacts" / "math_advisor_reliability"
OUT.mkdir(parents=True, exist_ok=True)

# Reproducibility: task_180 × 3 outer runs
TARGET_REPEAT = ("task_180", 3)
# FP check: diverse passing tasks × 1 outer run each
SANITY = ["task_19", "task_22", "task_24", "task_67", "task_75",
          "task_243", "task_269", "task_283", "task_287", "task_305"]

N_ATTEMPTS = 3
MAX_WORKERS = 4

_EXPERT_SYS = """You are a math/SQL expert. Read the question and decide:
  (a) Does this question imply ANY of:
      - percentage / ratio / proportion / rate
      - average / sum / count aggregation
      - division (per unit, per X, X / Y, ratio comparison)
      - multiplication factor (*100 for %)
      - superlative / extremum (highest, lowest, best, worst, max, min, top, bottom)
      - "more than", "less than", "compared to", "more than X times"
      - any explicit math operator (+, -, *, /) on column values
  (b) If NO → output ONLY the token: NO_CALC
  (c) If YES → output a one-line pseudo-formula showing the EXPECTED calculation

For YES cases, make explicit:
  - aggregation (SUM, COUNT, AVG, MIN, MAX)
  - filters on numerator vs denominator
  - multiplication factors (*100 for %)
  - division (for averages, ratios, per-unit)
  - DISTINCT scope
  - filter-back for superlatives (= "highest X" → WHERE col = (SELECT MAX(col)))

Output: NO_CALC  OR  one-line "result = ..." formula. No JSON wrapping. Be SHORT.

Examples (= unrelated topics):
Q: "What is the capital of France?"
A: NO_CALC

Q: "List the employees in the marketing department"
A: NO_CALC

Q: "What is the name of the book with ISBN 978-..."
A: NO_CALC

Q: "What percentage of orders are shipped late?"
A: result = COUNT(orders WHERE status='late') * 100.0 / COUNT(orders)

Q: "Which player has the best score?"
A: result = SELECT player WHERE score = (SELECT MAX(score) FROM table)  -- filter-back for ties

Q: "Who is the fastest runner?"
A: result = SELECT runner WHERE time = (SELECT MIN(time) FROM table)  -- "fastest" = MIN(time)

Q: "What is the average daily traffic in 2020?"
A: result = AVG(visits_per_day) if records are per-day, else SUM(yearly_visits)/365

Q: "How many times did France win compared to Spain?"
A: result = COUNT(DISTINCT match_id WHERE winner='France') / COUNT(DISTINCT match_id WHERE winner='Spain')

Q: "How many times was Q4 sales more than Q3 sales?" (= ratio, NOT difference)
A: result = SUM(value WHERE quarter='Q4') / SUM(value WHERE quarter='Q3')

Q: "Average yearly rainfall in 1995"
A: result = AVG(rainfall)  -- per-year records, use AVG; if monthly aggregates, use SUM/12

Output: NO_CALC OR formula line. ONLY that."""


def make_model(temp=0.0):
    return OpenAIModelAdapter(model="qwen3.5-35b-a3b", api_base=os.environ["AGENT_API_BASE"],
        api_key=os.environ["AGENT_API_KEY"], temperature=temp,
        extra_headers={"CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID",""),
                       "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET","")})


def get_formula(question, model):
    r = model.complete([ModelMessage(role="system", content=_EXPERT_SYS),
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


def run_one(tid, run_label):
    ds = DABenchPublicDataset(root_dir=REPO/"data"/"public"/"input")
    task = ds.get_task(tid)
    model = make_model()
    formula = get_formula(task.question, model)
    skip_advisor = formula.strip().upper().startswith("NO_CALC")
    task_dir = OUT / f"{tid}_{run_label}"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "formula.txt").write_text(formula)
    (task_dir / "advisor_used.txt").write_text("False" if skip_advisor else "True")
    def run_attempt(i):
        m = make_model(temp=0.6)
        preamble = build_preamble(task)
        if skip_advisor:
            injected = preamble.text
        else:
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
            tr = {
                "attempt": i,
                "answer": {"columns": list(r.answer.columns), "rows": [list(row) for row in r.answer.rows]} if r.answer else None,
                "steps": [
                    {"i": j, "action": s.action, "action_input": s.action_input,
                     "thought": (s.thought or "")[:2000],
                     "observation_ok": s.observation.get("ok") if isinstance(s.observation, dict) else None,
                     "observation_content_preview": str(s.observation.get("content") if isinstance(s.observation, dict) else s.observation)[:400],
                    } for j, s in enumerate(r.steps)],
            }
            ans = r.answer if (r.answer and r.answer.rows) else None
            return (ans, tr)
        except Exception as e:
            return (None, {"attempt": i, "error": str(e)[:300]})

    # Parallel inner attempts
    from concurrent.futures import ThreadPoolExecutor as _IExec
    answers = []
    attempt_traces = []
    with _IExec(max_workers=N_ATTEMPTS) as iex:
        for ans, tr in iex.map(run_attempt, range(N_ATTEMPTS)):
            if ans is not None: answers.append(ans)
            attempt_traces.append(tr)
    with open(task_dir / "trace.json", "w") as f:
        json.dump(attempt_traces, f, indent=2, default=str)
    if not answers:
        return {"tid": tid, "run": run_label, "formula": formula, "score": 0.0}
    voted = adaptive_vote(answers)
    if voted is None or not voted.rows:
        return {"tid": tid, "run": run_label, "formula": formula, "score": 0.0}
    pred = task_dir / "prediction.csv"
    with open(pred, "w", newline="") as f:
        w = csv.writer(f); w.writerow(voted.columns); [w.writerow(r) for r in voted.rows]
    gold = REPO/"data"/"public"/"output"/tid/"gold.csv"
    s = score_csv(pred, gold)
    return {"tid": tid, "run": run_label, "formula": formula, "score": s,
            "cols": voted.columns, "first_row": voted.rows[0] if voted.rows else None}


def main():
    jobs = [(TARGET_REPEAT[0], f"rep{i}") for i in range(TARGET_REPEAT[1])]
    jobs += [(t, "fp_check") for t in SANITY]
    print(f"=== math advisor reliability: {len(jobs)} runs ({TARGET_REPEAT[1]} target + {len(SANITY)} sanity) ===\n")
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(run_one, t, r): (t, r) for t, r in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            label = "TGT" if r["tid"] == TARGET_REPEAT[0] else "san"
            print(f"  [{label}] {r['tid']} {r['run']}: score={r.get('score',0):.2f}", flush=True)
    target = [r for r in results if r["tid"] == TARGET_REPEAT[0]]
    sanity = [r for r in results if r["tid"] != TARGET_REPEAT[0]]
    t_scores = [r["score"] for r in target]
    s_scores = [r["score"] for r in sanity]
    print(f"\n=== TARGET ({TARGET_REPEAT[0]}) reproducibility ===")
    print(f"  scores: {t_scores}, mean={sum(t_scores)/len(t_scores):.3f}")
    print(f"\n=== SANITY (FP check, n={len(s_scores)}) ===")
    for r in sorted(sanity, key=lambda x: x["tid"]):
        print(f"  {r['tid']}: score={r['score']:.2f}")
    print(f"  sanity mean={sum(s_scores)/len(s_scores):.3f}, n_regressed={sum(1 for s in s_scores if s<1.0)}/{len(s_scores)}")
    with open(OUT/"results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
