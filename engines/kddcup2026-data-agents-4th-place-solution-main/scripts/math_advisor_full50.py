"""exp_137 math advisor — full 50-task bench.

Pipeline per task:
  1. math_expert(question) → one-line pseudo-formula
  2. inject formula into preamble
  3. PhasedReActAgent × 3 attempts
  4. adaptive_vote → prediction.csv
  5. trace.json per attempt saved

Baseline = main rerun (exp_122_column_auditor_012) = 0.760.
"""
from __future__ import annotations

import csv, json, os, re, sys, time
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

# Auto-pick next free dir (= avoid overwriting prior runs)
import os as _os
_runs = REPO / "artifacts" / "runs"
_i = 1
while (_runs / f"exp_137_math_advisor_{_i:03d}").exists():
    _i += 1
OUT = _runs / f"exp_137_math_advisor_{_i:03d}"
OUT.mkdir(parents=True, exist_ok=True)
print(f"[init] output dir: {OUT}", flush=True)

N_ATTEMPTS = 3
MAX_WORKERS = 4  # same as original LB config


# Leak-fixed prompt (= different examples from any DABench task wording)
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
  - multiple result columns: if the question asks for "X and Y" as separate
    values, output them comma-separated; never collapse into one aggregate
  - verbatim nouns: preserve any multi-word proper noun (capitalized phrase,
    quoted name, or specific event/entity title) AS-IS as a single literal
    filter value, never split into geographic/category components
    (= "Boston Marathon" → `name='Boston Marathon'`,
       NOT `city='Boston' AND type='marathon'`).

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
   -- use DISTINCT on the unique row identifier when JOINs may duplicate

Q: "How many times was Q4 sales more than Q3 sales?" (= ratio, NOT difference)
A: result = SUM(value WHERE quarter='Q4') / SUM(value WHERE quarter='Q3')

Q: "List the books with the highest rating"
A: result = SELECT books WHERE rating = (SELECT MAX(rating) FROM books)  -- filter-back for ties

Q: "Average yearly rainfall in 1995"
A: result = AVG(rainfall)  -- per-year records, use AVG; if monthly aggregates, use SUM/12

Output the formula line (no JSON, no markdown). Just the formula."""


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
    """DEPRECATED — was row-based set match, not the official column-multiset scorer.
    Caused fake +0.06 v9 gain that vanished under official scoring on 2026-05-18.
    Use kobushi_core.eval._evaluate_task instead.
    """
    raise RuntimeError(
        "score_csv (row-based) is removed. Use kobushi_core.eval._evaluate_task "
        "(= official column-multiset λ=0.5 scorer) so bench numbers match LB metric. "
        "See memory/feedback_scorer_unification.md for context."
    )


def _official_score(pred_path, gold_path):
    """Per-task official λ=0.5 score from kobushi_core (= same metric as LB)."""
    from kobushi_core.eval.csv_compare import _evaluate_task, EvaluationOptions
    e = _evaluate_task(
        task_id=str(pred_path.parent.name),
        prediction_path=pred_path,
        gold_path=gold_path,
        options=EvaluationOptions(),
    )
    return float(e.official_score_lambda_0_5)


def run_one(tid):
    ds = DABenchPublicDataset(root_dir=REPO/"data"/"public"/"input")
    task = ds.get_task(tid)
    model = make_model()
    formula = get_formula(task.question, model)
    task_dir = OUT / tid
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "formula.txt").write_text(formula)
    # NO_CALC branch: skip injection, use raw preamble
    skip_advisor = formula.strip().upper().startswith("NO_CALC")
    (task_dir / "advisor_used.txt").write_text("False" if skip_advisor else "True")
    answers = []
    attempt_traces = []
    for i in range(N_ATTEMPTS):
        m = make_model(temp=0.6)
        preamble = build_preamble(task)
        if skip_advisor:
            injected = preamble.text  # no advisor for non-computation tasks
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
            config=PhasedAgentConfig(max_steps=64, min_explore_queries=3),
            preamble=injected,
        )
        try:
            r = agent.run(task)
            if r.answer and r.answer.rows:
                answers.append(r.answer)
            # Save per-step trace for this attempt
            attempt_traces.append({
                "attempt": i,
                "succeeded": r.succeeded if hasattr(r, "succeeded") else None,
                "answer": {"columns": list(r.answer.columns), "rows": [list(row) for row in r.answer.rows]} if r.answer else None,
                "steps": [
                    {
                        "i": j, "phase": getattr(s, "phase", None),
                        "action": s.action,
                        "action_input": s.action_input,
                        "thought": (s.thought or "")[:2000],
                        "observation_ok": s.observation.get("ok") if isinstance(s.observation, dict) else None,
                        "observation_content_preview": str(s.observation.get("content") if isinstance(s.observation, dict) else s.observation)[:400],
                    }
                    for j, s in enumerate(r.steps)
                ],
            })
        except Exception as e:
            attempt_traces.append({"attempt": i, "error": str(e)[:300]})
    # Save full trace per task
    with open(task_dir / "trace.json", "w") as f:
        json.dump(attempt_traces, f, indent=2, default=str)
    if not answers:
        return {"tid": tid, "formula": formula, "score": 0.0, "n_ok": 0}
    voted = adaptive_vote(answers)
    if voted is None or not voted.rows:
        return {"tid": tid, "formula": formula, "score": 0.0, "n_ok": len(answers)}
    pred = task_dir / "prediction.csv"
    with open(pred, "w", newline="") as f:
        w = csv.writer(f); w.writerow(voted.columns); [w.writerow(r) for r in voted.rows]
    gold = REPO/"data"/"public"/"output"/tid/"gold.csv"
    s = _official_score(pred, gold)  # official column-multiset λ=0.5 (= LB metric)
    return {"tid": tid, "formula": formula, "score": s, "n_ok": len(answers),
            "cols": list(voted.columns), "first_row": list(voted.rows[0]) if voted.rows else None}


def main():
    ds = DABenchPublicDataset(root_dir=REPO/"data"/"public"/"input")
    task_ids = sorted([t.task_id for t in ds.iter_tasks()], key=lambda x: int(x.split("_")[1]))
    print(f"=== math advisor full 50-task bench (workers={MAX_WORKERS}, n_attempts={N_ATTEMPTS}) ===\n", flush=True)
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(run_one, t): t for t in task_ids}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            results.append(r)
            print(f"  [{done:2d}/50] {r['tid']}: score={r.get('score',0):.2f}", flush=True)
    elapsed = time.time() - t0
    mean = sum(r["score"] for r in results) / len(results)
    n_perfect = sum(1 for r in results if r["score"] >= 0.99)
    n_zero = sum(1 for r in results if r["score"] < 0.01)
    print(f"\n=== Summary ===")
    print(f"  mean score: {mean:.4f}")
    print(f"  perfect:    {n_perfect}/50")
    print(f"  zero:       {n_zero}/50")
    print(f"  elapsed:    {int(elapsed/60)}m")
    print(f"\nbaseline (main rerun): 0.760")
    print(f"delta:                 {mean - 0.760:+.4f}")
    with open(OUT/"results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
