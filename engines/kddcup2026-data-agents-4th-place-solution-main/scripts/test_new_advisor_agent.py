"""Focused agent test with NEW advisor prompt + official scorer.

Differences from `math_advisor_full50.py`:
- NEW advisor system prompt (= pseudo-formula with `|` for naturalized predicates,
  WHERE-uppercase only for filter-back computations)
- Official scorer (= kobushi_core.eval._evaluate_task, λ=0.5 column-multiset metric)
  NOT the row-based score_csv. Per feedback_scorer_unification.md.
- Focused 5-task subset (= rescue + regression + sanity)

Usage:
    set -a && source .env && set +a && uv run python scripts/test_new_advisor_agent.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kobushi_core.eval.csv_compare import _evaluate_task, EvaluationOptions
from kobushi_core.model import OpenAIModelAdapter, ModelMessage

# Reuse exp_137 components (= same agent, same tools, same vote)
from experiments.exp_137_math_advisor.preamble import build_preamble
from experiments.exp_137_math_advisor.phased_agent import (
    PhasedReActAgent,
    PhasedAgentConfig,
)
from experiments.exp_137_math_advisor.tools.registry import create_default_tool_registry
from experiments.exp_137_math_advisor.adaptive_vote import adaptive_vote
from kobushi_core.benchmark import DABenchPublicDataset


NEW_SYS = """You are a math expert. Read the question and decide whether it
requires any calculation (= aggregation, division, multiplication, superlative,
ratio, percentage, count, average).

(a) If no calculation is implied (= pure lookup, name retrieval, list) → output
    only the token: NO_CALC
(b) Otherwise → output one short pseudo-formula using these conventions:

SYNTAX:
- Aggregation operators (UPPERCASE): SUM, COUNT, AVG, MIN, MAX, DISTINCT
- Scaling: *100 for percentage, /N for averages, / for ratios
- Filter-back computation (= compare against an aggregation result):
    use UPPERCASE `WHERE col = MAX(col)` or `WHERE col = MIN(col)`.
    This is a real computation, the agent follows it literally.
- Concept-level predicates (= filter by a literal-valued condition):
    use the math `|` symbol (set-builder "such that"), with natural-language
    descriptors on the right. The agent verifies these via EXPLORE.
    Examples:
      `COUNT(orders | status is late)`
      `SUM(value | quarter is Q4)`
      `COUNT(DISTINCT patient | age is under 70 AND creatinine is abnormal)`
      `consumption | price_per_unit is above 29 AND date is in August 2012`
- Never include SQL-style literal predicates like `WHERE col='value'`.
  Always express literal-valued filters with `|` and natural-language.
- Never invent specific column or table names beyond what the question states.
- Special pattern for ratios "how many times more X for A than B":
    express as `SUM(X | A_filter) / SUM(X | B_filter)` (= division of sums),
    NOT as COUNT. The result is a number, not a count.

Examples:

Q: "What is the capital of France?"
A: NO_CALC

Q: "List the employees in the marketing department"
A: NO_CALC

Q: "What percentage of orders are shipped late?"
A: result = COUNT(orders | status is late) * 100.0 / COUNT(orders)

Q: "Which player has the best score?"
A: result = player WHERE score = MAX(score)

Q: "Who is the fastest runner?"
A: result = runner WHERE time = MIN(time)

Q: "What is the average daily traffic in 2020?"
A: result = AVG(daily_traffic | year is 2020)

Q: "How many times was Q4 sales more than Q3 sales?" (= ratio)
A: result = SUM(sales | quarter is Q4) / SUM(sales | quarter is Q3)

Q: "How many times more X for region A than region B?" (= ratio)
A: result = SUM(X | region is A) / SUM(X | region is B)

Q: "Average yearly rainfall in 1995"
A: result = AVG(yearly_rainfall | year is 1995)

Q: "How many distinct patients have abnormal creatinine and are under 70?"
A: result = COUNT(DISTINCT patient | creatinine is abnormal AND age is under 70)

Output: NO_CALC OR `result = ...` formula. ONLY that. Be SHORT."""


def make_advisor():
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


def make_agent_model(temp=0.6):
    return OpenAIModelAdapter(
        model="qwen3.5-35b-a3b",
        api_base=os.environ["AGENT_API_BASE"],
        api_key=os.environ["AGENT_API_KEY"],
        temperature=temp,
        extra_headers={
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
    )


def get_formula_new(question: str) -> str:
    m = make_advisor()
    r = m.complete(
        [ModelMessage(role="system", content=NEW_SYS),
         ModelMessage(role="user", content=f"Q: {question}\n\nA:")],
        enable_thinking=False,
        max_tokens=400,
    )
    return r.strip()[:500]


HINT_TEMPLATE = (
    "# MATH HINT\n"
    "{formula}\n\n"
    "WHERE (uppercase) = SQL filter-back computation — follow literally.\n"
    "`|` (such that) = concept predicate — verify column names and value\n"
    "formats in EXPLORE before turning into SQL.\n\n"
)


def run_one(tid: str, n_attempts: int = 2):
    ds = DABenchPublicDataset(root_dir=ROOT / "data" / "public" / "input")
    task = ds.get_task(tid)
    formula = get_formula_new(task.question)
    skip = formula.strip().upper().startswith("NO_CALC")

    answers = []
    for i in range(n_attempts):
        m = make_agent_model(temp=0.6)
        preamble = build_preamble(task)
        if skip:
            injected = preamble.text
        else:
            injected = HINT_TEMPLATE.format(formula=formula) + preamble.text
        tools = create_default_tool_registry(
            auditor_model=m,
            question_provider=lambda: task.question,
            context_dir=task.context_dir,
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
            print(f"  [{tid}] attempt {i} error: {str(e)[:200]}")

    if not answers:
        return {"tid": tid, "formula": formula, "official_score": 0.0, "n_ok": 0}

    voted = adaptive_vote(answers)

    out_dir = ROOT / "artifacts" / "test_new_advisor" / tid
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "prediction.csv"
    import csv as _csv
    with pred_path.open("w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(voted.columns)
        for row in voted.rows:
            w.writerow(row)
    (out_dir / "formula.txt").write_text(formula)
    (out_dir / "skip_advisor.txt").write_text(str(skip))

    gold_path = ROOT / "data" / "public" / "output" / tid / "gold.csv"
    e = _evaluate_task(task_id=tid, prediction_path=pred_path, gold_path=gold_path,
                       options=EvaluationOptions())
    return {
        "tid": tid,
        "formula": formula,
        "skip_advisor": skip,
        "official_score": float(e.official_score_lambda_0_5),
        "recall": float(e.recall),
        "extras_ratio": float(e.extras_ratio),
        "matched_cols": e.matched_cols,
        "gold_cols": e.gold_cols,
        "pred_cols": e.pred_cols,
        "n_ok": len(answers),
    }


def main():
    # Target tasks: rescue + regression + sanity + new test
    target_tids = [
        "task_418",  # CRITICAL: regression to fix (= advisor formula 'abnormal' as literal)
        "task_180",  # rescue (= multi-filter listing)
        "task_25",   # rescue (= filter-back)
        "task_352",  # NEW: ratio task where OLD prompt failed
        "task_22",   # NO_CALC sanity
    ]
    print(f"=== NEW advisor prompt focused test ({len(target_tids)} tasks × 2 attempts) ===")
    results = []
    t0 = time.time()
    for tid in target_tids:
        ts = time.time()
        r = run_one(tid)
        elapsed = time.time() - ts
        r["elapsed_s"] = round(elapsed, 1)
        results.append(r)
        f_short = r["formula"][:80]
        print(f"  [{tid}] score={r['official_score']:.3f}  formula='{f_short}'  t={elapsed:.0f}s")

    total_elapsed = time.time() - t0
    out = ROOT / "artifacts" / "test_new_advisor" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "results": results,
        "mean": sum(r["official_score"] for r in results) / len(results),
        "total_elapsed_s": round(total_elapsed, 1),
    }
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nMean score: {summary['mean']:.3f}")
    print(f"Total elapsed: {total_elapsed/60:.1f} min")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
