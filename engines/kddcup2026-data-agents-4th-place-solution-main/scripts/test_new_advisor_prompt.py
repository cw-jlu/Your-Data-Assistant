"""Quick test of revised math advisor prompt.

Generates formulas for a representative set of tasks under both OLD and NEW
prompts, side-by-side, so we can compare formula quality before running a full
agent bench.

NEW prompt rules:
- SQL WHERE (uppercase) kept only for filter-back computations
  (e.g. WHERE score = MAX(score))
- Math `|` (set-builder) used for literal-valued predicates
  (e.g. COUNT(orders | status is late))
- The right side of `is`/`above`/`below`/`between` is a concept hint, not
  a SQL value. Agent verifies via EXPLORE.

Output: per-task formula from OLD vs NEW prompt. No agent run, no scoring.
"""
from __future__ import annotations

import os
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kobushi_core.model import OpenAIModelAdapter, ModelMessage  # noqa: E402


OLD_SYS = """You are a math/SQL expert. Read the question and decide:
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
  - multiple result columns

Output: NO_CALC OR one-line "result = ..." formula. No JSON wrapping. Be SHORT.

Examples:
Q: "What percentage of orders are shipped late?"
A: result = COUNT(orders WHERE status='late') * 100.0 / COUNT(orders)

Q: "Which player has the best score?"
A: result = SELECT player WHERE score = (SELECT MAX(score) FROM table)

Q: "How many times was Q4 sales more than Q3 sales?" (= ratio, NOT difference)
A: result = SUM(value WHERE quarter='Q4') / SUM(value WHERE quarter='Q3')

Output: NO_CALC OR formula line. ONLY that."""


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

Q: "Average yearly rainfall in 1995"
A: result = AVG(yearly_rainfall | year is 1995)

Q: "How many distinct patients have abnormal creatinine and are under 70?"
A: result = COUNT(DISTINCT patient | creatinine is abnormal AND age is under 70)

Output: NO_CALC OR `result = ...` formula. ONLY that. Be SHORT."""


def make_model():
    api_base = os.environ.get("AGENT_API_BASE") or os.environ.get("MODEL_API_URL")
    api_key = os.environ.get("AGENT_API_KEY") or os.environ.get("MODEL_API_KEY")
    model_name = os.environ.get("AGENT_MODEL") or os.environ.get("MODEL_NAME", "qwen3.5-35b-a3b")
    return OpenAIModelAdapter(
        model=model_name,
        api_base=api_base,
        api_key=api_key,
        temperature=0.0,
        extra_headers={
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
    )


def call_advisor(model, sys_prompt, question):
    r = model.complete(
        [ModelMessage(role="system", content=sys_prompt),
         ModelMessage(role="user", content=f"Q: {question}\n\nA:")],
        enable_thinking=False,
        max_tokens=300,
    )
    return r.strip()[:400]


def main():
    tasks = [
        # rescue under OLD (= keep working under NEW)
        ("task_180", "rescue (per_unit, multi-filter)"),
        ("task_25",  "rescue (filter-back, superlative)"),
        ("task_22",  "rescue (NO_CALC, lookup)"),
        # regression under OLD (= want to fix under NEW)
        ("task_418", "regress (literal mismatch: 'abnormal' is concept not value)"),
        # other CALC patterns to sanity check
        ("task_352", "ratio with literals"),
        ("task_75",  "superlative (best lap)"),
        ("task_169", "complex lookup"),
        ("task_199", "comparison"),
    ]

    model = make_model()
    print(f"{'TID':10s}  {'pattern':45s}")
    print("=" * 80)

    for tid, note in tasks:
        task_json = json.load(open(ROOT / "data" / "public" / "input" / tid / "task.json"))
        q = task_json["question"]
        print(f"\n--- {tid} ({note}) ---")
        print(f"Q: {q}")
        old_f = call_advisor(model, OLD_SYS, q)
        new_f = call_advisor(model, NEW_SYS, q)
        print(f"OLD: {old_f}")
        print(f"NEW: {new_f}")


if __name__ == "__main__":
    main()
