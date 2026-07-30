"""Math expert sub-agent for exp_137.

`generate_formula(question)` returns either:
  - "NO_CALC" (= question is pure lookup, no math needed → caller skips injection)
  - "result = ..." pseudo-formula (= structural template for SQL)

Uses a single OpenAIModelAdapter call at temperature=0 with non-leaky examples.
"""
from __future__ import annotations

import os
from kobushi_core.model import OpenAIModelAdapter, ModelMessage
from experiments.exp_146_modality.prefix_cache import with_prefix_cache_header


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

Row-preserving requests (show/list/records/rows/details/是什么样)
are NO_CALC unless an aggregate is explicit.

For YES cases, make explicit:
  - aggregation (SUM, COUNT, AVG, MIN, MAX) only when explicitly requested
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

Q: "How many times was Q4 sales more than Q3 sales?" (= ratio, NOT difference)
A: result = SUM(value WHERE quarter='Q4') / SUM(value WHERE quarter='Q3')

Q: "Average yearly rainfall in 1995"
A: result = AVG(rainfall)  -- per-year records, use AVG; if monthly aggregates, use SUM/12

Output: NO_CALC OR formula line. ONLY that."""


def _make_model(task_id: str | None = None) -> OpenAIModelAdapter:
    api_base = os.environ.get("AGENT_API_BASE") or os.environ.get("MODEL_API_URL") or ""
    api_key = os.environ.get("AGENT_API_KEY") or os.environ.get("MODEL_API_KEY") or ""
    model_name = os.environ.get("AGENT_MODEL") or os.environ.get("MODEL_NAME", "qwen3.5-35b-a3b")
    headers = with_prefix_cache_header(
        {
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
        task_id,
    )
    return OpenAIModelAdapter(
        model=model_name,
        api_base=api_base,
        api_key=api_key,
        temperature=0.0,
        extra_headers=headers,
    )


def generate_formula(question: str, task_id: str | None = None) -> str:
    """Generate pseudo-math formula or NO_CALC token."""
    try:
        model = _make_model(task_id=task_id)
        r = model.complete(
            [
                ModelMessage(role="system", content=_EXPERT_SYS),
                ModelMessage(role="user", content=f"Q: {question}\n\nA:"),
            ],
            enable_thinking=False,
            max_tokens=512,
        )
        return r.strip()
    except Exception:
        return "NO_CALC"
