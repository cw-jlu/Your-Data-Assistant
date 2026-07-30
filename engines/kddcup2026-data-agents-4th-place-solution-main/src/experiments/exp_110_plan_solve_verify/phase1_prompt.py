"""Phase 1 (Plan generation) prompt + JSON parser for exp_110.

Single LLM call (no tools). Extended thinking. Output: structured plan JSON.
The plan binds the downstream Phase 2 ReAct execution.
"""
from __future__ import annotations

import json
import re
from typing import Any

from kobushi_core.model import ModelAdapter, ModelMessage, strip_thinking


PHASE1_SYSTEM_PROMPT = """\
You are a data-analysis planner. Given a question and a dataset overview,
produce a structured plan that guides a downstream agent to compute the
correct answer.

Use extended thinking to:
- Resolve ambiguous nouns by enumerating 2-3 plausible interpretations
- Identify the calculation outline (= what aggregations / joins are needed)
- Estimate the answer's structural shape (column_count, per_column,
  row_count_category, expected magnitude/range when applicable)
- Identify which files / tables / columns to focus on
- Identify information NEEDED for the calculation that is NOT yet visible
  in the preamble or knowledge.md (= missing_information). For each,
  state why it's needed, how to obtain it, and a fallback if obtaining
  fails. Common cases: medical/business thresholds (e.g. "normal range
  for WBC"), date format details, ambiguous column semantics.
- Order the first 3 exploration actions Phase 2 should take, focusing
  on resolving the missing_information items first (= exploration_priorities).

Output JSON only, fenced with ```json. The output must be a single JSON
object with this exact structure:

{
  "column_count": <int>,
  "per_column": ["<col1 description>", ...],
  "row_count_category": "single" | "tied-multiple" | "list-filter-subset" | "all-rows",
  "expected_row_range": "<rough estimate, e.g. '1', '5-15', '50-200'>",
  "interpretations": [
    {"term": "<noun>", "primary": "<chosen meaning>", "alternatives": ["<other1>", "<other2>"], "rationale": "<why primary>"}
  ],
  "target_files": ["<rel path or table name>", ...],
  "key_filter_conditions": [
    {"column": "<col>", "predicate": "<e.g. = 'SME'>"}
  ],
  "calculation_outline": "<one-sentence pseudo-formula>",
  "expected_magnitude": "<rough scale of final value if scalar, e.g. '0-100', '1000-10000', or 'N/A'>",
  "verify_hints": [
    "<sanity check the verifier should apply, e.g. 'value should be percentage 0-100'>"
  ],
  "missing_information": [
    {
      "what": "<concrete information needed but NOT yet in preamble>",
      "why_needed": "<which step requires it>",
      "how_to_find": "<which tool/file/heuristic should resolve it>",
      "fallback": "<what to do if resolution fails>"
    }
  ],
  "exploration_priorities": [
    "<step-1 action: e.g. 'inspect_sqlite_schema on db/atom.db to confirm join key'>",
    "<step-2 action>",
    "<step-3 action>"
  ]
}

Rules:
- Output JSON only inside a single ```json ... ``` fence.
- All keys are mandatory. Use "" / [] / {} for genuinely empty content.
- `interpretations` may be [] if no ambiguous nouns.
- `missing_information` may be [] if everything is in the preamble.
- `exploration_priorities` should always have 1-5 items.
- Be CONCRETE: name actual files/columns/values, not abstract placeholders.

CRITICAL classification rules:

1. SUPERLATIVE → "tied-multiple". If the question contains any superlative
   word (lowest, highest, least, most, minimum, maximum, cheapest, largest,
   smallest, fastest, slowest, oldest, newest, best, worst, top, bottom,
   first, last), set:
       "row_count_category": "tied-multiple"
   "single" is ONLY allowed when the question explicitly says "the single"
   or "any one". Multiple events/items can tie on the extreme value, and
   gold answers reliably include all tied rows. Defaulting to "single"
   here causes catastrophic recall failure.

2. COLUMN_COUNT must match what the question literally asks for, not what
   would be "useful to show". E.g. "Which event has the lowest cost?"
   asks for the event (1 column), not (event, cost) — the cost is the
   filter criterion, not part of the answer. Re-read the question and
   list ONLY the entities/values it requests.

3. AGGREGATE QUESTIONS ("How many", "What is the total", "What is the
   average", "What is the percentage") are typically column_count=1,
   row_count_category="single".

4. LIST QUESTIONS ("List all X", "Which X are ...", "What are the X")
   that filter without superlative are column_count=1 or 2,
   row_count_category="list-filter-subset".
"""


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_FENCE_UNCLOSED_RE = re.compile(r"```(?:json)?\s*(.*)", re.DOTALL | re.IGNORECASE)


def _strip_fence(text: str) -> str:
    text = text.strip()
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _FENCE_UNCLOSED_RE.search(text)
    if m:
        return m.group(1).strip()
    return text


def parse_plan_json(raw: str) -> dict[str, Any]:
    """Parse the plan JSON from raw LLM output. Tolerant of fence variants
    and trailing junk (= same approach as agent.py:parse_model_step)."""
    text = strip_thinking(raw)
    text = _strip_fence(text)
    payload, end = json.JSONDecoder().raw_decode(text)
    if not isinstance(payload, dict):
        raise ValueError("Plan must be a JSON object.")
    # Backfill mandatory keys with safe defaults so downstream code doesn't crash
    payload.setdefault("column_count", 1)
    payload.setdefault("per_column", [])
    payload.setdefault("row_count_category", "single")
    payload.setdefault("expected_row_range", "1")
    payload.setdefault("interpretations", [])
    payload.setdefault("target_files", [])
    payload.setdefault("key_filter_conditions", [])
    payload.setdefault("calculation_outline", "")
    payload.setdefault("expected_magnitude", "N/A")
    payload.setdefault("verify_hints", [])
    payload.setdefault("missing_information", [])
    payload.setdefault("exploration_priorities", [])
    return payload


def generate_plan(
    *,
    question: str,
    preamble: str,
    model: ModelAdapter,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run Phase 1 plan generation.

    Returns (plan_dict, metadata) where metadata captures latency / parse
    success / raw response excerpt for the trace.
    """
    import time

    user_msg = (
        f"{preamble}\n\n---\n\nQuestion: {question}\n\n"
        f"Generate the plan JSON inside a ```json fence."
    )
    messages = [
        ModelMessage(role="system", content=PHASE1_SYSTEM_PROMPT),
        ModelMessage(role="user", content=user_msg),
    ]
    t0 = time.time()
    # Phase 1 thinking budget: VERY generous (~192K tokens) so the planner
    # can think deeply about interpretations, missing_information,
    # exploration order, and ambiguities. The vLLM max_model_len=262144
    # leaves ~70K input headroom for preamble + system prompt, which is
    # enough for any task in DABench public-50 (= rich preamble caps near
    # 50-60K chars = 15K tokens worst case).
    raw = model.complete(messages, enable_thinking=True, max_tokens=196608)
    dt = time.time() - t0

    parse_error: str | None = None
    try:
        plan = parse_plan_json(raw)
    except Exception as exc:
        parse_error = repr(exc)
        # Fallback to a minimal default plan so downstream phases can still run
        plan = parse_plan_json("{}")

    meta = {
        "phase": "plan",
        "latency_s": round(dt, 2),
        "raw_excerpt": raw[:1500],
        "parse_error": parse_error,
        "n_interpretations": len(plan.get("interpretations") or []),
        "n_missing_info": len(plan.get("missing_information") or []),
        "n_exploration_priorities": len(plan.get("exploration_priorities") or []),
    }
    return plan, meta
