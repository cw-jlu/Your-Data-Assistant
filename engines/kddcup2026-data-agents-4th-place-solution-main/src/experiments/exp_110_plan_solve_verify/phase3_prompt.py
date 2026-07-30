"""Phase 3 (Verification) prompt + JSON parser for exp_110.

Single LLM call (no tools). Extended thinking. Output: verdict + optional
fix instruction. Triggers Phase 4 repair when verdict starts with FIX.
"""
from __future__ import annotations

import json
import re
from typing import Any

from kobushi_core.model import ModelAdapter, ModelMessage, strip_thinking


PHASE3_SYSTEM_PROMPT = """\
You are an answer verifier. Given a Phase-1 plan and a candidate answer
(plus a brief execution trace summary), check the following:

1. Column scope: does answer.column_count match plan.column_count?
   If extra columns exist that are NOT in plan.per_column, they must be
   dropped. Use FIX_DROP_COLUMNS for this case (= mechanically removable).

2. Row count plausibility: does the answer's row count fit
   plan.row_count_category and plan.expected_row_range?
   - "single" → expect 1 row (allow 1-2 if ties were not anticipated)
   - "tied-multiple" → expect 1-N rows; the answer should NOT be 1 alone
     unless the underlying data has no ties.
   - "list-filter-subset" → use expected_row_range as guide.
   - "all-rows" → row count = source table row count (be lenient).

3. Magnitude sanity: if plan.expected_magnitude is not "N/A", does the
   scalar answer fit that range? E.g. percentage should be 0-100, monthly
   avg should fit data scale. Out-of-range values likely mean a missed
   calculation step (= FIX_RECOMPUTE).

4. verify_hints: apply each hint as a check.

5. Missing information closure: did the trace resolve every
   plan.missing_information item? An item is resolved when the trace
   shows a tool call that found the value, OR the agent applied the
   stated `fallback`. If a missing item was silently skipped, the answer
   is suspect (= FIX_RECOMPUTE).

Verdict choices (use exactly one):
- "OK" — answer matches plan and passes all checks.
- "FIX_DROP_COLUMNS" — answer has extra columns; mechanical fix possible.
- "FIX_RECOMPUTE" — answer has wrong magnitude / row count / missing
   info skipped. Needs new ReAct turn(s) to fix.
- "FIX_TARGETED" — answer mostly correct but one specific issue
   (e.g. value rounding, format mismatch). Provide fix_instruction.

Output JSON only, fenced with ```json:

{
  "verdict": "OK" | "FIX_DROP_COLUMNS" | "FIX_RECOMPUTE" | "FIX_TARGETED",
  "issues": ["<short description of each issue, [] if OK>"],
  "fix_instruction": "<one-sentence fix; '' if OK>",
  "missing_info_status": [
    {"item": "<missing_info.what verbatim>", "resolved": true/false, "how": "observed|fallback|skipped"}
  ],
  "confidence": "high" | "medium" | "low"
}

Rules:
- Output JSON only inside a single ```json fence.
- Be conservative: prefer "OK" or low confidence FIX over aggressive
  rewrites. False alarms break correct answers.
- "confidence: high" should ONLY be used when you can name a specific
  rule violation (column mismatch, magnitude X vs expected Y, etc.).

CRITICAL anti-false-alarm rules:

A. NEVER force a multi-row answer down to 1 row just because plan
   says "single". If the agent's answer has multiple rows AND the
   question contains a superlative word (lowest, highest, etc.), those
   rows are ALMOST CERTAINLY ties — the plan was wrong, not the answer.
   In this case: verdict = "OK", note in issues that "plan said single
   but answer suggests ties — agent's multi-row answer kept".

B. Plan vs answer mismatch on row_count is a WEAK signal. Only flag
   it when:
   - Answer has 1 row but question is a list/multi-row, OR
   - Answer has 100+ rows when expected_row_range was much smaller
     (= filter scope blow-up).
   Single-digit-row drift in either direction is NOT a violation.

C. FIX_RECOMPUTE should be reserved for clear computational errors:
   - Magnitude wildly off (e.g. answer is 1e8 but expected_magnitude
     is 0-100)
   - The trace shows the agent skipped a missing_information item
     without applying the fallback
   - The trace shows wrong column was used (e.g. used positionText
     when positionOrder was needed)
   Do NOT trigger FIX_RECOMPUTE just to "make answer match plan."

D. If the agent's trace shows it followed the filter-back pattern
   (= `WHERE col = (SELECT MAX(col) FROM t)`) for a superlative
   question and got multiple rows, the answer is CORRECT EVEN IF
   plan.row_count_category was "single". Verdict = "OK".
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


def parse_verdict_json(raw: str) -> dict[str, Any]:
    text = strip_thinking(raw)
    text = _strip_fence(text)
    payload, _ = json.JSONDecoder().raw_decode(text)
    if not isinstance(payload, dict):
        raise ValueError("Verdict must be a JSON object.")
    payload.setdefault("verdict", "OK")
    payload.setdefault("issues", [])
    payload.setdefault("fix_instruction", "")
    payload.setdefault("missing_info_status", [])
    payload.setdefault("confidence", "low")
    return payload


def _summarize_trace(steps: list[dict]) -> str:
    """Compact trace summary for the verifier — just action names and
    short observation excerpts to keep the prompt concise."""
    lines = []
    for s in steps[-12:]:  # last 12 steps is enough context
        action = s.get("action", "?")
        thought = (s.get("thought") or "")[:120]
        obs = s.get("observation") or {}
        ok = obs.get("ok", "?")
        content = obs.get("content")
        content_repr = ""
        if isinstance(content, dict):
            content_repr = json.dumps({k: str(v)[:80] for k, v in content.items()}, ensure_ascii=False)
        elif isinstance(content, str):
            content_repr = content[:120]
        lines.append(f"step {s.get('step_index','?')}: action={action} ok={ok} | thought={thought}")
        if content_repr:
            lines.append(f"  → {content_repr[:240]}")
    return "\n".join(lines)


def verify_answer(
    *,
    plan: dict[str, Any],
    answer: dict[str, Any],
    steps: list[dict],
    model: ModelAdapter,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run Phase 3 verification.

    Returns (verdict_dict, metadata).
    """
    import time

    trace_summary = _summarize_trace(steps)
    user_msg = (
        f"## Plan\n{json.dumps(plan, indent=2, ensure_ascii=False)}\n\n"
        f"## Candidate Answer\n{json.dumps(answer, indent=2, ensure_ascii=False)[:4000]}\n\n"
        f"## Trace summary (last 12 steps)\n{trace_summary[:3000]}\n\n"
        f"Verify and output JSON inside a ```json fence."
    )
    messages = [
        ModelMessage(role="system", content=PHASE3_SYSTEM_PROMPT),
        ModelMessage(role="user", content=user_msg),
    ]
    t0 = time.time()
    # Phase 3 thinking budget: same VERY generous ~192K tokens as Phase 1.
    # The verifier needs to reason over plan + answer + trace summary to
    # detect subtle calculation / scope / interpretation issues. Better to
    # over-allocate than to starve thinking and get an empty verdict.
    raw = model.complete(messages, enable_thinking=True, max_tokens=196608)
    dt = time.time() - t0

    parse_error: str | None = None
    try:
        verdict = parse_verdict_json(raw)
    except Exception as exc:
        parse_error = repr(exc)
        verdict = parse_verdict_json('{"verdict":"OK","issues":["verifier parse failed"],"confidence":"low"}')

    meta = {
        "phase": "verify",
        "latency_s": round(dt, 2),
        "raw_excerpt": raw[:1500],
        "parse_error": parse_error,
        "verdict": verdict.get("verdict"),
        "confidence": verdict.get("confidence"),
        "n_issues": len(verdict.get("issues") or []),
    }
    return verdict, meta
