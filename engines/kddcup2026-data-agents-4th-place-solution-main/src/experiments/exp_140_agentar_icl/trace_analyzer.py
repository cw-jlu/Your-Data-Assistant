"""Trace analyzer sub-agent — diagnose why a previous answer might be wrong
and propose an alternative direction for the next attempt.

Used in retry-with-analysis POC:
  Round 1 → standard agent.run()
  Analyzer → reads trace + answer, outputs ALT_DIRECTION
  Round 2 → agent.run() with ALT_DIRECTION injected in preamble
"""
from __future__ import annotations

import os
import re

from kobushi_core.model import ModelMessage, OpenAIModelAdapter


_ANALYZER_SYS = """You are a debugging assistant for a text-to-SQL agent.

The agent answered a question, but the answer may be wrong (= score below
threshold, or you observed inconsistency). Your job is to identify ONE
alternative interpretation or approach that the agent should try in a NEW
attempt.

Common failure patterns to consider:
  - COLUMN AMBIGUITY: agent picked a semantically-related column when the
    GOLD wants the literal column name match (e.g. "type" → column `type`,
    not `category` or `description`).
  - VALUE LITERAL: agent treated a proper noun as a category set when GOLD
    wants the literal cell value (e.g. "European Grand Prix" → race name
    literal, not "races in European countries").
  - AGGREGATION FORMULA: agent used SUM where AVG is needed, or missed a
    *100/N step in percentage / monthly average calculations.
  - FILTER STRICTNESS: agent's WHERE clause was too strict (= missed rows)
    or too loose (= extra rows).
  - COLUMN COUNT EXTRAS: agent returned X+Y columns when GOLD wants only X
    (= no explanation columns, only the asked-for value).
  - DIFFERENT TABLE / DATA SOURCE: agent looked in the wrong table; another
    table might hold the answer (= e.g. yearmonth vs transactions).

Examine the trace + final answer + question, identify ONE specific
alternative direction, and output:

ALT_DIRECTION: <one specific, actionable alternative for the next agent>
RATIONALE: <one sentence why this alternative might match the gold>

Be CONCRETE. Mention specific column names, filter conditions, or
aggregation patterns from the schema. Don't be vague.

If the answer looks correct to you, output:
ALT_DIRECTION: (none — answer looks right)
RATIONALE: (no obvious issue)

Output ONLY the two lines above. No markdown, no preamble, no JSON."""


_ALT_RE = re.compile(r"ALT_DIRECTION\s*:\s*(.+?)(?=\nRATIONALE|\Z)", re.DOTALL)
_RAT_RE = re.compile(r"RATIONALE\s*:\s*(.+?)$", re.DOTALL)


def _make_model() -> OpenAIModelAdapter:
    return OpenAIModelAdapter(
        model=os.environ.get("AGENT_MODEL", "qwen3.5-35b-a3b"),
        api_base=os.environ.get("AGENT_API_BASE") or os.environ.get("MODEL_API_URL"),
        api_key=os.environ.get("AGENT_API_KEY") or os.environ.get("MODEL_API_KEY"),
        temperature=0.0,
        extra_headers={
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
    )


def _trim_trace(trace_text: str, max_chars: int = 6000) -> str:
    """Keep the head + tail of the trace; drop the middle if too long."""
    if len(trace_text) <= max_chars:
        return trace_text
    half = max_chars // 2 - 50
    return trace_text[:half] + "\n... [truncated middle] ...\n" + trace_text[-half:]


def analyze_attempt(
    question: str,
    trace_text: str,
    prediction_preview: str,
    model: OpenAIModelAdapter | None = None,
) -> dict:
    """Run the analyzer sub-agent. Returns {alt_direction, rationale, raw_reply}."""
    if model is None:
        model = _make_model()
    trace = _trim_trace(trace_text or "(trace empty)")
    user_msg = (
        f"# Question\n{question}\n\n"
        f"# Agent's reasoning trace (= key steps)\n```\n{trace}\n```\n\n"
        f"# Agent's final answer\n```\n{prediction_preview}\n```\n\n"
        f"# Your output (= EXACTLY two lines, ALT_DIRECTION + RATIONALE)"
    )
    try:
        reply = model.complete(
            [
                ModelMessage(role="system", content=_ANALYZER_SYS),
                ModelMessage(role="user", content=user_msg),
            ],
            enable_thinking=False,
            max_tokens=512,
        )
    except Exception as exc:
        return {"alt_direction": "", "rationale": "", "raw_reply": f"error: {exc}"}
    alt_m = _ALT_RE.search(reply or "")
    rat_m = _RAT_RE.search(reply or "")
    alt = alt_m.group(1).strip() if alt_m else ""
    rat = rat_m.group(1).strip() if rat_m else ""
    return {"alt_direction": alt, "rationale": rat, "raw_reply": (reply or "")[:1000]}


def build_retry_hint_block(analysis: dict) -> str | None:
    """Format analyzer output as a preamble block for round 2 agent.

    Returns None if analyzer said "(none)" / empty alt direction.
    """
    alt = analysis.get("alt_direction") or ""
    if not alt or "none" in alt.lower()[:10]:
        return None
    rat = analysis.get("rationale") or ""
    return (
        "# RETRY HINT — PRIOR ATTEMPT REVIEW\n"
        "A previous attempt at this task produced an answer that may be wrong.\n"
        "A debugging analyzer reviewed the agent's trace and identified an\n"
        "alternative direction worth trying. Strongly prefer this alternative\n"
        "unless your exploration clearly disproves it.\n\n"
        f"## Alternative direction\n{alt}\n\n"
        f"## Why\n{rat}\n"
    )
