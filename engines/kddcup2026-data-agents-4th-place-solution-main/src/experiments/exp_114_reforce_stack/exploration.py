"""Column exploration pre-phase (= ReFoRCE pillar (d)).

Before the main ReAct agent loop, run an exploration phase:
1. Send (question + view catalog + knowledge.md slice) to the LLM
2. Ask it to generate 3-10 DISTINCT/COUNT/range queries that would help
   understand column values, value distributions, and key cardinalities
3. Execute the queries against the unified DuckDB connection
4. Format results as a "few-shot examples" block to inject into the
   main agent's preamble

The intent is to surface in advance:
- distinct values of categorical columns (= disambiguates aggregate axis)
- value distributions of numeric columns (= sanity check magnitudes)
- count of rows per filter (= sanity check filter selectivity)
- format of date/text columns (= avoid format mismatch errors)

These directly target our known failure modes: aggregate-axis miss
(task_163), formula scope ambiguity (task_169), filter complexity (task_180).
"""
from __future__ import annotations

import re
from typing import Any

from kobushi_core.benchmark.schema import PublicTask
from kobushi_core.model import ModelAdapter, ModelMessage

from experiments.exp_114_reforce_stack.tools.duckdb_unified import (
    describe_catalog,
    execute_sql,
)


EXPLORATION_SYSTEM_PROMPT = """\
You are an SQL exploration helper. The orchestrator agent is about to answer a
data question. BEFORE the main query, you generate 3 to 8 short exploratory
DuckDB SELECT queries that surface the information the orchestrator needs to
phrase a correct answer.

What a good exploration query looks like:
- `SELECT DISTINCT type, COUNT(*) FROM expense GROUP BY type` — discover the
  category dimension if the question filters by some textual class.
- `SELECT MIN(date), MAX(date), COUNT(*) FROM transactions` — confirm the
  date range covers the question's filter.
- `SELECT COUNT(*) FROM atom WHERE element IN ('p', 'br')` — verify a filter
  matches non-zero rows before relying on it.
- `SELECT DISTINCT date_format(date, '%Y-%m'), COUNT(*) FROM trans GROUP BY 1
  ORDER BY 1 LIMIT 5` — reveal date format / granularity.

What NOT to do:
- Don't try to answer the question itself. That's the orchestrator's job.
- Don't return raw rows from a base table without aggregation (= wasteful).
- Don't write more than 8 queries.

Output format: a JSON array of strings, each string a single SELECT statement.
Wrap in ```json fence. Example:
```json
[
  "SELECT DISTINCT type FROM expense LIMIT 20",
  "SELECT COUNT(*) FROM expense",
  "SELECT DISTINCT segment FROM customers"
]
```
No surrounding text.
"""


def _knowledge_slice(task: PublicTask, max_chars: int = 4000) -> str:
    p = task.context_dir / "knowledge.md"
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8", errors="replace")
    return text[:max_chars] + ("\n[...truncated]" if len(text) > max_chars else "")


def _extract_queries(raw: str) -> list[str]:
    import json as _json
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    payload = m.group(1).strip() if m else raw.strip()
    try:
        arr = _json.loads(payload)
        if isinstance(arr, list):
            return [str(q).strip() for q in arr if isinstance(q, str) and q.strip()]
    except Exception:
        pass
    # Fallback: extract individual SELECT/WITH statements line-wise
    candidates: list[str] = []
    for line in payload.split("\n"):
        stripped = line.strip().rstrip(",").strip().strip('"\'')
        if stripped.lower().startswith(("select", "with")):
            candidates.append(stripped)
    return candidates


def _format_result(sql: str, result: dict[str, Any]) -> str:
    cols = result.get("columns") or []
    rows = result.get("rows") or []
    if not cols:
        return f"-- {sql}\n(no columns returned)\n"
    truncated = result.get("truncated", False)
    sample = rows[:8]
    lines = [f"-- {sql}", "| " + " | ".join(cols) + " |"]
    for row in sample:
        lines.append("| " + " | ".join("" if v is None else str(v)[:60] for v in row) + " |")
    suffix = f"\n  ({len(rows)} rows{', truncated' if truncated else ''})"
    return "\n".join(lines) + suffix + "\n"


def run_exploration_phase(
    *,
    task: PublicTask,
    model: ModelAdapter,
    max_queries: int = 8,
    max_total_chars: int = 8000,
) -> dict[str, Any]:
    """Run the LLM-guided exploration phase. Returns formatted findings."""
    catalog = describe_catalog(task.context_dir)
    knowledge = _knowledge_slice(task)
    user_prompt = (
        f"## Question\n{task.question}\n\n"
        f"## DuckDB views available\n{catalog}\n\n"
        f"## knowledge.md (= excerpt)\n{knowledge}\n\n"
        "Output ONE ```json array of 3-8 short exploratory SELECT statements."
    )
    messages = [
        ModelMessage(role="system", content=EXPLORATION_SYSTEM_PROMPT),
        ModelMessage(role="user", content=user_prompt),
    ]
    raw = model.complete(messages)
    queries = _extract_queries(raw)[:max_queries]

    findings: list[str] = []
    successful = 0
    failed = 0
    total_chars = 0
    for sql in queries:
        try:
            result = execute_sql(task.context_dir, sql, limit=50)
            piece = _format_result(sql, result)
            successful += 1
        except Exception as exc:
            piece = f"-- {sql}\nERROR: {type(exc).__name__}: {str(exc)[:200]}\n"
            failed += 1
        if total_chars + len(piece) > max_total_chars:
            findings.append("...[remaining queries truncated due to budget]")
            break
        findings.append(piece)
        total_chars += len(piece)

    block = (
        "# Column exploration findings (= run BEFORE main answer)\n"
        "These are sanity-check queries already executed against the actual data. "
        "Use them to confirm filter values, aggregate axes, value formats, and row "
        "cardinalities before writing the main answer SQL.\n\n"
        + "\n".join(findings)
    ) if findings else ""

    return {
        "block": block,
        "raw_response": raw,
        "queries_attempted": len(queries),
        "successful": successful,
        "failed": failed,
    }
